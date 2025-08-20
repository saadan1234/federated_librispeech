#!/usr/bin/env python3
"""
Simplified Federated HuBERT pretraining with Flower (FedAdam).
Core flow:
- HuBERT-like transformer model with sinusoidal positional encoding and frame-level masking
- LibriSpeech-style dataset loader, MFCC + KMeans pseudo-labels, span masking at frame granularity
- Flower NumPyClient for fit/evaluate; federated aggregation with FedAdam
"""

import os
import logging
import time
import signal
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import yaml
import argparse
from torch.utils.data import DataLoader, Dataset

from flwr.common.typing import NDArrays, Config
from flwr.client import NumPyClient, ClientApp
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

# Global config path variable
CLIENT_CONFIG_PATH = None
# Global config variable to store overridden configuration
GLOBAL_CONFIG = None

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown signal received, cleaning up...")
    sys.exit(0)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to inputs."""

    def __init__(self, hidden_size: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, hidden_size, 2)
                             * (-np.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, hidden_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        t = x.size(1)
        return x + self.pe[:t, :]


# No dynamic KMeans/centroid usage in training for privacy


class HubertBase(nn.Module):
    """Tiny HuBERT-like model for pretraining with frame-rate outputs."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12, vocab_size: int = 504, frame_stride: int = 320):
        super().__init__()
        logger.info(
            f"Initializing HubertBase model with hidden_size={hidden_size}, layers={num_layers}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.frame_stride = frame_stride

        # Transformer layers with proper configuration
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,  # Important for proper input format
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

        logger.info(
            f"HubertBase model created with {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def forward(self, input_values: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # input_values: [batch, samples]
        # Project to hidden and downsample to frame rate using average pooling
        x = input_values.unsqueeze(-1)                      # [B, T, 1]
        x = self.input_projection(x)                        # [B, T, H]
        x = x.transpose(1, 2)                               # [B, H, T]
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)                               # [B, T_frames, H]

        # Apply mask embedding to masked frame positions if provided
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)        # [B, T_frames, 1]
            x = torch.where(
                mask_expanded,
                self.mask_embedding.view(1, 1, -1).expand_as(x),
                x,
            )

        # Positional encoding and Transformer encoder
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)

        # Project to vocab
        # [B, T_frames, vocab]
        logits = self.output_projection(x)
        return {"predictions": logits}


class LibriSpeechPretrainingDataset(Dataset):
    """Dataset for LibriSpeech pretraining with span masking at frame level."""

    def __init__(
        self,
        manifest_file: str,
        audio_root: str,
        split: str = "train",  # "train" or "validation"
        max_length: int = 40000,  # 2.5 seconds at 16kHz
        sample_rate: int = 16000,
        mask_prob: float = 0.08,
        mask_length: int = 10,
        vocab_size: int = 504,
        kmeans_targets_path: Optional[str] = None,
    ):
        # Read manifest (keep both full and filtered views to align targets)
        df_all = pd.read_csv(manifest_file)
        logger.info(f"Loaded manifest with {len(df_all)} samples")

        self.audio_root = Path(audio_root)
        self.split = split
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.vocab_size = vocab_size

        # Compute filtered dataframe and index mapping to original rows
        if 'split' in df_all.columns:
            original_indices = df_all.index[df_all['split'] == split].to_list()
            self.df = df_all.loc[original_indices].reset_index(drop=True)
            self._orig_indices = original_indices
            logger.info(f"Filtered to {split} split: {len(self.df)} samples")
        else:
            self.df = df_all.reset_index(drop=True)
            self._orig_indices = list(range(len(self.df)))

        # Frame setup
        self.frame_stride = 320
        self.vocab_size = vocab_size

        # Optional precomputed per-client KMeans targets (preferred for privacy)
        self.precomputed_targets: Optional[List[np.ndarray]] = None
        if kmeans_targets_path is not None and Path(kmeans_targets_path).exists():
            loaded = np.load(kmeans_targets_path, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype != object:
                all_targets = [row for row in loaded]
            else:
                all_targets = list(loaded)
            # Align targets to the current split using original indices
            self.precomputed_targets = [all_targets[i]
                                        for i in self._orig_indices]
            logger.info(
                f"Loaded and aligned precomputed KMeans targets from {kmeans_targets_path}")
        else:
            raise FileNotFoundError(
                "kmeans_targets.npy not found; training requires per-client precomputed frame targets")

        logger.info(f"Dataset initialized with {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Generate boolean span mask over a 1D sequence of length seq_len."""
        mask = torch.zeros(seq_len, dtype=torch.bool)

        # Generate random spans to mask
        num_spans = int(seq_len * self.mask_prob / self.mask_length)
        for _ in range(num_spans):
            start = torch.randint(
                0, seq_len - self.mask_length + 1, (1,)).item()
            mask[start:start + self.mask_length] = True

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with masking"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                row = self.df.iloc[idx]
                if 'audio_file' in row:
                    audio_rel = row['audio_file']
                elif 'audio_path' in row:
                    audio_rel = row['audio_path']
                else:
                    raise KeyError(
                        "Manifest row must contain 'audio_file' or 'audio_path'")
                audio_path = self.audio_root / audio_rel

                # Load audio
                audio, sr = sf.read(str(audio_path))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Take first channel if stereo

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = T.Resample(
                        orig_freq=sr, new_freq=self.sample_rate)
                    audio = resampler(torch.tensor(audio, dtype=torch.float32))
                else:
                    audio = torch.tensor(audio, dtype=torch.float32)

                # Truncate or pad to max_length
                if len(audio) > self.max_length:
                    start = torch.randint(
                        0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    # Pad with zeros
                    padding = self.max_length - len(audio)
                    audio = F.pad(audio, (0, padding))

                # Determine model frame sequence length
                target_seq_len = len(audio) // self.frame_stride

                # Frame-level mask
                mask = self._span_mask(target_seq_len)

                # Use precomputed per-client targets for privacy
                if self.precomputed_targets is not None:
                    arr = self.precomputed_targets[idx]
                    arr_t = torch.as_tensor(arr, dtype=torch.long)
                    if arr_t.numel() >= target_seq_len:
                        targets = arr_t[:target_seq_len]
                    else:
                        pad = torch.zeros(
                            target_seq_len - arr_t.numel(), dtype=torch.long)
                        targets = torch.cat([arr_t, pad], dim=0)
                else:
                    raise RuntimeError(
                        "Precomputed targets missing for sample")

                # Pad targets if needed
                if len(targets) < target_seq_len:
                    padding = target_seq_len - len(targets)
                    targets = F.pad(targets, (0, padding), value=0)

                # Debug logging for first few samples (only show shapes, not full tensors)
                if idx < 3:
                    logger.info(f"Dataset sample {idx}: audio_shape={audio.shape}, "
                                f"targets_shape={targets.shape}, mask_shape={mask.shape}")
                    # Log mask statistics instead of full content
                    logger.info(
                        f"Dataset sample {idx}: mask has {mask.sum().item()} True values out of {mask.numel()}")

                result = {
                    "input_values": audio,
                    "targets": targets,
                    "mask": mask
                }

                # Verify result structure
                assert isinstance(
                    result, dict), f"Result is not a dict: {type(result)}"
                assert 'input_values' in result, f"Missing input_values in result: {result.keys()}"
                assert 'targets' in result, f"Missing targets in result: {result.keys()}"
                assert 'mask' in result, f"Missing mask in result: {result.keys()}"

                return result

            except Exception as e:
                logger.warning(
                    f"Failed to load sample {idx} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Return fallback data with minimal logging
                    fallback_audio = torch.zeros(
                        self.max_length, dtype=torch.float32)
                    fallback_targets = torch.zeros(
                        self.max_length // self.frame_stride, dtype=torch.long)
                    fallback_mask = torch.zeros(
                        self.max_length // self.frame_stride, dtype=torch.bool)

                    fallback_result = {
                        "input_values": fallback_audio,
                        "targets": fallback_targets,
                        "mask": fallback_mask
                    }

                    logger.warning(
                        f"Returning fallback data for sample {idx} after {max_retries} failed attempts")
                    return fallback_result

        raise RuntimeError(
            f"Failed to load sample after {max_retries} attempts")


def custom_collate_fn(batch):
    """Custom collate function to ensure proper dictionary batching."""
    import logging
    logger = logging.getLogger(__name__)

    # Ensure batch is a list of dictionaries
    if isinstance(batch, tuple):
        batch = list(batch)

    # Extract each component
    try:
        input_values = [item['input_values'] for item in batch]
        targets = [item['targets'] for item in batch]
        masks = [item['mask'] for item in batch]

        # Stack tensors
        input_values = torch.stack(input_values, dim=0)
        targets = torch.stack(targets, dim=0)
        masks = torch.stack(masks, dim=0)

        result = {
            'input_values': input_values,
            'targets': targets,
            'mask': masks
        }

        return result

    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {e}")
        logger.error(f"Batch content: {batch}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# Checkpointing removed for simplicity


class FederatedClient(NumPyClient):
    """Federated client with HubertBase model using real data"""

    def __init__(self, client_id: int, train_dataset, val_dataset, model, device=None):
        """Initialize the client with datasets and model."""
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

        # Better device detection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(
                    f"Client {client_id}: Using CUDA device: {torch.cuda.get_device_name()}")
                # Set memory fraction to avoid OOM
                torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                self.device = torch.device("cpu")
                logger.warning(
                    f"Client {client_id}: CUDA not available, using CPU")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Enable mixed precision if available and on GPU
        if self.device.type == "cuda" and hasattr(torch, 'amp'):
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(
                f"Client {client_id}: Enabled mixed precision training")
        else:
            self.use_amp = False
            self.scaler = None
            logger.info(f"Client {client_id}: Mixed precision not available")

        # Load config to get batch size and other parameters
        config_path = Path(
            "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Get batch size from config, with fallback to small value for testing
        batch_size = cfg.get('pretraining', {}).get('batch_size', 1)
        num_workers = cfg.get('pretraining', {}).get('num_workers', 4)
        pin_memory = cfg.get('pretraining', {}).get('pin_memory', False)

        logger.info(
            f"Client {client_id}: Using batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

        # Optimize DataLoader for better performance
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,  # Increased batch size
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory if self.device.type == "cuda" else False,
            drop_last=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory if self.device.type == "cuda" else False,
            drop_last=False,
            persistent_workers=True
        )

    def get_parameters(self, config: Config) -> NDArrays:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """Train the model using the provided parameters."""
        import time

        start_time = time.time()

        try:
            logger.info(
                f"Client {self.client_id}: Starting fit method at {time.strftime('%H:%M:%S')}")

            # Set parameters with timing
            param_start = time.time()
            self.set_parameters(parameters)
            param_time = time.time() - param_start
            logger.info(
                f"Client {self.client_id}: Parameters set in {param_time:.2f}s")

            # Setup optimizer with timing
            opt_start = time.time()
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=5e-4, weight_decay=0.01)
            opt_time = time.time() - opt_start
            logger.info(
                f"Client {self.client_id}: Optimizer setup in {opt_time:.2f}s")

            # Training loop
            self.model.train()
            total_loss = 0.0
            num_samples = 0

            # Use server-provided hyperparameters if available; fallback to config file
            cfg_start = time.time()
            cfg_path = Path(
                "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            pre_cfg = cfg.get('pretraining', {})
            cfg_time = time.time() - cfg_start
            logger.info(
                f"Client {self.client_id}: Config loaded in {cfg_time:.2f}s")

            # Add gradient accumulation for effective larger batch size
            accumulation_steps = pre_cfg.get(
                'gradient_accumulation_steps', 4)  # Get from config
            optimizer.zero_grad()

            local_epochs = int(config.get("local_epochs", pre_cfg.get('local_epochs', pre_cfg.get('epochs', 1)))) if isinstance(
                config, dict) else int(pre_cfg.get('local_epochs', pre_cfg.get('epochs', 1)))
            learning_rate = float(config.get("lr", pre_cfg.get('learning_rate', 5e-4))
                                  ) if isinstance(config, dict) else float(pre_cfg.get('learning_rate', 5e-4))

            logger.info(
                f"Client {self.client_id}: Training config - epochs: {local_epochs}, lr: {learning_rate}")

            for g in optimizer.param_groups:
                g["lr"] = learning_rate

            # Get dataset info
            logger.info(
                f"Client {self.client_id}: Dataset size - train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")
            logger.info(
                f"Client {self.client_id}: DataLoader batch size: {self.train_loader.batch_size}")
            logger.info(
                f"Client {self.client_id}: DataLoader num workers: {self.train_loader.num_workers}")

            for epoch in range(local_epochs):
                epoch_start = time.time()
                epoch_loss = 0.0

                # Add progress indicator for simulation mode
                if CLIENT_CONFIG_PATH:  # Simulation mode
                    print(
                        f"🔄 Client {self.client_id}: Training Epoch {epoch + 1}/{local_epochs}")
                else:
                    logger.info(
                        f"Client {self.client_id}: Starting epoch {epoch + 1}/{local_epochs}")
                epoch_samples = 0

                # Add progress bar for batches
                from tqdm import tqdm
                batch_iterator = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f"Client {self.client_id} Epoch {epoch + 1}",
                    leave=False
                )

                for batch_idx, batch in enumerate(batch_iterator):
                    batch_start = time.time()

                    try:
                        # Handle batch format - could be tuple or dict
                        if isinstance(batch, tuple):
                            # If batch is a tuple, unpack it
                            if len(batch) == 3:
                                input_values, targets, mask = batch
                            elif len(batch) == 2:
                                # Handle (index, data) format from DataLoader
                                batch_idx_from_tuple, batch_data = batch
                                if isinstance(batch_data, dict):
                                    input_values = batch_data['input_values']
                                    targets = batch_data['targets']
                                    mask = batch_data['mask']
                                else:
                                    raise ValueError(
                                        f"Unexpected batch data type: {type(batch_data)}")
                            elif len(batch) == 1:
                                # Single item tuple containing dict
                                batch_dict = batch[0]
                                input_values = batch_dict['input_values']
                                targets = batch_dict['targets']
                                mask = batch_dict['mask']
                            else:
                                raise ValueError(
                                    f"Unexpected batch tuple length: {len(batch)}")
                        elif isinstance(batch, dict):
                            # If batch is already a dict
                            input_values = batch['input_values']
                            targets = batch['targets']
                            mask = batch['mask']
                        else:
                            # Fallback: try to extract data from any format
                            logger.warning(
                                f"Client {self.client_id}: Unexpected batch type {type(batch)}, attempting fallback extraction")

                            # Try to find tensors in the batch
                            tensors = []
                            for item in batch if hasattr(batch, '__iter__') else [batch]:
                                if isinstance(item, torch.Tensor):
                                    tensors.append(item)

                            if len(tensors) >= 3:
                                # Assume order: input_values, targets, mask
                                input_values, targets, mask = tensors[:3]
                                logger.info(
                                    f"Client {self.client_id}: Fallback extraction successful from {len(tensors)} tensors")
                            else:
                                raise ValueError(
                                    f"Fallback extraction failed: found {len(tensors)} tensors, need at least 3")

                        # Additional validation
                        if not isinstance(input_values, torch.Tensor):
                            raise ValueError(
                                f"input_values is not a tensor: {type(input_values)}")
                        if not isinstance(targets, torch.Tensor):
                            raise ValueError(
                                f"targets is not a tensor: {type(targets)}")
                        if not isinstance(mask, torch.Tensor):
                            raise ValueError(
                                f"mask is not a tensor: {type(mask)}")

                        logger.debug(f"Client {self.client_id}: Batch {batch_idx} - input_values: {input_values.shape}, "
                                     f"targets: {targets.shape}, mask: {mask.shape}")

                        # Data transfer to device
                        data_start = time.time()
                        input_values = input_values.to(self.device)
                        targets = targets.to(self.device)
                        mask = mask.to(self.device)
                        data_time = time.time() - data_start

                        # Update progress bar with timing info
                        batch_iterator.set_postfix({
                            'data_time': f'{data_time:.3f}s',
                            'loss': f'{epoch_loss/max(1, batch_idx):.4f}' if batch_idx > 0 else 'N/A'
                        })

                        optimizer.zero_grad()

                        # Forward pass with timing
                        forward_start = time.time()

                        if self.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(
                                    input_values, frame_mask=mask)
                                predictions = outputs['predictions']
                        else:
                            outputs = self.model(input_values, frame_mask=mask)
                            predictions = outputs['predictions']

                        forward_time = time.time() - forward_start

                        # Apply frame-level mask and compute loss across batch/time
                        batch_size, seq_len, vocab_size = predictions.size()
                        predictions_flat = predictions.view(
                            batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        mask_flat = mask.view(batch_size * seq_len)

                        if mask_flat.any():
                            loss_start = time.time()
                            loss = F.cross_entropy(
                                predictions_flat[mask_flat],
                                targets_flat[mask_flat]
                            )
                            loss_time = time.time() - loss_start

                            # Scale loss for gradient accumulation
                            loss = loss / accumulation_steps

                            # Backward pass with timing
                            backward_start = time.time()

                            if self.use_amp:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                            backward_time = time.time() - backward_start

                            # Optimizer step after accumulation
                            step_time = 0.0  # Initialize step_time
                            if (batch_idx + 1) % accumulation_steps == 0:
                                step_start = time.time()

                                if self.use_amp:
                                    self.scaler.step(optimizer)
                                    self.scaler.update()
                                else:
                                    optimizer.step()

                                optimizer.zero_grad()
                                step_time = time.time() - step_start

                            epoch_loss += loss.item() * accumulation_steps
                            epoch_samples += input_values.size(0)

                            # Log detailed timing for first few batches
                            if batch_idx < 3:
                                logger.info(f"Client {self.client_id}: Batch {batch_idx} timing - "
                                            f"data: {data_time:.3f}s, forward: {forward_time:.3f}s, "
                                            f"loss: {loss_time:.3f}s, backward: {backward_time:.3f}s, "
                                            f"step: {step_time:.3f}s, total: {time.time() - batch_start:.3f}s")
                        else:
                            # Skip batches with no masked frames instead of warning
                            continue

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in batch {batch_idx}: {e}")
                        logger.error(
                            f"Client {self.client_id}: Batch type: {type(batch)}")
                        # Don't log the full batch content to avoid verbose output
                        raise

                epoch_time = time.time() - epoch_start
                avg_epoch_loss = epoch_loss / max(1, len(self.train_loader))

                logger.info(
                    f"Client {self.client_id}: Epoch {epoch + 1}/{local_epochs} completed in {epoch_time:.2f}s, "
                    f"Loss = {avg_epoch_loss:.4f}, Samples = {epoch_samples}")

                total_loss += epoch_loss
                num_samples += epoch_samples

            # Training completion
            total_time = time.time() - start_time
            avg_loss = total_loss / \
                max(1, local_epochs * len(self.train_loader))

            logger.info(
                f"Client {self.client_id}: Training completed in {total_time:.2f}s, "
                f"avg_loss={avg_loss:.4f}, total_samples={num_samples}")

            # Get parameters with timing
            param_get_start = time.time()
            final_params = self.get_parameters(config={})
            param_get_time = time.time() - param_get_start
            logger.info(
                f"Client {self.client_id}: Parameters retrieved in {param_get_time:.2f}s")

            return final_params, num_samples, {"pretrain_loss": avg_loss}

        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Training failed with error: {e}")
            raise

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model using the provided parameters."""
        import time
        eval_start = time.time()

        # Add progress indicator for simulation mode
        if CLIENT_CONFIG_PATH:  # Simulation mode
            print(f"🔍 Client {self.client_id}: Starting evaluation...")
        else:
            logger.info(
                f"Client {self.client_id}: Starting evaluation at {time.strftime('%H:%M:%S')}")

        # Set parameters with timing
        param_start = time.time()
        self.set_parameters(parameters)
        param_time = time.time() - param_start

        if CLIENT_CONFIG_PATH:  # Simulation mode
            print(
                f"⚙️  Client {self.client_id}: Parameters loaded in {param_time:.2f}s")
        else:
            logger.info(
                f"Client {self.client_id}: Evaluation parameters set in {param_time:.2f}s")

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0

        if CLIENT_CONFIG_PATH:  # Simulation mode
            print(
                f"📊 Client {self.client_id}: Evaluating {len(self.val_loader)} batches...")
        else:
            logger.info(
                f"Client {self.client_id}: Starting validation loop with {len(self.val_loader)} batches")

        with torch.no_grad():
            # Add progress bar for evaluation
            from tqdm import tqdm
            eval_iterator = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"Client {self.client_id} Evaluation",
                leave=False
            )

            for batch_idx, batch in eval_iterator:
                batch_start = time.time()

                try:
                    # Handle batch format - could be tuple or dict
                    if isinstance(batch, tuple):
                        # If batch is a tuple, unpack it
                        if len(batch) == 3:
                            input_values, targets, mask = batch
                        elif len(batch) == 2:
                            # Handle (index, data) format from DataLoader
                            batch_idx_from_tuple, batch_data = batch
                            if isinstance(batch_data, dict):
                                input_values = batch_data['input_values']
                                targets = batch_data['targets']
                                mask = batch_data['mask']
                            else:
                                raise ValueError(
                                    f"Unexpected batch data type: {type(batch_data)}")
                        elif len(batch) == 1:
                            # Single item tuple containing dict
                            batch_dict = batch[0]
                            input_values = batch_dict['input_values']
                            targets = batch_dict['targets']
                            mask = batch_dict['mask']
                        else:
                            raise ValueError(
                                f"Unexpected batch tuple length: {len(batch)}")
                    elif isinstance(batch, dict):
                        # If batch is already a dict
                        input_values = batch['input_values']
                        targets = batch['targets']
                        mask = batch['mask']
                    else:
                        # Fallback: try to extract data from any format
                        logger.warning(
                            f"Client {self.client_id}: Unexpected eval batch type {type(batch)}, attempting fallback extraction")

                        # Try to find tensors in the batch
                        tensors = []
                        for item in batch if hasattr(batch, '__iter__') else [batch]:
                            if isinstance(item, torch.Tensor):
                                tensors.append(item)

                        if len(tensors) >= 3:
                            # Assume order: input_values, targets, mask
                            input_values, targets, mask = tensors[:3]
                            logger.info(
                                f"Client {self.client_id}: Eval fallback extraction successful from {len(tensors)} tensors")
                        else:
                            raise ValueError(
                                f"Eval fallback extraction failed: found {len(tensors)} tensors, need at least 3")

                    # Additional validation
                    if not isinstance(input_values, torch.Tensor):
                        raise ValueError(
                            f"input_values is not a tensor: {type(input_values)}")
                    if not isinstance(targets, torch.Tensor):
                        raise ValueError(
                            f"targets is not a tensor: {type(targets)}")
                    if not isinstance(mask, torch.Tensor):
                        raise ValueError(f"mask is not a tensor: {type(mask)}")

                    logger.debug(f"Client {self.client_id}: Eval batch {batch_idx} - input_values: {input_values.shape}, "
                                 f"targets: {targets.shape}, mask: {mask.shape}")

                    # Data transfer to device
                    data_start = time.time()
                    input_values = input_values.to(self.device)
                    targets = targets.to(self.device)
                    mask = mask.to(self.device)
                    data_time = time.time() - data_start

                    # Forward pass with timing
                    forward_start = time.time()
                    outputs = self.model(input_values, frame_mask=mask)
                    forward_time = time.time() - forward_start

                    # Compute masked loss and accuracy
                    predictions = outputs['predictions']  # [B, T, V]
                    bsz, tlen, vsize = predictions.size()
                    predictions_flat = predictions.view(bsz * tlen, vsize)
                    targets_flat = targets.view(bsz * tlen)
                    mask_flat = mask.view(bsz * tlen)

                    if mask_flat.any():
                        loss_start = time.time()
                        loss = F.cross_entropy(
                            predictions_flat[mask_flat], targets_flat[mask_flat])
                        loss_time = time.time() - loss_start

                        acc_start = time.time()
                        preds = torch.argmax(
                            predictions_flat[mask_flat], dim=-1)
                        accuracy = (
                            preds == targets_flat[mask_flat]).float().mean()
                        acc_time = time.time() - acc_start
                    else:
                        # Fallback if no masked frames present
                        loss_start = time.time()
                        loss = F.cross_entropy(predictions_flat, targets_flat)
                        loss_time = time.time() - loss_start

                        acc_start = time.time()
                        preds = torch.argmax(predictions_flat, dim=-1)
                        accuracy = (preds == targets_flat).float().mean()
                        acc_time = time.time() - acc_start

                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_samples += input_values.size(0)

                    # Update progress bar
                    eval_iterator.set_postfix({
                        'loss': f'{total_loss/(batch_idx+1):.4f}',
                        'acc': f'{total_accuracy/(batch_idx+1):.4f}',
                        'data_time': f'{data_time:.3f}s',
                        'forward_time': f'{forward_time:.3f}s'
                    })

                    # Log detailed timing for first few batches
                    if batch_idx < 3:
                        logger.info(f"Client {self.client_id}: Eval batch {batch_idx} timing - "
                                    f"data: {data_time:.3f}s, forward: {forward_time:.3f}s, "
                                    f"loss: {loss_time:.3f}s, acc: {acc_time:.3f}s, "
                                    f"total: {time.time() - batch_start:.3f}s")

                except Exception as e:
                    logger.error(
                        f"Client {self.client_id}: Error in evaluation batch {batch_idx}: {e}")
                    logger.error(
                        f"Client {self.client_id}: Batch type: {type(batch)}")
                    logger.error(
                        f"Client {self.client_id}: Batch content: {batch}")
                    raise

        eval_time = time.time() - eval_start

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_accuracy = total_accuracy / max(1, len(self.val_loader))

        logger.info(
            f"Client {self.client_id}: Evaluation completed in {eval_time:.2f}s, "
            f"loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}, samples={num_samples}")

        return avg_loss, num_samples, {"accuracy": avg_accuracy}


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine the client ID based on the node ID
    node_id = context.node_id
    # Use a default number of clients if not specified in config
    num_clients = config.get('simulation', {}).get('num_supernodes', 2)
    client_id = hash(str(node_id)) % num_clients

    # Setup data path - use default if not specified in config
    data_root_config = config.get('data', {}).get(
        'partitioned_data_root', 'data/partitioned')
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)

    logger.info(f"Client {client_id}: Using data root: {data_root}")

    # Check if data root exists
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    client_data_path = data_root / f"client_{client_id}"

    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

    # First try to find manifest in client-specific directory
    manifest_path = client_data_path / "manifest.csv"
    if not manifest_path.exists():
        # Fallback to root directory manifest
        manifest_path = data_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found in either {client_data_path} or {data_root}")

    logger.info(
        f"Initializing client {client_id} with data from {client_data_path}")

    # Load config to align dataset/dataloader params
    cfg_start = time.time()
    cfg_path = Path(
        "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    pre_cfg = cfg.get('pretraining', {})
    dl_cfg = cfg.get('data', {}).get('dataloader', {})
    cfg_time = time.time() - cfg_start
    logger.info(
        f"Client {client_id}: Config loaded in {cfg_time:.2f}s")

    # Load manifest file
    manifest_start = time.time()
    df = pd.read_csv(manifest_path)
    manifest_time = time.time() - manifest_start
    logger.info(
        f"Client {client_id}: Manifest loaded in {manifest_time:.2f}s - {len(df)} samples")

    # Validate manifest structure
    if 'audio_file' not in df.columns and 'audio_path' not in df.columns:
        raise ValueError(
            f"Manifest missing required columns. Available columns: {list(df.columns)}")

    # Check first few audio paths to ensure they're relative
    audio_col = 'audio_file' if 'audio_file' in df.columns else 'audio_path'
    sample_paths = df[audio_col].head(3).tolist()
    logger.info(f"Client {client_id}: Sample audio paths: {sample_paths}")

    # Verify that audio files exist
    sample_audio_path = client_data_path / \
        sample_paths[0] if sample_paths else None
    if sample_audio_path and not sample_audio_path.exists():
        logger.warning(
            f"Client {client_id}: Sample audio file not found: {sample_audio_path}")
        logger.warning(
            f"Client {client_id}: Client data path: {client_data_path}")
        # Don't list all files to avoid verbose output
        logger.warning(
            f"Client {client_id}: Please check if audio files exist in the expected directory structure")

    # Optional precomputed per-client targets
    targets_start = time.time()
    # Look for targets in client-specific directory first, then fallback to root
    targets_path = client_data_path / "kmeans_targets.npy"
    if not targets_path.exists():
        targets_path = data_root / "kmeans_targets.npy"

    kmeans_targets_str = str(targets_path) if targets_path.exists() else None

    if kmeans_targets_str:
        targets_data = np.load(targets_path, allow_pickle=True)
        logger.info(
            f"Client {client_id}: KMeans targets loaded from {targets_path} - {len(targets_data)} sequences")
    else:
        logger.warning(
            f"Client {client_id}: No KMeans targets found in either {client_data_path} or {data_root}")

    targets_time = time.time() - targets_start
    logger.info(
        f"Client {client_id}: Targets processing completed in {targets_time:.2f}s")

    # Create train dataset with reduced sequence length
    train_dataset_start = time.time()

    # Create train dataset with reduced sequence length
    train_dataset = LibriSpeechPretrainingDataset(
        manifest_file=str(manifest_path),
        audio_root=str(client_data_path),  # Use client-specific directory
        split="train",
        max_length=int(pre_cfg.get('max_audio_length', 40000)),
        sample_rate=int(pre_cfg.get('sample_rate', 16000)),
        mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
        mask_length=int(pre_cfg.get('mask_length', 10)),
        vocab_size=504,
        kmeans_targets_path=kmeans_targets_str,
    )
    train_dataset_time = time.time() - train_dataset_start
    logger.info(
        f"Client {client_id}: Train dataset created in {train_dataset_time:.2f}s - {len(train_dataset)} samples")

    # Create validation dataset with reduced sequence length
    val_dataset_start = time.time()

    # Create validation dataset with reduced sequence length
    val_dataset = LibriSpeechPretrainingDataset(
        manifest_file=str(manifest_path),
        audio_root=str(client_data_path),  # Use client-specific directory
        split="validation",
        max_length=int(pre_cfg.get('max_audio_length', 40000)),
        sample_rate=int(pre_cfg.get('sample_rate', 16000)),
        mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
        mask_length=int(pre_cfg.get('mask_length', 10)),
        vocab_size=504,
        kmeans_targets_path=kmeans_targets_str,
    )
    val_dataset_time = time.time() - val_dataset_start
    logger.info(
        f"Client {client_id}: Val dataset created in {val_dataset_time:.2f}s - {len(val_dataset)} samples")

    # Initialize model
    model = HubertBase().to(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    return FederatedClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    ).to_client()


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics using weighted average."""
    # Calculate the total number of examples used during training
    num_total_evaluation_examples = sum(
        num_examples for num_examples, _ in metrics)

    # Create a list of weights, that multiplied by the number of examples
    # sum to 1.0
    weights = [num_examples /
               num_total_evaluation_examples for num_examples, _ in metrics]

    # Multiply each client's metrics by its weight
    weighted_metrics = {}
    for weight, client_metrics in zip(weights, metrics):
        for metric_name, metric_value in client_metrics[1].items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += weight * metric_value

    return weighted_metrics


def server_fn(context: Context, config_override: Optional[Dict] = None) -> ServerAppComponents:
    """Server function to initialize the federated learning server."""
    # Use the overridden config if provided, otherwise fall back to file reading
    if config_override is not None:
        config = config_override
    else:
        # Use the global config path
        if CLIENT_CONFIG_PATH is None:
            raise ValueError(
                "CLIENT_CONFIG_PATH not set. Please run the script from the main function.")

        config_path = CLIENT_CONFIG_PATH

        # If config_path is relative, make it absolute
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Create initial parameters from a dummy model
    dummy_model = HubertBase()
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in dummy_model.state_dict().items()])

    # Create strategy
    # Optionally provide server-to-client training config (e.g., lr, local_epochs)
    def fit_config_fn(server_round: int) -> Dict[str, float]:
        # Drive client hyperparams from 'pretraining' section for consistency
        pre_cfg = config.get('pretraining', {})
        lr = float(pre_cfg.get('learning_rate', 5e-4))
        local_epochs = int(pre_cfg.get(
            'local_epochs', pre_cfg.get('epochs', 1)))
        return {"lr": lr, "local_epochs": local_epochs}

    class SavingFedAdam(FedAdam):
        """FedAdam strategy with checkpointing capabilities."""

        def __init__(self, save_dir, state_keys, checkpoint_config, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_dir = Path(save_dir)
            self.state_keys = state_keys
            self.checkpoint_config = checkpoint_config
            self.best_loss = float('inf')
            self.best_round = 0

            # Create save directory if it doesn't exist
            self.save_dir.mkdir(parents=True, exist_ok=True)

            # Checkpointing settings
            self.save_latest = checkpoint_config.get('save_latest', True)
            self.save_best = checkpoint_config.get('save_best', True)
            self.save_best_round = checkpoint_config.get(
                'save_best_round', True)
            self.cleanup_old = checkpoint_config.get('cleanup_old', True)
            self.max_checkpoints = checkpoint_config.get('max_checkpoints', 3)

        def aggregate_fit(self, server_round, results, failures):
            """Aggregate fit results and save checkpoints."""
            # Add progress indicator for simulation mode
            if CLIENT_CONFIG_PATH:  # Simulation mode
                print(f"\n🎯 ROUND {server_round} COMPLETED!")
                print(f"📈 Aggregating results from {len(results)} clients...")

            # Call parent aggregation
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                # Save latest model
                if self.save_latest:
                    latest_path = self.save_dir / "latest_state.pt"
                    self._save_checkpoint(
                        aggregated_parameters, latest_path, server_round)
                    if CLIENT_CONFIG_PATH:  # Simulation mode
                        print(f"💾 Latest model saved: {latest_path}")

                # Save round-specific checkpoint
                if self.save_best_round:
                    round_path = self.save_dir / \
                        f"round_{server_round:03d}_state.pt"
                    self._save_checkpoint(
                        aggregated_parameters, round_path, server_round)

                # Check if this is the best model so far
                if aggregated_metrics and 'eval_pretrain_loss' in aggregated_metrics:
                    current_loss = aggregated_metrics['eval_pretrain_loss']
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.best_round = server_round

                        # Save best model
                        if self.save_best:
                            best_path = self.save_dir / "best_state.pt"
                            self._save_checkpoint(
                                aggregated_parameters, best_path, server_round)
                            if CLIENT_CONFIG_PATH:  # Simulation mode
                                print(
                                    f"🏆 New best model saved: {best_path} (Loss: {current_loss:.4f})")

                # Cleanup old checkpoints
                if self.cleanup_old:
                    self._cleanup_old_checkpoints(server_round)

                if CLIENT_CONFIG_PATH:  # Simulation mode
                    print(f"✅ Round {server_round} aggregation successful!")
            else:
                # No aggregated parameters - log appropriate message
                if CLIENT_CONFIG_PATH:  # Simulation mode
                    print(
                        f"⚠️  Round {server_round} aggregation failed - no parameters returned")
                else:
                    logger.warning(
                        f"Round {server_round} aggregation failed - no parameters returned")

            return aggregated_parameters, aggregated_metrics

        def _save_checkpoint(self, parameters, path, server_round):
            """Save model checkpoint."""
            try:
                # Convert parameters to state dict format
                state_dict = OrderedDict()
                for i, key in enumerate(self.state_keys):
                    if i < len(parameters):
                        state_dict[key] = torch.tensor(parameters[i])

                # Save checkpoint
                torch.save({
                    'round': server_round,
                    'state_dict': state_dict,
                    'best_loss': self.best_loss,
                    'best_round': self.best_round
                }, path)

            except Exception as e:
                if CLIENT_CONFIG_PATH:  # Simulation mode
                    print(
                        f"⚠️  Warning: Could not save checkpoint to {path}: {e}")
                else:
                    logger.warning(f"Could not save checkpoint to {path}: {e}")

        def _cleanup_old_checkpoints(self, current_round):
            """Remove old round-specific checkpoints, keep only latest and best."""
            try:
                for checkpoint_file in self.save_dir.glob("round_*_state.pt"):
                    # Keep only the current round and best round checkpoints
                    if checkpoint_file.name != f"round_{current_round:03d}_state.pt" and checkpoint_file.name != f"round_{self.best_round:03d}_state.pt":
                        checkpoint_file.unlink()
                        if CLIENT_CONFIG_PATH:  # Simulation mode
                            print(
                                f"🗑️  Cleaned up old checkpoint: {checkpoint_file.name}")
            except Exception as e:
                if CLIENT_CONFIG_PATH:  # Simulation mode
                    print(
                        f"⚠️  Warning: Could not cleanup old checkpoints: {e}")
                else:
                    logger.warning(f"Could not cleanup old checkpoints: {e}")

    # Determine save directory from config if available
    save_dir = None
    if 'checkpointing' in config and isinstance(config['checkpointing'], dict) and 'save_dir' in config['checkpointing']:
        save_dir = Path(config['checkpointing']['save_dir'])
    elif 'output' in config and isinstance(config['output'], dict) and 'save_dir' in config['output']:
        save_dir = Path(config['output']['save_dir'])
    else:
        save_dir = Path(
            "/home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining")

    # Keys from dummy model define state_dict ordering
    state_keys = list(dummy_model.state_dict().keys())

    # Get strategy parameters from config with fallbacks
    strategy_config = config.get('strategy', {})
    pretraining_config = config.get('pretraining', {})

    # Log strategy configuration
    logger.info(f"Strategy configuration - fraction_fit: {pretraining_config.get('fraction_fit', 1.0)}, "
                f"min_fit_clients: {pretraining_config.get('min_fit_clients', 2)}, "
                f"num_rounds: {pretraining_config.get('num_rounds', 1)}")

    # Create the strategy with checkpointing
    strategy = SavingFedAdam(
        save_dir=save_dir,
        state_keys=state_keys,
        checkpoint_config=config.get('checkpointing', {}),
        fraction_fit=strategy_config.get(
            'fraction_fit', pretraining_config.get('fraction_fit', 1.0)),
        fraction_evaluate=strategy_config.get(
            'fraction_evaluate', pretraining_config.get('fraction_evaluate', 1.0)),
        min_fit_clients=strategy_config.get(
            'min_fit_clients', pretraining_config.get('min_fit_clients', 2)),
        min_evaluate_clients=strategy_config.get(
            'min_evaluate_clients', pretraining_config.get('min_evaluate_clients', 2)),
        min_available_clients=strategy_config.get(
            'min_available_clients', pretraining_config.get('min_fit_clients', 2)),
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
    )

    # Server config - use default or get from config
    num_rounds = config.get('pretraining', {}).get('num_rounds', 2)
    server_config = ServerConfig(num_rounds=num_rounds)

    # Debug: Print strategy configuration
    if CLIENT_CONFIG_PATH:  # Simulation mode
        print(f"🔧 Strategy configured with num_rounds: {num_rounds}")
        print(
            f"🔧 Config num_rounds: {pretraining_config.get('num_rounds', 1)}")

    # Create server app components
    server_app_components = ServerAppComponents(
        config=server_config,
        strategy=strategy,
    )

    return server_app_components


def main():
    """Main function to run federated HuBERT pretraining."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set environment variables to reduce thread usage
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Set torch to use single thread
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(
        description="Federated HuBERT Pretraining")
    parser.add_argument(
        "--config", type=str, default="/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml", help="Path to config file")
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode")
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of clients (supernodes)")
    parser.add_argument("--num-rounds", type=int,
                        default=2, help="Number of rounds")

    args = parser.parse_args()

    if args.simulation:
        print("🚀 Starting Federated HuBERT Pretraining")
        print(
            f"📊 Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
        print("=" * 50)
        print("Progress will be shown below (reduced logging for clarity):")
        print()

        # Set the global config path
        global CLIENT_CONFIG_PATH
        CLIENT_CONFIG_PATH = args.config

        # Create necessary directories
        os.makedirs('logs/pretraining', exist_ok=True)
        os.makedirs('checkpoints/pretraining', exist_ok=True)

        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Override num_rounds if specified
        if args.num_rounds:
            config['pretraining']['num_rounds'] = args.num_rounds

        # Use minimal backend config for simulation
        backend_config = {
            "client_resources": {
                "num_cpus": 0.25,
                "num_gpus": 0.05,
                "memory": 1000000000
            }
        }

        try:
            # Create a custom server function that uses the overridden config
            def server_fn_with_config(context: Context) -> ServerAppComponents:
                return server_fn(context, config_override=config)

            run_simulation(
                client_app=ClientApp(client_fn=client_fn),
                server_app=ServerApp(server_fn=server_fn_with_config),
                num_supernodes=args.num_clients,
                backend_config=backend_config
            )
            print(f"\n✅ Simulation completed successfully!")
            print(
                f"📊 Trained {args.num_clients} clients for {args.num_rounds} rounds")
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation failed: {e}")
            raise
    else:
        # Legacy mode - load configuration and run
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("Starting federated HuBERT pretraining...")

        # Get number of rounds from config
        num_rounds = config.get('pretraining', {}).get('num_rounds', 1)
        logger.info(f"Configuring federated learning for {num_rounds} rounds")

        # Run simulation
        run_simulation(
            client_app=ClientApp(client_fn=client_fn),
            server_app=ServerApp(server_fn=server_fn),
            num_supernodes=args.num_clients,
            backend_config=config
        )

        logger.info("Simulation completed!")
        logger.info("Federated HuBERT pretraining completed!")

    return None


if __name__ == "__main__":
    main()
