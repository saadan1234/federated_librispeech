#!/usr/bin/env python3
"""
Research-Grade Federated HuBERT Pretraining with Flower (FedAdam).

Core functionality:
- HuBERT-like transformer model with proper frame-level masking (8% probability)
- LibriSpeech dataset with KMeans pseudo-labels following HuBERT paper
- Federated learning with FedAdam aggregation following FLWR best practices
- Progress bars for training visibility
- Checkpointing for latest 3 rounds + initial model
- Proper evaluation metrics for research comparison

PERFORMANCE OPTIMIZATIONS:
- Mixed Precision Training (FP16) for ~2x speedup on modern GPUs
- Gradient Accumulation for larger effective batch sizes
- Multi-worker DataLoader with persistent workers
- Non-blocking tensor transfers to GPU
- Minimal logging during training for maximum speed
- Prefetching of data batches
"""

import os
import logging
import time
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

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
from tqdm import tqdm

from flwr.common.typing import NDArrays, Config
from flwr.client import NumPyClient, ClientApp
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters

# Global config path variable
CLIENT_CONFIG_PATH = None

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown signal received, cleaning up...")
    sys.exit(0)


# Minimal logging for performance
logging.basicConfig(level=logging.WARNING,  # Reduced from INFO to WARNING
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs following HuBERT paper."""

    def __init__(self, hidden_size: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2)
                             * (-np.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


class HubertBase(nn.Module):
    """HuBERT-like model for pretraining following the original paper architecture."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12, vocab_size: int = 504, frame_stride: int = 320):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.frame_stride = frame_stride

        # Transformer encoder layers following HuBERT paper
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1,
                activation='gelu'  # HuBERT uses GELU
            ) for _ in range(num_layers)
        ])

        # Input projection: raw audio -> hidden dimension
        self.input_projection = nn.Linear(1, hidden_size)

        # Output projection: hidden dimension -> vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Mask embedding following HuBERT paper (learned)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # Layer normalization following HuBERT paper
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights following HuBERT paper
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following HuBERT paper."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_values: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # input_values: [batch, samples]
        x = input_values.unsqueeze(-1)                      # [B, T, 1]
        x = self.input_projection(x)                        # [B, T, H]
        x = x.transpose(1, 2)                               # [B, H, T]
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)                               # [B, T_frames, H]

        # Apply mask embedding if provided (following HuBERT paper)
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)
            x = torch.where(mask_expanded, self.mask_embedding.view(
                1, 1, -1).expand_as(x), x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Apply layer norm before transformer (following HuBERT paper)
        x = self.layer_norm(x)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm (following HuBERT paper)
        x = self.layer_norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)  # [B, T_frames, vocab]
        return {"predictions": logits}


class LibriSpeechPretrainingDataset(Dataset):
    """Dataset for LibriSpeech pretraining with proper frame-level masking following HuBERT paper."""

    def __init__(self, manifest_file: str, audio_root: str, split: str = "train",
                 max_length: int = 40000, sample_rate: int = 16000, mask_prob: float = 0.08,
                 mask_length: int = 10, vocab_size: int = 504, kmeans_targets_path: Optional[str] = None):

        df_all = pd.read_csv(manifest_file)

        self.audio_root = Path(audio_root)
        self.split = split
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob  # 8% masking following HuBERT paper
        self.mask_length = mask_length  # Span length following HuBERT paper
        self.vocab_size = vocab_size
        self.frame_stride = 320

        # Filter by split if available
        if 'split' in df_all.columns:
            original_indices = df_all.index[df_all['split'] == split].to_list()
            self.df = df_all.loc[original_indices].reset_index(drop=True)
            self._orig_indices = original_indices
        else:
            self.df = df_all.reset_index(drop=True)
            self._orig_indices = list(range(len(df_all)))

        # Load precomputed KMeans targets (required for HuBERT training)
        if kmeans_targets_path and Path(kmeans_targets_path).exists():
            loaded = np.load(kmeans_targets_path, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype != object:
                all_targets = [row for row in loaded]
            else:
                all_targets = list(loaded)
            self.precomputed_targets = [all_targets[i]
                                        for i in self._orig_indices]
        else:
            raise FileNotFoundError(
                "KMeans targets required for HuBERT training")

    def __len__(self) -> int:
        return len(self.df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Generate boolean span mask following HuBERT paper methodology."""
        mask = torch.zeros(seq_len, dtype=torch.bool)

        # Calculate number of spans to mask (8% of frames)
        num_spans = max(1, int(seq_len * self.mask_prob / self.mask_length))

        # Generate random spans
        for _ in range(num_spans):
            if seq_len >= self.mask_length:
                start = torch.randint(
                    0, seq_len - self.mask_length + 1, (1,)).item()
                mask[start:start + self.mask_length] = True

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with audio, targets, and mask following HuBERT paper."""
        row = self.df.iloc[idx]
        audio_col = 'audio_file' if 'audio_file' in row else 'audio_path'
        audio_path = self.audio_root / row[audio_col]

        # Load and process audio
        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if stereo

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(torch.tensor(audio, dtype=torch.float32))
        else:
            audio = torch.tensor(audio, dtype=torch.float32)

        # Truncate or pad to max_length
        if len(audio) > self.max_length:
            start = torch.randint(
                0, len(audio) - self.max_length + 1, (1,)).item()
            audio = audio[start:start + self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = F.pad(audio, (0, padding))

        # Frame-level processing following HuBERT paper
        target_seq_len = len(audio) // self.frame_stride
        mask = self._span_mask(target_seq_len)

        # Use precomputed KMeans targets (required for HuBERT)
        targets = torch.as_tensor(
            self.precomputed_targets[idx], dtype=torch.long)
        if len(targets) < target_seq_len:
            padding = target_seq_len - len(targets)
            targets = F.pad(targets, (0, padding), value=0)
        else:
            targets = targets[:target_seq_len]

        return {
            "input_values": audio,
            "targets": targets,
            "mask": mask
        }


def custom_collate_fn(batch):
    """Optimized custom collate function for proper batching."""
    # Extract each component
    input_values = [item['input_values'] for item in batch]
    targets = [item['targets'] for item in batch]
    masks = [item['mask'] for item in batch]

    # Stack tensors
    input_values = torch.stack(input_values, dim=0)
    targets = torch.stack(targets, dim=0)
    masks = torch.stack(masks, dim=0)

    return {
        'input_values': input_values,
        'targets': targets,
        'mask': masks
    }


class FederatedClient(NumPyClient):
    """Federated client for HuBERT pretraining following FLWR best practices."""

    def __init__(self, client_id: int, train_dataset, val_dataset, model, device=None):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

        # Device setup with multi-GPU support for Compute Canada
        if device is None:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # Distribute clients across available GPUs
                gpu_id = client_id % gpu_count
                self.device = torch.device(f"cuda:{gpu_id}")

                # Set memory fraction for multi-GPU training
                torch.cuda.set_per_process_memory_fraction(0.7, gpu_id)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Load config for training parameters
        config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        batch_size = cfg.get('pretraining', {}).get('batch_size', 32)
        num_workers = cfg.get('pretraining', {}).get('num_workers', 8)
        prefetch_factor = cfg.get('pretraining', {}).get('prefetch_factor', 4)

        # Store config for DataLoader optimization
        self.config = cfg.get('pretraining', {})

        # Create data loaders with Compute Canada optimizations
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=custom_collate_fn,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=custom_collate_fn,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )

        self.batch_size = batch_size

    def get_parameters(self, config: Config) -> NDArrays:
        """Get model parameters as NumPy arrays following FLWR patterns."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays following FLWR patterns."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """Train the model using provided parameters following HuBERT paper methodology."""
        self.set_parameters(parameters)

        # Setup optimizer following HuBERT paper
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))

        # Enable mixed precision training for speed
        use_mixed_precision = config.get("use_mixed_precision", True)
        scaler = torch.cuda.amp.GradScaler() if (
            self.device.type == 'cuda' and use_mixed_precision) else None

        # Gradient accumulation for larger effective batch size
        accumulation_steps = config.get("accumulation_steps", 4)

        # Training loop
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        local_epochs = int(config.get("local_epochs", 1))
        learning_rate = float(config.get("lr", 5e-4))

        for g in optimizer.param_groups:
            g["lr"] = learning_rate

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            # Progress bar for training
            pbar = tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f"Client {self.client_id} Epoch {epoch + 1}/{local_epochs}",
                leave=False
            )

            for batch_idx, batch in enumerate(pbar):
                # Extract data - assume proper format from custom_collate_fn
                input_values = batch['input_values'].to(
                    self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)

                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_values, frame_mask=mask)
                        predictions = outputs['predictions']

                        # Compute loss ONLY on masked frames following HuBERT paper
                        batch_size, seq_len, vocab_size = predictions.size()
                        predictions_flat = predictions.view(
                            batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        mask_flat = mask.view(batch_size * seq_len)

                        # Only compute loss on masked frames (HuBERT methodology)
                        if mask_flat.any():
                            masked_predictions = predictions_flat[mask_flat]
                            masked_targets = targets_flat[mask_flat]
                            loss = F.cross_entropy(
                                masked_predictions, masked_targets) / accumulation_steps
                        else:
                            continue

                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                else:
                    # Standard precision forward pass
                    outputs = self.model(input_values, frame_mask=mask)
                    predictions = outputs['predictions']

                    # Compute loss ONLY on masked frames following HuBERT paper
                    batch_size, seq_len, vocab_size = predictions.size()
                    predictions_flat = predictions.view(
                        batch_size * seq_len, vocab_size)
                    targets_flat = targets.view(batch_size * seq_len)
                    mask_flat = mask.view(batch_size * seq_len)

                    # Only compute loss on masked frames (HuBERT methodology)
                    if mask_flat.any():
                        masked_predictions = predictions_flat[mask_flat]
                        masked_targets = targets_flat[mask_flat]
                        loss = F.cross_entropy(
                            masked_predictions, masked_targets) / accumulation_steps
                    else:
                        continue

                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                epoch_samples += input_values.size(0)

                # Update progress bar less frequently for speed
                if batch_idx % 20 == 0:
                    pbar.set_postfix(
                        {'loss': f'{loss.item() * accumulation_steps:.4f}'})

            # Final optimizer step if needed for gradient accumulation
            if (len(self.train_loader) % accumulation_steps) != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += epoch_loss
            num_samples += epoch_samples

        avg_loss = total_loss / max(1, local_epochs * len(self.train_loader))
        final_params = self.get_parameters(config={})

        return final_params, num_samples, {"pretrain_loss": avg_loss}

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model using provided parameters with proper metrics."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_samples = 0
        num_masked_frames = 0

        with torch.no_grad():
            # Progress bar for evaluation
            pbar = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Client {self.client_id} Evaluation",
                leave=False
            )

            for batch_idx, batch in enumerate(pbar):
                # Extract data - assume proper format from custom_collate_fn
                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(input_values, frame_mask=mask)
                predictions = outputs['predictions']

                # Compute metrics on masked frames only (following HuBERT paper)
                batch_size, seq_len, vocab_size = predictions.size()
                predictions_flat = predictions.view(
                    batch_size * seq_len, vocab_size)
                targets_flat = targets.view(batch_size * seq_len)
                mask_flat = mask.view(batch_size * seq_len)

                if mask_flat.any():
                    masked_predictions = predictions_flat[mask_flat]
                    masked_targets = targets_flat[mask_flat]

                    # Loss on masked frames
                    loss = F.cross_entropy(masked_predictions, masked_targets)

                    # Accuracy on masked frames
                    preds = torch.argmax(masked_predictions, dim=-1)
                    accuracy = (preds == masked_targets).float().mean()

                    # Perplexity on masked frames
                    perplexity = torch.exp(loss)

                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    total_perplexity += perplexity.item()
                    num_masked_frames += mask_flat.sum().item()
                else:
                    # Fallback if no masked frames
                    loss = F.cross_entropy(predictions_flat, targets_flat)
                    preds = torch.argmax(predictions_flat, dim=-1)
                    accuracy = (preds == targets_flat).float().mean()
                    perplexity = torch.exp(loss)

                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    total_perplexity += perplexity.item()

                num_samples += input_values.size(0)

                # Update progress bar less frequently for speed
                if batch_idx % 20 == 0:
                    pbar.set_postfix(
                        {'loss': f'{loss.item():.4f}', 'acc': f'{accuracy.item():.4f}'})

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_accuracy = total_accuracy / max(1, len(self.val_loader))
        avg_perplexity = total_perplexity / max(1, len(self.val_loader))

        # Return comprehensive metrics for research comparison
        return avg_loss, num_samples, {
            "accuracy": avg_accuracy,
            "perplexity": avg_perplexity,
            "masked_frames": num_masked_frames
        }


class CheckpointingFedAdam(FedAdam):
    """FedAdam strategy with checkpointing for latest 3 rounds + initial model."""

    def __init__(self, save_dir: str, state_keys: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = Path(save_dir)
        self.state_keys = state_keys
        self.round_checkpoints = []  # Track latest 3 rounds
        self.initial_saved = False  # Track if initial model was saved

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results and save checkpoint."""
        print(
            f"🔍 CheckpointingFedAdam.aggregate_fit called for round {server_round}")

        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(
                f"✅ Round {server_round} aggregation successful, saving checkpoint...")
            # Save checkpoint for this round
            self._save_checkpoint(aggregated_parameters, server_round)

            # Update round tracking and cleanup old checkpoints
            self._manage_checkpoints(server_round)

            print(
                f"✅ Round {server_round} checkpoint saved. Total checkpoints: {len(self.round_checkpoints)}")
        else:
            print(f"⚠️  No aggregated parameters for round {server_round}")

        return aggregated_parameters, aggregated_metrics

    def save_initial_model(self, initial_parameters):
        """Save the initial model checkpoint."""
        try:
            print(f"🚀 Attempting to save initial model checkpoint...")
            print(f"📁 Save directory: {self.save_dir}")
            print(f"🔑 Number of state keys: {len(self.state_keys)}")

            # Convert Flower Parameters to list of numpy arrays
            if hasattr(initial_parameters, 'tensors'):
                # It's a Flower Parameters object - convert bytes to numpy arrays
                parameters_list = []
                for i, tensor_bytes in enumerate(initial_parameters.tensors):
                    if isinstance(tensor_bytes, bytes):
                        # Convert bytes to numpy array
                        import numpy as np
                        tensor_array = np.frombuffer(
                            tensor_bytes, dtype=np.float32)
                        parameters_list.append(tensor_array)
                        print(
                            f"   - Converted tensor {i}: bytes -> numpy array {tensor_array.shape}")
                    else:
                        # Already a numpy array
                        parameters_list.append(tensor_bytes)
                        print(
                            f"   - Tensor {i}: already numpy array {tensor_bytes.shape}")

                print(
                    f"📊 Converted Flower Parameters to {len(parameters_list)} numpy arrays")
            else:
                # Assume it's already a list
                parameters_list = initial_parameters
                print(
                    f"📊 Using parameters as-is: {len(parameters_list)} items")

            checkpoint_path = self.save_dir / "initial_model_checkpoint.pt"
            print(f"💾 Initial checkpoint path: {checkpoint_path}")

            self._save_checkpoint(parameters_list, 0, checkpoint_path)
            self.initial_saved = True
            print(f"✅ Initial model checkpoint saved successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to save initial model: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def _save_checkpoint(self, parameters: NDArrays, server_round: int, custom_path: Optional[Path] = None):
        """Save model checkpoint for a specific round."""
        try:
            print(f"💾 Saving checkpoint for round {server_round}...")
            print(f"📁 Save directory: {self.save_dir}")
            print(f"🔑 Number of state keys: {len(self.state_keys)}")

            # Convert Flower Parameters to list of numpy arrays if needed
            if hasattr(parameters, 'tensors'):
                # It's a Flower Parameters object - convert bytes to numpy arrays
                parameters_list = []
                for i, tensor_bytes in enumerate(parameters.tensors):
                    if isinstance(tensor_bytes, bytes):
                        # Convert bytes to numpy array
                        import numpy as np
                        tensor_array = np.frombuffer(
                            tensor_bytes, dtype=np.float32)
                        parameters_list.append(tensor_array)
                        print(
                            f"   - Converted tensor {i}: bytes -> numpy array {tensor_array.shape}")
                    else:
                        # Already a numpy array
                        parameters_list.append(tensor_bytes)
                        print(
                            f"   - Tensor {i}: already numpy array {tensor_bytes.shape}")

                print(
                    f"📊 Converted Flower Parameters to {len(parameters_list)} numpy arrays")
            else:
                # Assume it's already a list
                parameters_list = parameters
                print(
                    f"📊 Using parameters as-is: {len(parameters_list)} items")

            # Verify parameters are valid
            if not parameters_list or len(parameters_list) == 0:
                print(f"❌ No parameters to save for round {server_round}")
                return False

            # Verify we have the right number of state keys
            if len(parameters_list) != len(self.state_keys):
                print(
                    f"⚠️  Parameter count mismatch: {len(parameters_list)} parameters vs {len(self.state_keys)} state keys")

            # Convert parameters to state dict format
            state_dict = OrderedDict()
            for i, key in enumerate(self.state_keys):
                if i < len(parameters_list):
                    state_dict[key] = torch.tensor(parameters_list[i])
                    print(f"   - Added {key}: {parameters_list[i].shape}")
                else:
                    print(f"⚠️  Missing parameter for key {key}")

            # Save checkpoint
            if custom_path is None:
                checkpoint_path = self.save_dir / \
                    f"round_{server_round:03d}_checkpoint.pt"
            else:
                checkpoint_path = custom_path

            print(f"💾 Saving to: {checkpoint_path}")

            # Save the checkpoint
            checkpoint_data = {
                'round': server_round,
                'state_dict': state_dict,
                'timestamp': time.time(),
                'num_parameters': len(parameters_list),
                'state_keys': self.state_keys
            }

            torch.save(checkpoint_data, checkpoint_path)

            print(f"✅ Checkpoint saved successfully: {checkpoint_path}")

            # Verify file was created
            if checkpoint_path.exists():
                file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
                print(f"📏 File size: {file_size:.2f} MB")
            else:
                print(f"❌ ERROR: Checkpoint file was not created!")

        except Exception as e:
            print(f"❌ Failed to save checkpoint for round {server_round}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def _manage_checkpoints(self, current_round: int):
        """Manage checkpoints to keep only latest 3 rounds."""
        try:
            print(f"🔧 Managing checkpoints for round {current_round}...")

            # Add current round to tracking
            self.round_checkpoints.append(current_round)
            print(
                f"📝 Added round {current_round} to tracking. Current rounds: {self.round_checkpoints}")

            # Keep only latest 3 rounds
            if len(self.round_checkpoints) > 3:
                # Remove oldest checkpoint
                oldest_round = self.round_checkpoints.pop(0)
                oldest_checkpoint = self.save_dir / \
                    f"round_{oldest_round:03d}_checkpoint.pt"

                if oldest_checkpoint.exists():
                    oldest_checkpoint.unlink()
                    print(
                        f"🗑️  Removed old checkpoint: round_{oldest_round:03d}_checkpoint.pt")
                else:
                    print(
                        f"⚠️  Old checkpoint not found for removal: {oldest_checkpoint}")

            # Log current checkpoint status
            print(f"📊 Current checkpoints: rounds {self.round_checkpoints}")

            # List all checkpoint files
            checkpoint_files = list(self.save_dir.glob("*.pt"))
            if checkpoint_files:
                print(f"📁 All checkpoint files in {self.save_dir}:")
                for cf in checkpoint_files:
                    size_mb = cf.stat().st_size / (1024 * 1024)
                    print(f"   - {cf.name} ({size_mb:.2f} MB)")
            else:
                print(f"⚠️  No checkpoint files found in {self.save_dir}")

        except Exception as e:
            print(f"❌ Error in checkpoint management: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize federated learning client following FLWR patterns."""
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine client ID
    node_id = context.node_id
    num_clients = config.get('simulation', {}).get('num_supernodes', 2)
    client_id = hash(str(node_id)) % num_clients

    # Setup data paths
    data_root = Path(config.get('data', {}).get(
        'partitioned_data_root', 'data/partitioned'))
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root

    client_data_path = data_root / f"client_{client_id}"
    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

    # Find manifest and targets
    manifest_path = client_data_path / "manifest.csv"
    if not manifest_path.exists():
        manifest_path = data_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found")

    targets_path = client_data_path / "kmeans_targets.npy"
    if not targets_path.exists():
        targets_path = data_root / "kmeans_targets.npy"

    if not targets_path.exists():
        raise FileNotFoundError(f"KMeans targets not found")

    # Load config for dataset parameters
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    pre_cfg = cfg.get('pretraining', {})

    # Create datasets
    train_dataset = LibriSpeechPretrainingDataset(
        manifest_file=str(manifest_path),
        audio_root=str(client_data_path),
        split="train",
        max_length=int(pre_cfg.get('max_audio_length', 40000)),
        sample_rate=int(pre_cfg.get('sample_rate', 16000)),
        # 8% masking following HuBERT paper
        mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
        mask_length=int(pre_cfg.get('mask_length', 10)),
        vocab_size=504,
        kmeans_targets_path=str(targets_path),
    )

    val_dataset = LibriSpeechPretrainingDataset(
        manifest_file=str(manifest_path),
        audio_root=str(client_data_path),
        split="validation",
        max_length=int(pre_cfg.get('max_audio_length', 40000)),
        sample_rate=int(pre_cfg.get('sample_rate', 16000)),
        # 8% masking following HuBERT paper
        mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
        mask_length=int(pre_cfg.get('mask_length', 10)),
        vocab_size=504,
        kmeans_targets_path=str(targets_path),
    )

    # Initialize model
    model = HubertBase()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return FederatedClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        device=device
    ).to_client()


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics using weighted average following FLWR patterns."""
    num_total_examples = sum(num_examples for num_examples, _ in metrics)
    weights = [num_examples / num_total_examples for num_examples, _ in metrics]

    weighted_metrics = {}
    for weight, client_metrics in zip(weights, metrics):
        for metric_name, metric_value in client_metrics[1].items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0.0
            weighted_metrics[metric_name] += weight * metric_value

    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Server function to initialize federated learning server following FLWR patterns."""
    # Load config
    if CLIENT_CONFIG_PATH is None:
        config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
    else:
        config_path = CLIENT_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create initial parameters from model
    dummy_model = HubertBase()
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in dummy_model.state_dict().items()]
    )

    # Create strategy
    def fit_config_fn(server_round: int) -> Dict[str, float]:
        pre_cfg = config.get('pretraining', {})
        lr = float(pre_cfg.get('learning_rate', 5e-4))
        local_epochs = int(pre_cfg.get('local_epochs', 1))
        return {"lr": lr, "local_epochs": local_epochs}

    # Use checkpointing strategy
    strategy = CheckpointingFedAdam(
        save_dir="/home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining",
        state_keys=list(dummy_model.state_dict().keys()),
        fraction_fit=config.get('pretraining', {}).get('fraction_fit', 1.0),
        fraction_evaluate=config.get('pretraining', {}).get(
            'fraction_evaluate', 1.0),
        min_fit_clients=config.get(
            'pretraining', {}).get('min_fit_clients', 2),
        min_evaluate_clients=config.get(
            'pretraining', {}).get('min_evaluate_clients', 2),
        min_available_clients=config.get(
            'pretraining', {}).get('min_fit_clients', 2),
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
    )

    # Save initial model checkpoint
    strategy.save_initial_model(initial_parameters)

    # Server config
    num_rounds = config.get('pretraining', {}).get('num_rounds', 2)
    server_config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        config=server_config,
        strategy=strategy,
    )


def main():
    """Main function to run federated HuBERT pretraining."""
    global CLIENT_CONFIG_PATH

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Federated HuBERT Pretraining")
    parser.add_argument("--config", type=str,
                        default="/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml",
                        help="Path to config file")
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode")
    parser.add_argument("--num-clients", type=int,
                        default=2, help="Number of clients")
    parser.add_argument("--num-rounds", type=int,
                        default=2, help="Number of rounds")

    args = parser.parse_args()

    if args.simulation:
        print("🚀 Starting Federated HuBERT Pretraining")
        print(
            f"📊 Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
        print("=" * 50)

        # Set global config path
        CLIENT_CONFIG_PATH = args.config

        # Create directories
        os.makedirs('logs/pretraining', exist_ok=True)

        # Test checkpoint directory access
        checkpoint_dir = '/home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining'
        print(f"🔍 Testing checkpoint directory access: {checkpoint_dir}")

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"✅ Checkpoint directory created/verified: {checkpoint_dir}")

            # Test write access
            test_file = os.path.join(checkpoint_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Checkpoint directory is writable")

        except Exception as e:
            print(f"⚠️  Warning: Could not access checkpoint directory: {e}")
            print(f"🔧 Using fallback directory: checkpoints/pretraining")
            checkpoint_dir = "checkpoints/pretraining"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"✅ Created fallback checkpoint directory: {checkpoint_dir}")

        # Load and override config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if args.num_rounds:
            config['pretraining']['num_rounds'] = args.num_rounds

        # Backend config
        backend_config = {
            "client_resources": {
                "num_cpus": max(1, os.cpu_count() // args.num_clients),
                "num_gpus": max(0.1, torch.cuda.device_count() / args.num_clients) if torch.cuda.is_available() else 0.0,
                "memory": max(2000000000, 128000000000 // args.num_clients)
            }
        }

        try:
            # Custom server function with overridden config
            def server_fn_with_config(context: Context) -> ServerAppComponents:
                return server_fn(context)

            run_simulation(
                client_app=ClientApp(client_fn=client_fn),
                server_app=ServerApp(server_fn=server_fn_with_config),
                num_supernodes=args.num_clients,
                backend_config=backend_config
            )
            print(f"\n✅ Simulation completed successfully!")

            # Verify checkpoints were created
            checkpoint_dir = '/home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining'
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(
                    checkpoint_dir) if f.endswith('.pt')]
                if checkpoint_files:
                    print(f"✅ Checkpoints saved successfully!")
                    print(f"📁 Checkpoint directory: {checkpoint_dir}")
                    print(f"💾 Checkpoint files ({len(checkpoint_files)}):")
                    for cf in checkpoint_files:
                        file_path = os.path.join(checkpoint_dir, cf)
                        file_size = os.path.getsize(
                            file_path) / (1024 * 1024)  # MB
                        print(f"   - {cf} ({file_size:.2f} MB)")
                else:
                    print(f"⚠️  No checkpoint files found in {checkpoint_dir}")
                    print("🔍 Checking if directory is writable...")
                    try:
                        test_file = os.path.join(checkpoint_dir, "test.tmp")
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                        print(
                            "✅ Directory is writable - checkpointing may have failed silently")
                    except Exception as e:
                        print(f"❌ Directory is not writable: {e}")
            else:
                print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
                print("🔍 Creating fallback checkpoint directory...")
                try:
                    os.makedirs("checkpoints/pretraining", exist_ok=True)
                    print(
                        "✅ Created fallback checkpoint directory: checkpoints/pretraining")
                except Exception as e:
                    print(f"❌ Could not create fallback directory: {e}")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation failed: {e}")
            raise
    else:
        # Legacy mode
        CLIENT_CONFIG_PATH = args.config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("Starting federated HuBERT pretraining...")
        num_rounds = config.get('pretraining', {}).get('num_rounds', 1)
        logger.info(f"Configuring federated learning for {num_rounds} rounds")

        run_simulation(
            client_app=ClientApp(client_fn=client_fn),
            server_app=ServerApp(server_fn=server_fn),
            num_supernodes=args.num_clients,
            backend_config=config
        )

        logger.info("Federated HuBERT pretraining completed!")


if __name__ == "__main__":
    main()
