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
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

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

                return {
                    "input_values": audio,
                    "targets": targets,
                    "mask": mask
                }

            except Exception as e:
                logger.warning(
                    f"Failed to load sample {idx} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Return fallback data
                    fallback_audio = torch.zeros(
                        self.max_length, dtype=torch.float32)
                    fallback_targets = torch.zeros(
                        self.max_length // self.frame_stride, dtype=torch.long)
                    fallback_mask = torch.zeros(
                        self.max_length // self.frame_stride, dtype=torch.bool)

                    return {
                        "input_values": fallback_audio,
                        "targets": fallback_targets,
                        "mask": fallback_mask
                    }

        raise RuntimeError(
            f"Failed to load sample after {max_retries} attempts")


# Checkpointing removed for simplicity


class FederatedClient(NumPyClient):
    """Federated client with HubertBase model using real data"""

    def __init__(self, client_id: int, data_path: str):
        logger.info(f"Initializing client {client_id}")

        self.client_id = client_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Client {client_id} using device: {self.device}")

        # Initialize model
        self.model = HubertBase().to(self.device)

        # Setup data with real LibriSpeech partitions
        self._setup_data(data_path)

        logger.info(
            f"Client {client_id} initialized with {len(self.train_dataset)} train samples, {len(self.val_dataset)} val samples")

    def _setup_data(self, data_path: str):
        """Setup real LibriSpeech data loading"""
        data_path = Path(data_path)

        # Load config to align dataset/dataloader params
        cfg_path = Path(
            "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        pre_cfg = cfg.get('pretraining', {})
        dl_cfg = cfg.get('data', {}).get('dataloader', {})

        # Load manifest file
        manifest_path = data_path / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}")

        # Optional precomputed per-client targets
        # Optional precomputed per-client targets
        targets_path = data_path / "kmeans_targets.npy"
        kmeans_targets_str = str(
            targets_path) if targets_path.exists() else None

        # Create train dataset with reduced sequence length
        self.train_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="train",
            max_length=int(pre_cfg.get('max_audio_length', 40000)),
            sample_rate=int(pre_cfg.get('sample_rate', 16000)),
            mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
            mask_length=int(pre_cfg.get('mask_length', 10)),
            vocab_size=504,
            kmeans_targets_path=kmeans_targets_str,
        )

        # Create validation dataset with reduced sequence length
        self.val_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="validation",
            max_length=int(pre_cfg.get('max_audio_length', 40000)),
            sample_rate=int(pre_cfg.get('sample_rate', 16000)),
            mask_prob=float(pre_cfg.get('mask_prob', 0.08)),
            mask_length=int(pre_cfg.get('mask_length', 10)),
            vocab_size=504,
            kmeans_targets_path=kmeans_targets_str,
        )

        # Create data loaders with smaller batch size
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(pre_cfg.get('batch_size', 2)),
            shuffle=cfg.get('client', {}).get(
                'local_config', {}).get('shuffle', True),
            num_workers=int(dl_cfg.get('num_workers', 0)),
            pin_memory=bool(dl_cfg.get('pin_memory', True)),
            drop_last=cfg.get('client', {}).get(
                'local_config', {}).get('drop_last', True),
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(pre_cfg.get('batch_size', 2)),
            shuffle=False,
            num_workers=int(dl_cfg.get('num_workers', 0)),
            pin_memory=bool(dl_cfg.get('pin_memory', True)),
            drop_last=cfg.get('client', {}).get(
                'local_config', {}).get('drop_last', True),
        )

        logger.info(f"Data loaders created successfully")

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
        self.set_parameters(parameters)

        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=5e-4, weight_decay=0.01)

        # Training loop
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        total_steps = 0
        start_time = time.time()

        # Use server-provided hyperparameters if available; fallback to config file
        cfg_path = Path(
            "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        pre_cfg = cfg.get('pretraining', {})
        server_cfg = cfg.get('server', {})
        server_timeout = int(server_cfg.get('round_timeout', 1800))
        default_budget = min(1700, int(0.8 * server_timeout))
        max_fit_time_seconds = int(pre_cfg.get('max_fit_time_seconds', default_budget))
        budget_env = os.getenv("FLWR_CLIENT_FIT_BUDGET_SECONDS")
        if budget_env:
            try:
                max_fit_time_seconds = min(max_fit_time_seconds, int(budget_env))
            except ValueError:
                pass
        max_train_steps_per_epoch = int(pre_cfg.get('max_train_steps_per_epoch', 0))

        local_epochs = int(config.get("local_epochs", pre_cfg.get('local_epochs', pre_cfg.get('epochs', 1)))) if isinstance(
            config, dict) else int(pre_cfg.get('local_epochs', pre_cfg.get('epochs', 1)))
        learning_rate = float(config.get("lr", pre_cfg.get('learning_rate', 5e-4))
                              ) if isinstance(config, dict) else float(pre_cfg.get('learning_rate', 5e-4))
        for g in optimizer.param_groups:
            g["lr"] = learning_rate

        time_exceeded = False
        for epoch in range(local_epochs):
            if time_exceeded:
                break
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_steps = 0

            for batch in self.train_loader:
                # Step cap per epoch, if configured
                if max_train_steps_per_epoch and epoch_steps >= max_train_steps_per_epoch:
                    break
                # Time budget cap for the entire fit
                if time.time() - start_time >= max_fit_time_seconds:
                    logger.info(f"Client {self.client_id}: Fit time budget reached, stopping early at epoch {epoch + 1}, step {epoch_steps}")
                    time_exceeded = True
                    break

                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)

                optimizer.zero_grad()

                # Forward with frame mask to enable masked prediction training
                outputs = self.model(input_values, frame_mask=mask)
                predictions = outputs['predictions']

                # Apply frame-level mask and compute loss across batch/time
                batch_size, seq_len, vocab_size = predictions.size()
                predictions_flat = predictions.view(
                    batch_size * seq_len, vocab_size)
                targets_flat = targets.view(batch_size * seq_len)
                mask_flat = mask.view(batch_size * seq_len)

                if mask_flat.any():
                    loss = F.cross_entropy(
                        predictions_flat[mask_flat],
                        targets_flat[mask_flat]
                    )

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_samples += input_values.size(0)
                    epoch_steps += 1
                    total_steps += 1

            if epoch_steps > 0:
                logger.info(
                    f"Client {self.client_id}: Epoch {epoch + 1}, steps={epoch_steps}, loss = {epoch_loss / max(1, epoch_steps):.4f}")

            total_loss += epoch_loss
            num_samples += epoch_samples

        # Average over processed steps
        avg_loss = total_loss / max(1, total_steps)

        logger.info(
            f"Client {self.client_id}: training completed (early_stop={'yes' if time_exceeded else 'no'}), steps={total_steps}, avg_loss={avg_loss:.4f}")

        return self.get_parameters(config={}), num_samples, {"pretrain_loss": avg_loss}

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model using the provided parameters."""
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        eval_steps = 0
        start_time = time.time()

        # Load config for optional eval caps
        cfg_path = Path(
            "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        pre_cfg = cfg.get('pretraining', {})
        server_cfg = cfg.get('server', {})
        server_timeout = int(server_cfg.get('round_timeout', 1800))
        default_eval_budget = min(600, int(0.2 * server_timeout))
        max_eval_time_seconds = int(pre_cfg.get('max_eval_time_seconds', default_eval_budget))
        budget_env = os.getenv("FLWR_CLIENT_EVAL_BUDGET_SECONDS")
        if budget_env:
            try:
                max_eval_time_seconds = min(max_eval_time_seconds, int(budget_env))
            except ValueError:
                pass
        max_eval_batches = int(pre_cfg.get('max_eval_batches', 0))

        with torch.no_grad():
            for batch in self.val_loader:
                # Optional batch/time caps
                if max_eval_batches and eval_steps >= max_eval_batches:
                    break
                if time.time() - start_time >= max_eval_time_seconds:
                    logger.info(f"Client {self.client_id}: Eval time budget reached after {eval_steps} batches, stopping early")
                    break

                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)

                outputs = self.model(input_values, frame_mask=mask)

                # Compute masked loss and accuracy
                predictions = outputs['predictions']  # [B, T, V]
                bsz, tlen, vsize = predictions.size()
                predictions_flat = predictions.view(bsz * tlen, vsize)
                targets_flat = targets.view(bsz * tlen)
                mask_flat = mask.view(bsz * tlen)

                if mask_flat.any():
                    loss = F.cross_entropy(
                        predictions_flat[mask_flat], targets_flat[mask_flat])
                    preds = torch.argmax(predictions_flat[mask_flat], dim=-1)
                    accuracy = (
                        preds == targets_flat[mask_flat]).float().mean()
                else:
                    # Fallback if no masked frames present
                    loss = F.cross_entropy(predictions_flat, targets_flat)
                    preds = torch.argmax(predictions_flat, dim=-1)
                    accuracy = (preds == targets_flat).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_samples += input_values.size(0)
                eval_steps += 1

        avg_loss = total_loss / max(1, eval_steps)
        avg_accuracy = total_accuracy / max(1, eval_steps)

        logger.info(
            f"Client {self.client_id}: evaluation completed, batches={eval_steps}, loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

        return avg_loss, num_samples, {"accuracy": avg_accuracy}


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine the client ID based on the node ID
    node_id = context.node_id
    num_clients = int(config['simulation']['num_supernodes'])
    client_id = hash(str(node_id)) % num_clients

    # Setup data path
    data_root_config = config['data']['partitioned_data_root']
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)

    client_data_path = data_root / f"client_{client_id}"

    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

    manifest_path = client_data_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    logger.info(
        f"Initializing client {client_id} with data from {client_data_path}")

    return FederatedClient(
        client_id=client_id,
        data_path=str(client_data_path)
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


def server_fn(context: Context) -> ServerAppComponents:
    """Server function to initialize the federated learning server."""
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"

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

    # Minimal saving wrapper to persist latest global model state_dict
    class SavingFedAdam(FedAdam):
        def __init__(self, save_dir: Path, state_keys: List[str], **kwargs):
            super().__init__(**kwargs)
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.state_keys = state_keys

        def aggregate_fit(self, server_round, results, failures):
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                nds = parameters_to_ndarrays(aggregated_parameters)
                state_dict = OrderedDict(
                    {k: torch.tensor(v) for k, v in zip(self.state_keys, nds)})
                latest_path = self.save_dir / "latest_state.pt"
                round_path = self.save_dir / \
                    f"round_{server_round:03d}_state.pt"
                torch.save(state_dict, latest_path)
                torch.save(state_dict, round_path)
                logger.info(
                    f"Saved global model state_dict to {latest_path} and {round_path}")
            return aggregated_parameters, aggregated_metrics

    # Determine save directory from config if available
    save_dir = None
    if 'checkpointing' in config and isinstance(config['checkpointing'], dict) and 'save_dir' in config['checkpointing']:
        save_dir = Path(config['checkpointing']['save_dir'])
    elif 'output' in config and isinstance(config['output'], dict) and 'save_dir' in config['output']:
        save_dir = Path(config['output']['save_dir'])
    else:
        save_dir = Path("checkpoints/pretraining")

    # Keys from dummy model define state_dict ordering
    state_keys = list(dummy_model.state_dict().keys())

    strategy = SavingFedAdam(
        save_dir=save_dir,
        state_keys=state_keys,
        fraction_fit=config['strategy']['fraction_fit'],
        fraction_evaluate=config['strategy']['fraction_evaluate'],
        min_fit_clients=config['strategy']['min_fit_clients'],
        min_evaluate_clients=config['strategy']['min_evaluate_clients'],
        min_available_clients=config['strategy']['min_available_clients'],
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
    )

    # Create server app components
    server_app_components = ServerAppComponents(
        strategy=strategy,
    )

    return server_app_components


def main():
    """Main function to run federated HuBERT pretraining."""
    parser = argparse.ArgumentParser(
        description="Federated HuBERT Pretraining")
    parser.add_argument(
        "--config", type=str, default="/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml", help="Path to config file")
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients (supernodes)")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Starting federated HuBERT pretraining...")

    # Run simulation
    run_simulation(
        client_app=ClientApp(client_fn=client_fn),
        server_app=ServerApp(server_fn=server_fn),
        num_supernodes=args.num_clients,
        backend_config=config
    )

    logger.info("Federated HuBERT pretraining completed!")


if __name__ == "__main__":
    main()
