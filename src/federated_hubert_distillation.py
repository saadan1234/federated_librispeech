#!/usr/bin/env python3
"""
Federated HuBERT Knowledge Distillation with Flower (FedAdam).
Teacher model provides knowledge distillation targets, student learns from both teacher predictions and KMeans pseudo-labels.
"""

import os
import logging
import math
import time
import signal
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from flwr.common.typing import NDArrays
from flwr.client import Client
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, Status, Code
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from tqdm import tqdm
from collections import OrderedDict
import yaml
import argparse
import sys

# Global config path variable
CLIENT_CONFIG_PATH = None

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    shutdown_requested = True
    sys.exit(0)


# Performance optimizations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:128")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Setup logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class HubertTeacher(nn.Module):
    """Teacher model with frame-level processing."""

    def __init__(self, hidden_size=768, num_layers=12, vocab_size=504, frame_stride=320):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=12, dim_feedforward=3072,
                batch_first=True, dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.frame_stride = frame_stride

    def forward(self, input_values, frame_mask=None):
        x = input_values.unsqueeze(-1)
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)

        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)
            x = torch.where(mask_expanded, self.mask_embedding.view(
                1, 1, -1).expand_as(x), x)

        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)

        logits = self.output_projection(x)
        return {"predictions": logits}


class HubertStudent(nn.Module):
    """Student model with frame-level processing."""

    def __init__(self, hidden_size=384, num_layers=6, vocab_size=504, frame_stride=320, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=6, dim_feedforward=1536,
                batch_first=True, dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.frame_stride = frame_stride
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, input_values, frame_mask=None):
        x = input_values.unsqueeze(-1)
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)

        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)
            x = torch.where(mask_expanded, self.mask_embedding.view(
                1, 1, -1).expand_as(x), x)

        x = self.positional_encoding(x)
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        logits = self.output_projection(x)
        return {"predictions": logits}


class LibriSpeechDistillationDataset(Dataset):
    """LibriSpeech dataset for distillation training."""

    def __init__(self, manifest_file: str, audio_root: str, split: str = "train",
                 max_length: int = 40000, sample_rate: int = 16000, mask_prob: float = 0.08,
                 mask_length: int = 10, vocab_size: int = 504, kmeans_targets_path: Optional[str] = None):
        df_all = pd.read_csv(manifest_file)

        if 'audio_file' not in df_all.columns and 'audio_path' not in df_all.columns:
            raise ValueError(
                f"Manifest missing required columns. Available: {list(df_all.columns)}")

        self.audio_root = Path(audio_root)
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.vocab_size = vocab_size
        self.split = split
        self.frame_stride = 320

        # Filter by split
        if 'split' in df_all.columns:
            original_indices = df_all.index[df_all['split'] == split].to_list()
            self.manifest_df = df_all.loc[original_indices].reset_index(
                drop=True)
            self._orig_indices = original_indices
        else:
            self.manifest_df = df_all.reset_index(drop=True)
            self._orig_indices = list(range(len(self.manifest_df)))

        # Load k-means targets
        if kmeans_targets_path and os.path.exists(kmeans_targets_path):
            loaded = np.load(kmeans_targets_path, allow_pickle=True)
            all_targets = [row for row in loaded] if isinstance(
                loaded, np.ndarray) and loaded.dtype != object else list(loaded)
            self.kmeans_targets = [all_targets[i] for i in self._orig_indices]
        else:
            raise FileNotFoundError(
                f"K-means targets not found at {kmeans_targets_path}")

    def __len__(self) -> int:
        return len(self.manifest_df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Generate boolean span mask."""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        num_spans = int(seq_len * self.mask_prob / self.mask_length)
        for _ in range(num_spans):
            start = torch.randint(
                0, seq_len - self.mask_length + 1, (1,)).item()
            mask[start:start + self.mask_length] = True
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.manifest_df)
                row = self.manifest_df.iloc[current_idx]

                audio_col = 'audio_file' if 'audio_file' in row else 'audio_path'
                audio_rel = row[audio_col]
                audio_path = self.audio_root / audio_rel

                if not audio_path.exists() or not audio_path.is_file() or audio_path.stat().st_size == 0:
                    continue

                # Load and process audio
                audio, sr = sf.read(str(audio_path))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]

                if sr != self.sample_rate:
                    resampler = T.Resample(
                        orig_freq=sr, new_freq=self.sample_rate)
                    audio = resampler(torch.tensor(audio, dtype=torch.float32))
                else:
                    audio = torch.tensor(audio, dtype=torch.float32)

                # Truncate or pad
                if len(audio) > self.max_length:
                    start = torch.randint(
                        0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    padding = self.max_length - len(audio)
                    audio = F.pad(audio, (0, padding))

                # Generate mask and targets
                target_seq_len = len(audio) // self.frame_stride
                mask = self._span_mask(target_seq_len)

                kmeans_target = self.kmeans_targets[idx]
                if len(kmeans_target) >= target_seq_len:
                    targets = torch.tensor(
                        kmeans_target[:target_seq_len], dtype=torch.long)
                else:
                    padding = target_seq_len - len(kmeans_target)
                    targets = torch.cat([
                        torch.tensor(kmeans_target, dtype=torch.long),
                        torch.full(
                            (padding,), kmeans_target[-1], dtype=torch.long)
                    ])

                if len(targets) < target_seq_len:
                    padding = target_seq_len - len(targets)
                    targets = F.pad(targets, (0, padding), value=0)

                return {
                    "input_values": audio,
                    "targets": targets,
                    "mask": mask
                }

            except Exception as e:
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


class SavingFedAdam(FedAdam):
    """FedAdam strategy with checkpointing capabilities."""

    def __init__(self, save_dir, state_keys, checkpoint_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = Path(save_dir)
        self.state_keys = state_keys
        self.checkpoint_config = checkpoint_config
        self.best_loss = float('inf')
        self.best_round = 0
        self.previous_round = 0
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_latest = checkpoint_config.get('save_latest', True)
        self.save_best = checkpoint_config.get('save_best', True)
        self.save_best_round = checkpoint_config.get('save_best_round', True)
        self.cleanup_old = checkpoint_config.get('cleanup_old', True)
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 3)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and save checkpoints."""
        if CLIENT_CONFIG_PATH:
            print(f"\n🎯 ROUND {server_round} COMPLETED!")

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Save latest model
            if self.save_latest:
                latest_path = self.save_dir / "latest_state.pt"
                self._save_checkpoint(
                    aggregated_parameters, latest_path, server_round)

            # Save round-specific checkpoint
            if self.save_best_round:
                round_path = self.save_dir / \
                    f"round_{server_round:03d}_state.pt"
                self._save_checkpoint(
                    aggregated_parameters, round_path, server_round)

            self.previous_round = server_round

            # Check if this is the best model
            if aggregated_metrics and 'eval_distillation_loss' in aggregated_metrics:
                current_loss = aggregated_metrics['eval_distillation_loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_round = server_round
                    if self.save_best:
                        best_path = self.save_dir / "best_state.pt"
                        self._save_checkpoint(
                            aggregated_parameters, best_path, server_round)

            # Cleanup old checkpoints
            if self.cleanup_old:
                self._cleanup_old_checkpoints(server_round)

        return aggregated_parameters, aggregated_metrics

    def _save_checkpoint(self, parameters, path, server_round):
        """Save model checkpoint."""
        try:
            state_dict = OrderedDict()
            param_list = list(parameters) if hasattr(
                parameters, '__len__') else [p for p in parameters]

            for i, key in enumerate(self.state_keys):
                if i < len(param_list):
                    state_dict[key] = torch.tensor(param_list[i])

            torch.save({
                'round': server_round,
                'state_dict': state_dict,
                'best_loss': self.best_loss,
                'best_round': self.best_round,
                'timestamp': time.time()
            }, path)

        except Exception as e:
            logger.warning(f"Could not save checkpoint to {path}: {e}")

    def _cleanup_old_checkpoints(self, current_round):
        """Remove old round-specific checkpoints."""
        try:
            round_checkpoints = list(self.save_dir.glob("round_*_state.pt"))
            essential_rounds = {current_round,
                                self.best_round, self.previous_round}

            for checkpoint_file in round_checkpoints:
                try:
                    round_num = int(checkpoint_file.stem.split('_')[1])
                    if round_num not in essential_rounds:
                        checkpoint_file.unlink()
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            logger.warning(f"Could not cleanup old checkpoints: {e}")

    def save_initial_checkpoint(self, initial_parameters):
        """Save initial model checkpoint."""
        try:
            if initial_parameters is not None:
                initial_path = self.save_dir / "initial_state.pt"
                self._save_checkpoint(initial_parameters, initial_path, 0)
                return True
        except Exception as e:
            logger.warning(f"Could not save initial checkpoint: {e}")
            return False


class FederatedClient(Client):
    """Federated client with teacher/student models."""

    def __init__(self, client_id: int, data_path: str):
        self.client_id = client_id

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(0.05)
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")

        # Load config
        if CLIENT_CONFIG_PATH is None:
            raise ValueError("CLIENT_CONFIG_PATH is not set")

        with open(CLIENT_CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize models with safe defaults
        distillation_config = self.config.get('distillation', {})
        student_hidden_size = distillation_config.get(
            'student_hidden_size', 384)
        student_num_layers = distillation_config.get('student_num_layers', 6)
        vocab_size = distillation_config.get('vocab_size', 504)

        self.teacher = HubertTeacher(frame_stride=320).to(self.device)
        self.student = HubertStudent(
            hidden_size=student_hidden_size,
            num_layers=student_num_layers,
            vocab_size=vocab_size,
            frame_stride=320
        ).to(self.device)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Setup data
        self._setup_data(data_path)

    def _setup_data(self, data_path: str):
        """Setup data loading."""
        data_path = Path(data_path)
        manifest_path = data_path / "manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}")

        # Load k-means targets
        targets_path = data_path / "kmeans_targets.npy"
        if not targets_path.exists():
            data_root = data_path.parent
            targets_path = data_root / "kmeans_targets.npy"
            if not targets_path.exists():
                raise FileNotFoundError(f"K-means targets not found")

        # Get dataset parameters from config with safe defaults
        distillation_config = self.config.get('distillation', {})
        max_length = distillation_config.get('max_audio_length', 40000)
        sample_rate = distillation_config.get('sample_rate', 16000)
        mask_prob = distillation_config.get('mask_prob', 0.08)
        mask_length = distillation_config.get('mask_length', 10)
        vocab_size = distillation_config.get('vocab_size', 504)
        batch_size = distillation_config.get('batch_size', 16)
        num_workers = distillation_config.get('num_workers', 16)
        pin_memory = distillation_config.get('pin_memory', True)

        # Create datasets
        self.train_dataset = LibriSpeechDistillationDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="train",
            max_length=max_length,
            sample_rate=sample_rate,
            mask_prob=mask_prob,
            mask_length=mask_length,
            vocab_size=vocab_size,
            kmeans_targets_path=str(targets_path)
        )

        self.val_dataset = LibriSpeechDistillationDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="validation",
            max_length=max_length,
            sample_rate=sample_rate,
            mask_prob=mask_prob,
            mask_length=mask_length,
            vocab_size=vocab_size,
            kmeans_targets_path=str(targets_path)
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_parameters(self, get_ins) -> Parameters:
        """Get student model parameters."""
        try:
            params = [val.cpu().numpy()
                      for _, val in self.student.state_dict().items()]
            return ndarrays_to_parameters(params)
        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Error in get_parameters: {e}")
            return ndarrays_to_parameters([])

    def set_parameters(self, parameters: Parameters) -> None:
        """Set student model parameters."""
        if not parameters:
            return

        try:
            param_arrays = parameters_to_ndarrays(parameters)
            state_dict = self.student.state_dict()
            param_keys = list(state_dict.keys())

            for i, (key, param_array) in enumerate(zip(param_keys, param_arrays)):
                if i >= len(param_arrays):
                    break
                param_tensor = torch.tensor(param_array)
                if param_tensor.shape == state_dict[key].shape:
                    state_dict[key] = param_tensor

            self.student.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Error in set_parameters: {e}")

    def fit(self, fit_ins) -> FitRes:
        """Train with knowledge distillation."""
        try:
            parameters = getattr(fit_ins, 'parameters', None)
            if parameters is None:
                return FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=self.get_parameters({}),
                    num_examples=1,
                    metrics={"loss": 0.0, "client_id": str(self.client_id)}
                )

            # Memory cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Set parameters
            self.set_parameters(parameters)

            # Setup optimizer
            distillation_config = self.config.get('distillation', {})
            learning_rate = distillation_config.get('learning_rate', 5e-4)
            weight_decay = distillation_config.get('weight_decay', 0.01)
            optimizer = optim.AdamW(self.student.parameters(
            ), lr=learning_rate, weight_decay=weight_decay)

            # Training
            self.student.train()
            total_loss = 0.0
            num_batches = 0

            local_epochs = distillation_config.get('local_epochs', 10)

            for epoch in range(local_epochs):
                for batch_idx, batch in enumerate(self.train_loader):
                    try:
                        # Move data to device
                        input_values = batch['input_values'].to(
                            self.device, non_blocking=True)
                        targets = batch['targets'].to(
                            self.device, non_blocking=True)
                        mask = batch['mask'].to(self.device, non_blocking=True)

                        # Teacher forward pass
                        with torch.inference_mode():
                            teacher_outputs = self.teacher(
                                input_values, frame_mask=mask)

                        # Student forward pass
                        student_outputs = self.student(
                            input_values, frame_mask=mask)

                        # Loss calculation only on masked frames
                        batch_size, seq_len, vocab_size = student_outputs['predictions'].size(
                        )
                        predictions_flat = student_outputs['predictions'].view(
                            batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        mask_flat = mask.view(batch_size * seq_len)

                        if mask_flat.any():
                            # Get distillation parameters
                            temperature = distillation_config.get(
                                'temperature', 4.0)
                            alpha = distillation_config.get('alpha', 0.7)
                            beta = distillation_config.get('beta', 0.3)

                            # Knowledge distillation loss
                            distill_loss = F.kl_div(
                                F.log_softmax(
                                    predictions_flat[mask_flat] / temperature, dim=-1),
                                F.softmax(teacher_outputs['predictions'].view(
                                    batch_size * seq_len, vocab_size)[mask_flat] / temperature, dim=-1),
                                reduction='batchmean'
                            ) * (temperature ** 2)

                            # Task loss
                            task_loss = F.cross_entropy(
                                predictions_flat[mask_flat], targets_flat[mask_flat])

                            # Combined loss
                            total_loss_batch = alpha * distill_loss + beta * task_loss

                            # Backward pass
                            optimizer.zero_grad(set_to_none=True)
                            total_loss_batch.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.student.parameters(), 1.0)
                            optimizer.step()

                            # Update metrics
                            total_loss += total_loss_batch.item()
                            num_batches += 1

                        # Memory cleanup
                        if self.device.type == "cuda" and (batch_idx % 3 == 0):
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in training batch {batch_idx}: {e}")
                        continue

            # Final cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Calculate average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            params = self.get_parameters({})
            metrics = {
                "loss": float(avg_loss),
                "client_id": str(self.client_id)
            }

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=params,
                num_examples=len(self.train_dataset),
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error in fit method: {e}")
            empty_params = self.get_parameters({})
            return FitRes(
                status=Status(code=Code.OK, message=str(e)),
                parameters=empty_params,
                num_examples=1,
                metrics={"loss": 0.0, "client_id": str(
                    self.client_id), "error": str(e)}
            )

    def evaluate(self, eval_ins) -> EvaluateRes:
        """Evaluate student model on local validation set."""
        try:
            parameters = getattr(eval_ins, 'parameters', None)
            if parameters is None:
                return EvaluateRes(
                    status=Status(code=Code.OK, message="Success"),
                    loss=0.0,
                    num_examples=1,
                    metrics={"accuracy": 0.0}
                )

            # Set parameters
            self.set_parameters(parameters)

            # Evaluation
            self.student.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            num_samples = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    try:
                        input_values = batch['input_values'].to(self.device)
                        targets = batch['targets'].to(self.device)
                        mask = batch['mask'].to(self.device)

                        outputs = self.student(input_values, frame_mask=mask)

                        # Compute loss only on masked frames
                        batch_size, seq_len, vocab_size = outputs['predictions'].size(
                        )
                        predictions_flat = outputs['predictions'].view(
                            batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        mask_flat = mask.view(batch_size * seq_len)

                        if mask_flat.any():
                            loss = F.cross_entropy(
                                predictions_flat[mask_flat], targets_flat[mask_flat])
                            preds = torch.argmax(
                                predictions_flat[mask_flat], dim=-1)
                            accuracy = (
                                preds == targets_flat[mask_flat]).float().mean()

                            total_loss += loss.item()
                            total_accuracy += accuracy.item()
                            num_samples += input_values.size(0)

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in evaluation batch {batch_idx}: {e}")
                        continue

            # Calculate averages
            avg_loss = total_loss / max(len(self.val_loader), 1)
            avg_accuracy = total_accuracy / max(len(self.val_loader), 1)

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=float(avg_loss),
                num_examples=num_samples,
                metrics={"accuracy": float(avg_accuracy)}
            )

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error in evaluate: {e}")
            return EvaluateRes(
                status=Status(code=Code.OK, message=str(e)),
                loss=0.0,
                num_examples=1,
                metrics={"accuracy": 0.0, "error": str(e)}
            )


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    if CLIENT_CONFIG_PATH is None:
        raise ValueError("CLIENT_CONFIG_PATH not set.")

    # Load config from file
    with open(CLIENT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Determine client ID
    node_id = context.node_id
    num_clients = int(config.get('simulation', {}).get('num_supernodes', 2))
    client_id = hash(str(node_id)) % num_clients

    # Setup data path
    data_root_config = config.get('data', {}).get(
        'partitioned_data_root', 'federated_librispeech/data')
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)

    client_data_path = data_root / f"client_{client_id}"
    if not client_data_path.exists():
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

    return FederatedClient(client_id=client_id, data_path=str(client_data_path))


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """Aggregate metrics using weighted average based on number of samples."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_metrics = {}
    for metric_name in metrics[0][1].keys():
        if metric_name in ["client_id", "error"]:
            continue

        try:
            first_value = metrics[0][1][metric_name]
            if isinstance(first_value, (int, float)) and not isinstance(first_value, bool):
                weighted_sum = sum(
                    float(metric[metric_name]) * num_samples for num_samples, metric in metrics)
                weighted_metrics[metric_name] = weighted_sum / total_samples
        except (ValueError, TypeError):
            continue

    return weighted_metrics


def server_fn(context: Context, config_override: Optional[Dict] = None) -> ServerAppComponents:
    """Server function to initialize the federated learning server."""
    if CLIENT_CONFIG_PATH is None:
        raise ValueError("CLIENT_CONFIG_PATH not set.")

    # Load config from file
    with open(CLIENT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Override config if provided
    if config_override:
        for key, value in config_override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    # Initialize student model with safe defaults
    distillation_config = config.get('distillation', {})
    student_hidden_size = distillation_config.get('student_hidden_size', 384)
    student_num_layers = distillation_config.get('student_num_layers', 6)
    vocab_size = distillation_config.get('vocab_size', 504)

    student_model = HubertStudent(
        hidden_size=student_hidden_size,
        num_layers=student_num_layers,
        vocab_size=vocab_size,
        frame_stride=320
    )

    # Convert to parameters
    parameters = parameters_to_ndarrays(ndarrays_to_parameters([
        val.cpu().numpy() for _, val in student_model.state_dict().items()
    ]))

    # Strategy with checkpointing
    strategy = SavingFedAdam(
        save_dir=config.get('checkpointing', {}).get(
            'save_dir', '/home/saadan/scratch/federated_librispeech/src/checkpoints/distillation'),
        state_keys=list(student_model.state_dict().keys()),
        checkpoint_config=config.get('checkpointing', {}),
        fraction_fit=config.get('strategy', {}).get('fraction_fit', 1.0),
        fraction_evaluate=config.get('strategy', {}).get(
            'fraction_evaluate', 1.0),
        min_fit_clients=config.get('strategy', {}).get('min_fit_clients', 2),
        min_evaluate_clients=config.get(
            'strategy', {}).get('min_evaluate_clients', 2),
        min_available_clients=config.get(
            'strategy', {}).get('min_available_clients', 2),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(parameters),
    )

    # Save initial checkpoint
    if parameters is not None:
        strategy.save_initial_checkpoint(ndarrays_to_parameters(parameters))

    # Server config
    num_rounds = distillation_config.get('num_rounds', 10)
    server_config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        config=server_config,
        strategy=strategy,
    )


def main():
    """Main function to run the federated learning simulation."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(
        description="Federated HuBERT Distillation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode")
    parser.add_argument("--num-clients", type=int,
                        default=2, help="Number of clients")
    parser.add_argument("--num-rounds", type=int,
                        default=2, help="Number of rounds")

    args = parser.parse_args()

    if args.simulation:
        print("🚀 Starting Federated HuBERT Knowledge Distillation")
        print(
            f"📊 Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
        print("=" * 50)

        # Set global config path
        global CLIENT_CONFIG_PATH
        CLIENT_CONFIG_PATH = args.config

        # Create directories
        os.makedirs('logs/distillation', exist_ok=True)

        config_path = args.config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            checkpoint_dir = config.get('checkpointing', {}).get(
                'save_dir', '/home/saadan/scratch/federated_librispeech/src/checkpoints/distillation')
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Backend config
        backend_config = {
            "client_resources": {
                "num_cpus": 0.25,
                "num_gpus": 0.05,
                "memory": 1000000000
            }
        }

        try:
            # Create server function with config override
            def server_fn_with_config(context: Context) -> ServerAppComponents:
                config_override = {'distillation': {
                    'num_rounds': args.num_rounds}}
                return server_fn(context, config_override=config_override)

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

    return None


if __name__ == "__main__":
    main()
