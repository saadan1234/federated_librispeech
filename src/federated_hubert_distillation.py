#!/usr/bin/env python3
"""
Research-Grade Federated HuBERT Knowledge Distillation
Optimized for memory efficiency and performance
"""

import sys
import argparse
import yaml
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, Status, Code
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAdam
from flwr.client import Client
from flwr.common.typing import NDArrays
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import torchaudio.transforms as T
import soundfile as sf
import torch
import signal
import time
import math
import logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

# Set CUDA memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


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
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


def log_memory_usage(device, stage=""):
    """Log current GPU memory usage."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        free = torch.cuda.get_device_properties(
            device).total_memory / 1024**3 - reserved
        logger.info(
            f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")


# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Changed from WARNING to INFO for debugging
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
    """Student model with frame-level processing matching pretraining implementation"""

    def __init__(self, hidden_size=256, num_layers=4, vocab_size=504, frame_stride=320, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.frame_stride = frame_stride

        # Ensure hidden_size is divisible by num_heads
        # Use a reasonable number of heads that divides the hidden size
        if hidden_size >= 64:
            num_heads = 8  # 256 is divisible by 8
        elif hidden_size >= 32:
            num_heads = 4  # Fallback for smaller sizes
        else:
            num_heads = 2  # Minimum fallback

        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            # Adjust hidden_size to be divisible by num_heads
            adjusted_hidden_size = ((hidden_size // num_heads) + 1) * num_heads
            logger.warning(
                f"Hidden size {hidden_size} not divisible by {num_heads}, adjusting to {adjusted_hidden_size}")
            hidden_size = adjusted_hidden_size
            self.hidden_size = hidden_size

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=10000)

        # Input projection
        self.input_projection = nn.Linear(1, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,  # 4x hidden size for feedforward
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            for layer in self.layers:
                layer.use_checkpoint = True

    def forward(self, input_values, frame_mask=None):
        """Forward pass with frame-level processing."""
        batch_size, seq_len = input_values.shape

        # Reshape to frame-level processing
        num_frames = seq_len // self.frame_stride
        if num_frames * self.frame_stride < seq_len:
            num_frames += 1

        # Pad if necessary
        padded_length = num_frames * self.frame_stride
        if padded_length > seq_len:
            padding = torch.zeros(
                batch_size, padded_length - seq_len, device=input_values.device)
            input_values = torch.cat([input_values, padding], dim=1)

        # Reshape to frames
        x = input_values.view(batch_size, num_frames, self.frame_stride)

        # Take mean of each frame
        x = x.mean(dim=2, keepdim=True)  # [batch_size, num_frames, 1]

        # Project to hidden size
        x = self.input_projection(x)  # [batch_size, num_frames, hidden_size]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Project to vocabulary
        # [batch_size, num_frames, vocab_size]
        logits = self.output_projection(x)

        return {
            'predictions': logits,
            'hidden_states': x
        }


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
        """Aggregate fit results and save checkpoint."""
        if CLIENT_CONFIG_PATH:
            print(f"\n🎯 ROUND {server_round} COMPLETED!")

        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        # Track current loss for best checkpoint saving
        if aggregated_metrics and 'loss' in aggregated_metrics:
            self.current_loss = aggregated_metrics['loss']

        # Save checkpoint after aggregation
        if aggregated_parameters is not None:
            self.save_checkpoint(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def debug_parameters(self, parameters, stage=""):
        """Debug method to inspect parameters."""
        logger.info(f"=== DEBUG PARAMETERS {stage} ===")
        logger.info(f"Type: {type(parameters)}")
        logger.info(f"Dir: {dir(parameters)}")

        if hasattr(parameters, 'tensors'):
            logger.info(f"Has tensors: {len(parameters.tensors)}")
            for i, tensor in enumerate(parameters.tensors):
                logger.info(
                    f"  Tensor {i}: {type(tensor)}, shape: {tensor.shape}")

        if hasattr(parameters, '__len__'):
            logger.info(f"Length: {len(parameters)}")
            for i, param in enumerate(parameters):
                logger.info(
                    f"  Param {i}: {type(param)}, shape: {getattr(param, 'shape', 'N/A')}")

        logger.info("=== END DEBUG ===")

    def save_initial_checkpoint(self, parameters):
        """Save initial checkpoint."""
        if parameters is None:
            return

        try:
            initial_path = self.save_dir / "initial_state.pt"
            self._save_checkpoint(parameters, initial_path, 0)
        except Exception as e:
            logger.warning(f"Error saving initial checkpoint: {e}")

    def save_checkpoint(self, parameters, server_round):
        """Save checkpoint for current round."""
        if parameters is None:
            logger.warning("No parameters provided for checkpoint saving")
            return

        logger.info(f"Saving checkpoint for round {server_round}")

        # Debug parameters
        self.debug_parameters(parameters, f"ROUND_{server_round}")

        try:
            # Convert Parameters to proper format for saving
            if hasattr(parameters, 'tensors'):
                logger.info("Parameters has tensors attribute")
                # Parameters object with tensors
                param_tensors = parameters.tensors
                logger.info(
                    f"Found {len(param_tensors)} tensors in Parameters.tensors")
            else:
                logger.info("Parameters does not have tensors attribute")
                # Try to convert to list of tensors
                param_tensors = [torch.tensor(p) if not isinstance(
                    p, torch.Tensor) else p for p in parameters]
                logger.info(f"Converted to {len(param_tensors)} tensors")

            # Save latest checkpoint
            latest_path = self.save_dir / "latest_state.pt"
            logger.info(f"Saving latest checkpoint to {latest_path}")
            self._save_checkpoint(parameters, latest_path, server_round)

            # Save round-specific checkpoint
            round_path = self.save_dir / f"round_{server_round:03d}_state.pt"
            logger.info(f"Saving round checkpoint to {round_path}")
            self._save_checkpoint(parameters, round_path, server_round)

            # Save best checkpoint if loss improved
            if hasattr(self, 'current_loss') and self.current_loss < self.best_loss:
                logger.info(
                    f"New best loss: {self.current_loss} < {self.best_loss}")
                self.best_loss = self.current_loss
                self.best_round = server_round
                best_path = self.save_dir / "best_state.pt"
                logger.info(f"Saving best checkpoint to {best_path}")
                self._save_checkpoint(parameters, best_path, server_round)

        except Exception as e:
            logger.warning(f"Error saving checkpoint: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")

    def _save_checkpoint(self, parameters, path, server_round):
        """Save model checkpoint."""
        try:
            logger.info(f"Attempting to save checkpoint to {path}")

            # Convert parameters to tensors using the dedicated method
            param_list = self._convert_parameters_to_tensors(parameters)

            if not param_list:
                logger.error("No parameters converted, cannot save checkpoint")
                return

            logger.info(f"Successfully converted {len(param_list)} parameters")

            state_dict = OrderedDict()

            # Create state dict from converted parameters
            for i, key in enumerate(self.state_keys):
                if i < len(param_list):
                    param = param_list[i]
                    logger.info(
                        f"Processing parameter {i}: {key}, type: {type(param)}, shape: {getattr(param, 'shape', 'N/A')}")

                    if isinstance(param, np.ndarray):
                        state_dict[key] = torch.tensor(param)
                    else:
                        state_dict[key] = torch.tensor(param)

            logger.info(f"Saving state_dict with {len(state_dict)} keys")
            torch.save({
                'round': server_round,
                'state_dict': state_dict,
                'best_loss': self.best_loss,
                'best_round': self.best_round,
                'timestamp': time.time()
            }, path)
            logger.info(f"Successfully saved checkpoint to {path}")

        except Exception as e:
            logger.warning(f"Could not save checkpoint to {path}: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")

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

    def _convert_parameters_to_tensors(self, parameters):
        """Convert parameters to tensors, handling various formats."""
        try:
            logger.info(f"Converting parameters of type: {type(parameters)}")

            if hasattr(parameters, 'tensors'):
                logger.info("Using Parameters.tensors")
                # Check if tensors are actually bytes objects
                if parameters.tensors and isinstance(parameters.tensors[0], bytes):
                    logger.warning(
                        "Parameters.tensors contains bytes objects - converting them")
                    converted_params = []
                    for i, tensor_bytes in enumerate(parameters.tensors):
                        if isinstance(tensor_bytes, bytes):
                            logger.info(
                                f"Converting bytes tensor {i} to numpy array")
                            param_array = np.frombuffer(
                                tensor_bytes, dtype=np.float32)
                            converted_params.append(param_array)
                        else:
                            converted_params.append(tensor_bytes.numpy())
                    return converted_params
                else:
                    # Normal case - tensors are actual tensors
                    return [tensor.numpy() for tensor in parameters.tensors]

            elif hasattr(parameters, '__len__'):
                logger.info("Converting iterable parameters")
                param_list = list(parameters)
                logger.info(f"Found {len(param_list)} parameters")

                # Check if any are bytes
                bytes_count = sum(
                    1 for p in param_list if isinstance(p, bytes))
                if bytes_count > 0:
                    logger.warning(
                        f"Found {bytes_count} bytes parameters out of {len(param_list)}")

                converted_params = []
                for i, param in enumerate(param_list):
                    if isinstance(param, bytes):
                        logger.info(
                            f"Converting bytes parameter {i} to numpy array")
                        param_array = np.frombuffer(param, dtype=np.float32)
                        converted_params.append(param_array)
                    elif hasattr(param, 'numpy'):
                        converted_params.append(param.numpy())
                    else:
                        converted_params.append(param)

                return converted_params

            else:
                logger.warning(
                    "Unknown parameter format, attempting fallback conversion")
                return [p for p in parameters]

        except Exception as e:
            logger.error(f"Error converting parameters: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []


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
        student_hidden_size = int(distillation_config.get(
            'student_hidden_size', 256))  # Changed from 384 to 256
        student_num_layers = int(
            distillation_config.get('student_num_layers', 4))  # Changed from 6 to 4
        vocab_size = int(distillation_config.get('vocab_size', 504))

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
        max_length = int(distillation_config.get('max_audio_length', 40000))
        sample_rate = int(distillation_config.get('sample_rate', 16000))
        mask_prob = float(distillation_config.get('mask_prob', 0.08))
        mask_length = int(distillation_config.get('mask_length', 10))
        vocab_size = int(distillation_config.get('vocab_size', 504))
        batch_size = int(distillation_config.get(
            'batch_size', 8))  # Changed from 16 to 8
        # Limit to 8 to avoid warnings
        num_workers = min(int(distillation_config.get('num_workers', 16)), 8)
        pin_memory = bool(distillation_config.get('pin_memory', True))

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
            learning_rate = float(
                distillation_config.get('learning_rate', 5e-4))
            weight_decay = float(distillation_config.get('weight_decay', 0.01))
            optimizer = optim.AdamW(self.student.parameters(
            ), lr=learning_rate, weight_decay=weight_decay)

            # Training
            self.student.train()
            total_loss = 0.0
            num_batches = 0
            gradient_accumulation_steps = int(
                distillation_config.get('gradient_accumulation_steps', 4))

            local_epochs = int(distillation_config.get('local_epochs', 10))

            # Log initial memory usage
            log_memory_usage(
                self.device, f"Client {self.client_id} - Start of training")

            for epoch in range(local_epochs):
                for batch_idx, batch in enumerate(self.train_loader):
                    try:
                        # Log memory every 10 batches
                        if batch_idx % 10 == 0:
                            log_memory_usage(
                                self.device, f"Client {self.client_id} - Batch {batch_idx}")
                        input_values = batch['input_values'].to(
                            self.device, non_blocking=True)
                        targets = batch['targets'].to(
                            self.device, non_blocking=True)
                        mask = batch['mask'].to(self.device, non_blocking=True)

                        with torch.inference_mode():
                            teacher_outputs = self.teacher(
                                input_values, frame_mask=mask)

                        student_outputs = self.student(
                            input_values, frame_mask=mask)

                        batch_size, seq_len, vocab_size = student_outputs['predictions'].size(
                        )
                        predictions_flat = student_outputs['predictions'].view(
                            batch_size * seq_len, vocab_size)
                        targets_flat = targets.view(batch_size * seq_len)
                        mask_flat = mask.view(batch_size * seq_len)

                        if mask_flat.any():
                            # Get distillation parameters with proper type conversion
                            temperature = float(
                                distillation_config.get('temperature', 4.0))
                            alpha = float(
                                distillation_config.get('alpha', 0.7))
                            beta = float(distillation_config.get('beta', 0.3))

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

                            # Scale loss for gradient accumulation
                            total_loss_batch = total_loss_batch / gradient_accumulation_steps
                            total_loss_batch.backward()

                            # Gradient accumulation
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.student.parameters(), 1.0)
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)

                            total_loss += total_loss_batch.item() * gradient_accumulation_steps
                            num_batches += 1

                        # Clear memory more frequently
                        if self.device.type == "cuda" and (batch_idx % 2 == 0):
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in training batch {batch_idx}: {e}")
                        continue

                # Step optimizer at end of epoch if there are remaining gradients
                if optimizer.param_groups[0]['params'][0].grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

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
    student_hidden_size = int(
        distillation_config.get('student_hidden_size', 256))  # Changed from 384 to 256
    student_num_layers = int(distillation_config.get(
        'student_num_layers', 4))  # Changed from 6 to 4
    vocab_size = int(distillation_config.get('vocab_size', 504))

    student_model = HubertStudent(
        hidden_size=student_hidden_size,
        num_layers=student_num_layers,
        vocab_size=vocab_size,
        frame_stride=320
    )

    # Convert to parameters
    parameters = ndarrays_to_parameters([
        val.cpu().numpy() for _, val in student_model.state_dict().items()
    ])

    # Strategy with checkpointing
    strategy = SavingFedAdam(
        save_dir=config.get('checkpointing', {}).get(
            'save_dir', '/home/saadan/scratch/federated_librispeech/src/checkpoints/distillation'),
        state_keys=list(student_model.state_dict().keys()),
        checkpoint_config=config.get('checkpointing', {}),
        fraction_fit=float(config.get(
            'strategy', {}).get('fraction_fit', 1.0)),
        fraction_evaluate=float(config.get('strategy', {}).get(
            'fraction_evaluate', 1.0)),
        min_fit_clients=int(config.get(
            'strategy', {}).get('min_fit_clients', 2)),
        min_evaluate_clients=int(config.get(
            'strategy', {}).get('min_evaluate_clients', 2)),
        min_available_clients=int(config.get(
            'strategy', {}).get('min_available_clients', 2)),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Save initial checkpoint
    if parameters is not None:
        strategy.save_initial_checkpoint(parameters)

    # Server config
    num_rounds = int(distillation_config.get('num_rounds', 10))
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
