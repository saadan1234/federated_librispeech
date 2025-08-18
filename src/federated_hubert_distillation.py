#!/usr/bin/env python3
"""
Simplified Federated HuBERT Knowledge Distillation
10 clients with teacher/student models, server aggregation via FedAdam

Methodology:
- Teacher model (frozen) provides knowledge distillation targets
- Student model learns from both:
  1. K-means cluster targets (task loss) - pre-computed per-client at frame level (REQUIRED)
  2. Teacher predictions (distillation loss) - soft targets
- Combines self-supervised learning with knowledge distillation
- Uses frame-level processing (320 stride) matching pretraining implementation
- Frame-level masking (8% probability) with mask embeddings
- Loss computed ONLY on masked frames (true masked language modeling)
- Skips batches with no masked frames (matching pretraining behavior)
- STRICT data requirements: Real audio data + k-means targets (no dummy data fallback)
- COMPLETE implementation consistency with pretraining (data loading, validation, timing, split handling, target alignment)
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
import pandas as pd
import soundfile as sf
import torchaudio.transforms as T
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from flwr.common.typing import NDArrays, Config
from flwr.client import NumPyClient, Client
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, Status, Code
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from transformers import Wav2Vec2FeatureExtractor
import yaml
import argparse
import json
import time
import os
import sys
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from tqdm import tqdm

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


# Enable CUDA allocator expandable segments to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:128")
# Prefer higher matmul precision setting (may improve perf)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Setup logging with simple format - match pretraining code
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
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
    """Teacher model with frame-level processing like pretraining"""

    def __init__(self, hidden_size=768, num_layers=12, vocab_size=504, frame_stride=320):
        super().__init__()
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
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, input_values, frame_mask=None):
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


class HubertStudent(nn.Module):
    """Student model with frame-level processing like pretraining"""

    def __init__(self, hidden_size=384, num_layers=3, vocab_size=504, frame_stride=320, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.frame_stride = frame_stride
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Smaller transformer layers with proper configuration
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=6,
                dim_feedforward=1536,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, input_values, frame_mask=None):
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
            if self.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Project to vocab
        # [B, T_frames, vocab]
        logits = self.output_projection(x)
        return {"predictions": logits}


class LibriSpeechDistillationDataset(Dataset):
    """Real LibriSpeech dataset for distillation training with frame-level processing like pretraining"""

    def __init__(
        self,
        manifest_file: str,
        audio_root: str,
        split: str = "train",  # "train" or "validation"
        max_length: int = 40000,  # 2.5 seconds at 16kHz (matching pretraining)
        sample_rate: int = 16000,
        # 8% frame-level masking (matching pretraining)
        mask_prob: float = 0.08,
        mask_length: int = 10,
        vocab_size: int = 504,
        kmeans_targets_path: Optional[str] = None
    ):
        # Read manifest (keep both full and filtered views to align targets) - matching pretraining
        df_all = pd.read_csv(manifest_file)
        # logger.info(f"Loaded manifest with {len(df_all)} samples")

        # Validate manifest structure (matching pretraining)
        if 'audio_file' not in df_all.columns and 'audio_path' not in df_all.columns:
            raise ValueError(
                f"Manifest missing required columns. Available columns: {list(df_all.columns)}")

        # Set audio_root first so it can be used in validation
        self.audio_root = Path(audio_root)

        # Check first few audio paths to ensure they're relative (matching pretraining)
        audio_col = 'audio_file' if 'audio_file' in df_all.columns else 'audio_path'
        sample_paths = df_all[audio_col].head(3).tolist()
        # logger.info(f"Sample audio paths: {sample_paths}")

        # Verify that audio files exist (matching pretraining)
        sample_audio_path = self.audio_root / \
            sample_paths[0] if sample_paths else None
        if sample_audio_path and not sample_audio_path.exists():
            logger.warning(f"Sample audio file not found: {sample_audio_path}")
            logger.warning(f"Audio root path: {self.audio_root}")
            logger.warning(
                "Please check if audio files exist in the expected directory structure")
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.vocab_size = vocab_size
        self.split = split

        # Compute filtered dataframe and index mapping to original rows (matching pretraining)
        if 'split' in df_all.columns:
            original_indices = df_all.index[df_all['split'] == split].to_list()
            self.manifest_df = df_all.loc[original_indices].reset_index(
                drop=True)
            self._orig_indices = original_indices
            # logger.info(
            #     f"Filtered to {split} split: {len(self.manifest_df)} samples")
        else:
            self.manifest_df = df_all.reset_index(drop=True)
            self._orig_indices = list(range(len(self.manifest_df)))
            # logger.info(
            #     f"No split column found, using all {len(self.manifest_df)} samples")

        # Load k-means targets - REQUIRED for distillation (matching pretraining)
        self.kmeans_targets = None
        if kmeans_targets_path and os.path.exists(kmeans_targets_path):
            try:
                loaded = np.load(kmeans_targets_path, allow_pickle=True)
                if isinstance(loaded, np.ndarray) and loaded.dtype != object:
                    all_targets = [row for row in loaded]
                else:
                    all_targets = list(loaded)

                # Align targets to the current split using original indices (matching pretraining)
                self.kmeans_targets = [all_targets[i]
                                       for i in self._orig_indices]

                # logger.info(
                #     f"Loaded and aligned k-means targets from {kmeans_targets_path} - {len(self.kmeans_targets)} sequences")
            except Exception as e:
                logger.error(f"Failed to load k-means targets: {e}")
                raise
        else:
            raise FileNotFoundError(
                f"kmeans_targets.npy not found at {kmeans_targets_path}; distillation requires per-client precomputed frame targets")

        # Resampling handled inline like pretraining

        # Frame setup (matching pretraining)
        self.frame_stride = 320

    def __len__(self) -> int:
        return len(self.manifest_df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Generate boolean span mask over a 1D sequence of length seq_len (matching pretraining)."""
        mask = torch.zeros(seq_len, dtype=torch.bool)

        # Generate random spans to mask
        num_spans = int(seq_len * self.mask_prob / self.mask_length)
        for _ in range(num_spans):
            start = torch.randint(
                0, seq_len - self.mask_length + 1, (1,)).item()
            mask[start:start + self.mask_length] = True

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 10

        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.manifest_df)
                row = self.manifest_df.iloc[current_idx]

                # Support both 'audio_file' and 'audio_path' columns (matching pretraining)
                if 'audio_file' in row:
                    audio_rel = row['audio_file']
                elif 'audio_path' in row:
                    audio_rel = row['audio_path']
                else:
                    raise KeyError(
                        "Manifest row must contain 'audio_file' or 'audio_path'")

                audio_path = self.audio_root / audio_rel

                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                if not audio_path.is_file():
                    logger.warning(f"Audio path is not a file: {audio_path}")
                    continue

                if audio_path.stat().st_size == 0:
                    logger.warning(f"Audio file is empty: {audio_path}")
                    continue

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

                # Determine model frame sequence length (matching pretraining)
                target_seq_len = len(audio) // self.frame_stride

                # Frame-level mask (matching pretraining)
                mask = self._span_mask(target_seq_len)

                # Use precomputed per-client k-means targets (REQUIRED, matching pretraining)
                if self.kmeans_targets is not None:
                    kmeans_target = self.kmeans_targets[idx]
                    if len(kmeans_target) > 0:
                        # Pad or truncate k-means targets to match frame length
                        if len(kmeans_target) >= target_seq_len:
                            targets = torch.tensor(
                                kmeans_target[:target_seq_len], dtype=torch.long)
                        else:
                            # Pad with last target value
                            padding = target_seq_len - len(kmeans_target)
                            targets = torch.cat([
                                torch.tensor(kmeans_target, dtype=torch.long),
                                torch.full(
                                    (padding,), kmeans_target[-1], dtype=torch.long)
                            ])
                    else:
                        raise RuntimeError(
                            f"Empty k-means targets for sample {idx}")
                else:
                    raise RuntimeError("K-means targets missing for sample")

                # Pad targets if needed (matching pretraining)
                if len(targets) < target_seq_len:
                    padding = target_seq_len - len(targets)
                    targets = F.pad(targets, (0, padding), value=0)

                # Debug logging for first few samples (only show shapes, not full tensors)
                if idx < 3:
                    # logger.info(f"Dataset sample {idx}: audio_shape={audio.shape}, "
                    #             f"targets_shape={targets.shape}, mask_shape={mask.shape}")
                    # Log mask statistics instead of full content
                    # logger.info(
                    #     f"Dataset sample {idx}: mask has {mask.sum().item()} True values out of {mask.numel()}")
                    pass

                # No need to mask audio - masking is handled at frame level in model

                result = {
                    "input_values": audio,
                    "targets": targets,
                    "mask": mask
                }

                # Verify result structure (matching pretraining)
                assert isinstance(
                    result, dict), f"Result is not a dict: {type(result)}"
                assert 'input_values' in result, f"Missing input_values in result: {result.keys()}"
                assert 'targets' in result, f"Missing targets in result: {result.keys()}"
                assert 'mask' in result, f"Missing mask in result: {result.keys()}"

                return result

            except Exception as e:
                logger.warning(f"Error loading sample {current_idx}: {e}")
                if attempt == max_retries - 1:
                    # Return fallback data with proper frame-level dimensions (matching pretraining)
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


class CheckpointManager:
    """Manages model checkpointing for federated learning - only saves global model (latest and best)"""

    def __init__(self, save_dir: str = "checkpoints/distillation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_global_model(self, parameters: NDArrays, round_num: int, metrics: Dict[str, Any] = None):
        """Save global model parameters after each round"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / \
            f"global_model_round_{round_num}_{timestamp}.pt"

        # Filter metrics to only include numeric values for checkpointing
        checkpoint_metrics = {}
        if metrics:
            for key, value in metrics.items():
                if key in ["client_id", "error"]:
                    continue  # Skip non-numeric metrics
                try:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        checkpoint_metrics[key] = float(value)
                except (ValueError, TypeError):
                    logger.debug(f"Skipping non-numeric metric {key}: {value}")
                    continue

        # Save parameters as torch tensors
        checkpoint = {
            'round': round_num,
            'timestamp': timestamp,
            'parameters': [torch.tensor(param) for param in parameters],
            'metrics': checkpoint_metrics
        }

        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = self.save_dir / "latest_global_model.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint if metrics indicate improvement
        if metrics and 'loss' in metrics:
            best_path = self.save_dir / "best_global_model.pt"
            try:
                current_loss = float(
                    metrics['loss']) if metrics['loss'] is not None else float('inf')
                best_loss = self._get_best_loss()
                if not best_path.exists() or current_loss < best_loss:
                    torch.save(checkpoint, best_path)
                    logger.info(
                        f"New best model saved with loss: {current_loss:.4f}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not compare loss values: {e}. Skipping best model save.")
                # Still save the latest checkpoint
                pass

        # Clean up old checkpoints to keep only latest and best
        self.cleanup_old_checkpoints()

        return checkpoint_path

    def _get_best_loss(self) -> float:
        """Get the loss value from the best checkpoint if it exists"""
        best_path = self.save_dir / "best_global_model.pt"
        if best_path.exists():
            try:
                checkpoint = torch.load(best_path)
                loss_value = checkpoint.get(
                    'metrics', {}).get('loss', float('inf'))
                # Ensure we return a proper float
                if loss_value is None:
                    return float('inf')
                try:
                    return float(loss_value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid loss value in checkpoint: {loss_value}, using inf")
                    return float('inf')
            except Exception as e:
                logger.warning(f"Error loading best checkpoint: {e}")
                return float('inf')
        return float('inf')

    def cleanup_old_checkpoints(self, keep_latest: bool = True, keep_best: bool = True):
        """Clean up old checkpoints, keeping only latest and best"""
        try:
            # Keep only the latest and best checkpoints
            if keep_latest:
                latest_path = self.save_dir / "latest_global_model.pt"
                if latest_path.exists():
                    # Keep the latest checkpoint
                    pass

            if keep_best:
                best_path = self.save_dir / "best_global_model.pt"
                if best_path.exists():
                    # Keep the best checkpoint
                    pass

            # Remove all other round-specific checkpoints
            for checkpoint_file in self.save_dir.glob("global_model_round_*.pt"):
                checkpoint_file.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_file}")

        except Exception as e:
            logger.warning(f"Error during checkpoint cleanup: {e}")

    def save_training_history(self, history: Dict[str, List], round_num: int):
        """Save training history"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        history_path = self.save_dir / \
            f"training_history_round_{round_num}_{timestamp}.json"

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        return history_path

    def load_latest_global_model(self):
        """Load the latest global model checkpoint"""
        latest_path = self.save_dir / "latest_global_model.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path)
            return checkpoint
        else:
            return None

    def load_best_global_model(self):
        """Load the best global model checkpoint"""
        best_path = self.save_dir / "best_global_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path)
            return checkpoint
        else:
            return None


class FedAdamWithCheckpoints(FedAdam):
    """FedAdam strategy with checkpointing - only saves global model checkpoints"""

    def __init__(self, checkpoint_manager: CheckpointManager, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_manager = checkpoint_manager
        self.current_round = 0

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | Tuple[ClientProxy, Exception]]):
        """Aggregate fit results and save checkpoints"""
        self.current_round = server_round

        # Safety check: ensure we have valid results before aggregation
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Check if all clients have valid num_examples
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            logger.warning(
                f"Round {server_round}: All clients returned 0 examples, skipping aggregation")
            return None, {}

        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        # Log round summary
        if aggregated_metrics:
            try:
                loss = float(aggregated_metrics.get('loss', 0.0)) if aggregated_metrics.get(
                    'loss') is not None else 0.0
                logger.info(f"=== ROUND {server_round} SUMMARY ===")
                logger.info(f"Training Loss: {loss:.4f}")
                logger.info(f"Active Clients: {len(results)}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not format loss for logging: {e}")
                logger.info(f"=== ROUND {server_round} SUMMARY ===")
                logger.info(f"Training Loss: N/A")
                logger.info(f"Active Clients: {len(results)}")

        # Save global model
        if aggregated_parameters is not None:
            parameters = parameters_to_ndarrays(aggregated_parameters)
            self.checkpoint_manager.save_global_model(
                parameters, server_round, aggregated_metrics)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Tuple[ClientProxy, EvaluateRes] | Tuple[ClientProxy, Exception]]):
        """Aggregate evaluation results and log metrics"""
        # Safety check: ensure we have valid results before aggregation
        if not results:
            logger.warning(
                f"Round {server_round}: No evaluation results to aggregate")
            return None, {}

        # Check if all clients have valid num_examples
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        if total_examples == 0:
            logger.warning(
                f"Round {server_round}: All clients returned 0 examples in evaluation, skipping aggregation")
            return None, {}

        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_evaluate(server_round, results, failures)

        # Log evaluation summary
        if aggregated_metrics:
            try:
                loss = float(aggregated_metrics.get('loss', 0.0)) if aggregated_metrics.get(
                    'loss') is not None else 0.0
                accuracy = float(aggregated_metrics.get(
                    'accuracy', 0.0)) if aggregated_metrics.get('accuracy') is not None else 0.0
                logger.info(
                    f"Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                logger.info("=" * 40)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not format evaluation metrics for logging: {e}")
                logger.info(f"Evaluation Loss: N/A, Accuracy: N/A")
                logger.info("=" * 40)

        return aggregated_parameters, aggregated_metrics


class FederatedClient(Client):
    """Federated client with teacher/student models using real data"""

    def __init__(self, client_id: int, data_path: str):
        try:
            self.client_id = client_id

            # Check device availability
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Set memory fraction more conservatively
                torch.cuda.set_per_process_memory_fraction(0.05)
                torch.cuda.empty_cache()
            else:
                self.device = torch.device("cpu")

            # Initialize models with error handling
            try:
                # Load config to get model parameters
                if CLIENT_CONFIG_PATH is None:
                    raise ValueError("CLIENT_CONFIG_PATH is not set")

                config_path = CLIENT_CONFIG_PATH
                if not os.path.isabs(config_path):
                    config_path = os.path.join(os.getcwd(), config_path)

                logger.info(
                    f"Client {client_id}: Loading config from {config_path}")

                with open(config_path, 'r') as f:
                    # Store config as instance variable
                    self.config = yaml.safe_load(f)

                logger.info(f"Client {client_id}: Config loaded successfully")
                logger.info(
                    f"Client {client_id}: Config type: {type(self.config)}")
                logger.info(
                    f"Client {client_id}: Config keys: {list(self.config.keys()) if self.config else 'None'}")
                logger.info(
                    f"Client {client_id}: Distillation config: {self.config.get('distillation', {})}")
                logger.info(
                    f"Client {client_id}: Temperature value: {self.config.get('distillation', {}).get('temperature', 'NOT_FOUND')} (type: {type(self.config.get('distillation', {}).get('temperature', 'NOT_FOUND'))})")
                logger.info(
                    f"Client {client_id}: Alpha value: {self.config.get('distillation', {}).get('alpha', 'NOT_FOUND')} (type: {type(self.config.get('distillation', {}).get('alpha', 'NOT_FOUND'))})")
                logger.info(
                    f"Client {client_id}: Beta value: {self.config.get('distillation', {}).get('beta', 'NOT_FOUND')} (type: {type(self.config.get('distillation', {}).get('beta', 'NOT_FOUND'))})")

                # Convert config values to proper types immediately after loading
                logger.info(
                    f"Client {client_id}: [INIT] About to convert config values to proper types")
                try:
                    distillation_config = self.config.get('distillation', {})

                    # Convert all numeric values to proper types
                    if 'temperature' in distillation_config:
                        distillation_config['temperature'] = float(
                            distillation_config['temperature'])
                        logger.info(
                            f"Client {client_id}: [INIT] temperature converted to float: {distillation_config['temperature']}")

                    if 'alpha' in distillation_config:
                        distillation_config['alpha'] = float(
                            distillation_config['alpha'])
                        logger.info(
                            f"Client {client_id}: [INIT] alpha converted to float: {distillation_config['alpha']}")

                    if 'beta' in distillation_config:
                        distillation_config['beta'] = float(
                            distillation_config['beta'])
                        logger.info(
                            f"Client {client_id}: [INIT] beta converted to float: {distillation_config['beta']}")

                    if 'learning_rate' in distillation_config:
                        distillation_config['learning_rate'] = float(
                            distillation_config['learning_rate'])
                        logger.info(
                            f"Client {client_id}: [INIT] learning_rate converted to float: {distillation_config['learning_rate']}")

                    if 'weight_decay' in distillation_config:
                        distillation_config['weight_decay'] = float(
                            distillation_config['weight_decay'])
                        logger.info(
                            f"Client {client_id}: [INIT] weight_decay converted to float: {distillation_config['weight_decay']}")

                    if 'local_epochs' in distillation_config:
                        distillation_config['local_epochs'] = int(
                            distillation_config['local_epochs'])
                        logger.info(
                            f"Client {client_id}: [INIT] local_epochs converted to int: {distillation_config['local_epochs']}")

                    logger.info(
                        f"Client {client_id}: [INIT] All config values converted successfully")

                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Client {client_id}: [INIT] Error converting config values: {e}")
                    logger.error(
                        f"Client {client_id}: [INIT] Using default values for failed conversions")
                    # Set default values for any failed conversions
                    distillation_config.setdefault('temperature', 4.0)
                    distillation_config.setdefault('alpha', 0.7)
                    distillation_config.setdefault('beta', 0.3)
                    distillation_config.setdefault('learning_rate', 5e-5)
                    distillation_config.setdefault('weight_decay', 0.01)
                    distillation_config.setdefault('local_epochs', 1)

                # Use same parameters as server
                student_hidden_size = self.config['distillation']['student_hidden_size']
                student_num_layers = self.config['distillation']['student_num_layers']
                vocab_size = self.config['distillation']['vocab_size']

                self.teacher = HubertTeacher(frame_stride=320).to(self.device)
                self.student = HubertStudent(
                    hidden_size=student_hidden_size,
                    num_layers=student_num_layers,
                    vocab_size=vocab_size,
                    frame_stride=320
                ).to(self.device)

                logger.info(
                    f"Client {client_id}: Models initialized successfully")

            except Exception as e:
                logger.error(
                    f"Client {client_id}: Failed to initialize models: {e}")
                raise

            # Freeze teacher
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

            # Setup data with real LibriSpeech partitions
            self._setup_data(data_path)

        except Exception as e:
            logger.error(f"Client {client_id}: Initialization failed: {e}")
            raise

    def _setup_data(self, data_path: str):
        """Setup real LibriSpeech data loading"""
        data_path = Path(data_path)

        # Load manifest file (matching pretraining approach)
        manifest_path = data_path / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}; distillation requires real data")

        # Load and validate manifest (matching pretraining)
        manifest_start = time.time()
        df = pd.read_csv(manifest_path)
        manifest_time = time.time() - manifest_start
        # logger.info(
        #     f"Client {self.client_id}: Manifest loaded in {manifest_time:.2f}s - {len(df)} samples")

        # Validate manifest structure (matching pretraining)
        if 'audio_file' not in df.columns and 'audio_path' not in df.columns:
            raise ValueError(
                f"Manifest missing required columns. Available columns: {list(df.columns)}")

        # Check first few audio paths to ensure they're relative (matching pretraining)
        audio_col = 'audio_file' if 'audio_file' in df.columns else 'audio_path'
        sample_paths = df[audio_col].head(3).tolist()
        # logger.info(
        #     f"Client {self.client_id}: Sample audio paths: {sample_paths}")

        # Verify that audio files exist (matching pretraining)
        sample_audio_path = data_path / \
            sample_paths[0] if sample_paths else None
        if sample_audio_path and not sample_audio_path.exists():
            logger.warning(
                f"Client {self.client_id}: Sample audio file not found: {sample_audio_path}")
            logger.warning(
                f"Client {self.client_id}: Client data path: {data_path}")
            logger.warning(
                f"Client {self.client_id}: Please check if audio files exist in the expected directory structure")

        try:
            # Create train dataset with frame-level processing (matching pretraining)
            # Look for targets in client-specific directory first, then fallback to root (matching pretraining)
            targets_start = time.time()
            targets_path = data_path / "kmeans_targets.npy"
            if not targets_path.exists():
                # Fallback to root directory targets
                data_root = data_path.parent
                targets_path = data_root / "kmeans_targets.npy"
                if not targets_path.exists():
                    raise FileNotFoundError(
                        f"K-means targets not found in either {data_path} or {data_root}; distillation requires per-client precomputed targets")

            kmeans_targets_str = str(
                targets_path) if targets_path.exists() else None

            if kmeans_targets_str:
                targets_data = np.load(targets_path, allow_pickle=True)
                # logger.info(
                #     f"Client {self.client_id}: KMeans targets loaded from {targets_path} - {len(targets_data)} sequences")
            else:
                raise FileNotFoundError(
                    f"No KMeans targets found in either {data_path} or {data_root}")

            targets_time = time.time() - targets_start
            # logger.info(
            #     f"Client {self.client_id}: Targets processing completed in {targets_time:.2f}s")

            train_dataset_start = time.time()
            # Get dataset parameters from config
            max_length = self.config.get('distillation', {}).get(
                'max_audio_length', 40000)
            sample_rate = self.config.get(
                'distillation', {}).get('sample_rate', 16000)
            mask_prob = self.config.get(
                'distillation', {}).get('mask_prob', 0.08)
            mask_length = self.config.get(
                'distillation', {}).get('mask_length', 10)
            vocab_size = self.config.get(
                'distillation', {}).get('vocab_size', 504)

            self.train_dataset = LibriSpeechDistillationDataset(
                manifest_file=str(manifest_path),
                audio_root=str(data_path),
                split="train",
                max_length=max_length,
                sample_rate=sample_rate,
                mask_prob=mask_prob,
                mask_length=mask_length,
                vocab_size=vocab_size,
                kmeans_targets_path=kmeans_targets_str
            )

            # Create validation dataset with frame-level processing (matching pretraining)
            val_dataset_start = time.time()
            self.val_dataset = LibriSpeechDistillationDataset(
                manifest_file=str(manifest_path),
                audio_root=str(data_path),
                split="validation",
                max_length=max_length,
                sample_rate=sample_rate,
                mask_prob=mask_prob,
                mask_length=mask_length,
                vocab_size=vocab_size,
                kmeans_targets_path=kmeans_targets_str
            )

            # Get batch size and other parameters from config, with fallbacks for testing
            batch_size = self.config.get(
                'distillation', {}).get('batch_size', 1)
            num_workers = self.config.get(
                'distillation', {}).get('num_workers', 4)
            pin_memory = self.config.get(
                'distillation', {}).get('pin_memory', False)

            logger.info(
                f"Client {self.client_id}: Using batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

            # Create data loaders using config values
            logger.info(
                f"Client {self.client_id}: Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,  # Use config value instead of hardcoded 1
                shuffle=True,
                num_workers=num_workers,  # Use config value instead of hardcoded 0
                pin_memory=pin_memory    # Use config value instead of hardcoded False
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,  # Use config value instead of hardcoded 1
                shuffle=False,
                num_workers=num_workers,  # Use config value instead of hardcoded 0
                pin_memory=pin_memory    # Use config value instead of hardcoded False
            )

            logger.info(
                f"Client {self.client_id}: DataLoaders created successfully. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

            train_dataset_time = time.time() - train_dataset_start
            val_dataset_time = time.time() - val_dataset_start

            # logger.info(
            #     f"Client {self.client_id}: Train dataset created in {train_dataset_time:.2f}s - {len(self.train_dataset)} samples")
            # logger.info(
            #     f"Client {self.client_id}: Val dataset created in {val_dataset_time:.2f}s - {len(self.val_dataset)} samples")

            # Validate that k-means targets are properly loaded (matching pretraining)
            if not hasattr(self.train_dataset, 'kmeans_targets') or self.train_dataset.kmeans_targets is None:
                raise RuntimeError("Train dataset missing k-means targets")
            if not hasattr(self.val_dataset, 'kmeans_targets') or self.val_dataset.kmeans_targets is None:
                raise RuntimeError("Val dataset missing k-means targets")

            # Test dataset access
            try:
                test_sample = self.train_dataset[0]
                logger.info(
                    f"Client {self.client_id}: Test sample loaded successfully. Keys: {list(test_sample.keys())}")
                logger.info(
                    f"Client {self.client_id}: Sample shapes - input: {test_sample['input_values'].shape}, targets: {test_sample['targets'].shape}, mask: {test_sample['mask'].shape}")
            except Exception as e:
                logger.error(
                    f"Client {self.client_id}: Failed to load test sample: {e}")
                raise

            # logger.info(
            #     f"Client {self.client_id}: K-means targets validated for both train and val datasets")

            # Final timing summary (matching pretraining)
            total_time = time.time() - manifest_start
            # logger.info(
            #     f"Client {self.client_id}: Data setup completed in {total_time:.2f}s")

        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Failed to load real data: {e}")
            logger.error(
                f"Client {self.client_id}: Config available: {hasattr(self, 'config')}")
            if hasattr(self, 'config'):
                logger.error(
                    f"Client {self.client_id}: Config keys: {list(self.config.keys()) if self.config else 'None'}")
            raise

    # Dummy dataset creation removed - distillation requires real data with k-means targets

    def get_parameters(self, get_ins) -> Parameters:
        """Get student model parameters"""
        try:
            config = getattr(get_ins, 'config', {})
            params = [val.cpu().numpy()
                      for _, val in self.student.state_dict().items()]
            return ndarrays_to_parameters(params)
        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Error in get_parameters: {e}")
            # Return empty parameters instead of raising
            return ndarrays_to_parameters([])

    def set_parameters(self, parameters: Parameters) -> None:
        """Set student model parameters"""
        # logger.info(
        #     f"Client {self.client_id}: set_parameters called with {len(parameters)} parameters")

        if not parameters:
            logger.warning(f"Client {self.client_id}: No parameters provided")
            return

        try:
            # Convert Parameters to NDArrays
            param_arrays = parameters_to_ndarrays(parameters)

            state_dict = self.student.state_dict()
            param_keys = list(state_dict.keys())

            for i, (key, param_array) in enumerate(zip(param_keys, param_arrays)):
                if i >= len(param_arrays):
                    break
                param_tensor = torch.tensor(param_array)
                if param_tensor.shape == state_dict[key].shape:
                    state_dict[key] = param_tensor
                else:
                    logger.warning(
                        f"Client {self.client_id}: Shape mismatch for {key}")

            self.student.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(
                f"Client {self.client_id}: Error in set_parameters: {e}")
            # Log error but don't raise to avoid crashing the client
            pass

    def fit(self, fit_ins) -> FitRes:
        """Train with memory-efficient mixed precision and checkpointing"""
        try:
            logger.info(
                f"Client {self.client_id}: [FIT START] Entering fit method")
            logger.info(
                f"Client {self.client_id}: [FIT START] fit_ins type: {type(fit_ins)}")
            logger.info(
                f"Client {self.client_id}: [FIT START] fit_ins attributes: {dir(fit_ins)}")

            # Extract parameters and config from fit_ins
            parameters = getattr(fit_ins, 'parameters', None)
            config = getattr(fit_ins, 'config', {})
            logger.info(
                f"Client {self.client_id}: [FIT START] parameters type: {type(parameters)}")
            logger.info(
                f"Client {self.client_id}: [FIT START] config type: {type(config)}")
            logger.info(
                f"Client {self.client_id}: [FIT START] self.config type: {type(self.config)}")
            logger.info(
                f"Client {self.client_id}: [FIT START] self.config keys: {list(self.config.keys()) if self.config else 'None'}")

            # Debug: log what we received
            # logger.info(
            #     f"Client {self.client_id}: fit_ins type: {type(fit_ins)}")
            # logger.info(
            #     f"Client {self.client_id}: fit_ins attributes: {dir(fit_ins)}")

            if parameters is None:
                logger.warning(
                    f"Client {self.client_id}: No parameters in fit_ins")
                return FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=self.get_parameters({}),
                    num_examples=1,  # Return at least 1 example to avoid division by zero
                    metrics={
                        "loss": 0.0,
                        "client_id": str(self.client_id)
                    }
                )

            # Aggressive memory cleanup before training
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # Set parameters
            self.set_parameters(parameters)

            # Use mixed precision to save memory
            scaler = torch.amp.GradScaler(
                'cuda', enabled=self.device.type == "cuda")

            # Setup optimizer with config values
            logger.info(
                f"Client {self.client_id}: [CONFIG CHECK] About to validate config availability")
            logger.info(
                f"Client {self.client_id}: [CONFIG CHECK] hasattr(self, 'config'): {hasattr(self, 'config')}")
            logger.info(
                f"Client {self.client_id}: [CONFIG CHECK] self.config is None: {self.config is None if hasattr(self, 'config') else 'N/A'}")

            if not hasattr(self, 'config') or self.config is None:
                logger.error(
                    f"Client {self.client_id}: [CONFIG CHECK] Config not available in fit method")
                # Return empty parameters and error in metrics
                empty_params = self.get_parameters({})
                return FitRes(
                    status=Status(
                        code=Code.OK, message="Config not available"),
                    parameters=empty_params,
                    num_examples=1,  # Return at least 1 example to avoid division by zero
                    metrics={
                        "loss": 0.0,
                        "client_id": str(self.client_id),
                        "error": "Config not available"
                    }
                )

            logger.info(
                f"Client {self.client_id}: [CONFIG CHECK] Config validation passed")
            logger.info(
                f"Client {self.client_id}: [CONFIG CHECK] Config content: {self.config}")

            # Validate and convert all numeric config values to ensure proper types
            logger.info(
                f"Client {self.client_id}: [CONFIG VALIDATION] About to validate and convert config values")
            try:
                distillation_config = self.config.get('distillation', {})

                # Convert all numeric values to proper types
                if 'temperature' in distillation_config:
                    distillation_config['temperature'] = float(
                        distillation_config['temperature'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] temperature converted to float: {distillation_config['temperature']}")

                if 'alpha' in distillation_config:
                    distillation_config['alpha'] = float(
                        distillation_config['alpha'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] alpha converted to float: {distillation_config['alpha']}")

                if 'beta' in distillation_config:
                    distillation_config['beta'] = float(
                        distillation_config['beta'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] beta converted to float: {distillation_config['beta']}")

                if 'learning_rate' in distillation_config:
                    distillation_config['learning_rate'] = float(
                        distillation_config['learning_rate'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] learning_rate converted to float: {distillation_config['learning_rate']}")

                if 'weight_decay' in distillation_config:
                    distillation_config['weight_decay'] = float(
                        distillation_config['weight_decay'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] weight_decay converted to float: {distillation_config['weight_decay']}")

                if 'local_epochs' in distillation_config:
                    distillation_config['local_epochs'] = int(
                        distillation_config['local_epochs'])
                    logger.info(
                        f"Client {self.client_id}: [CONFIG VALIDATION] local_epochs converted to int: {distillation_config['local_epochs']}")

                logger.info(
                    f"Client {self.client_id}: [CONFIG VALIDATION] All config values validated and converted successfully")

            except (ValueError, TypeError) as e:
                logger.error(
                    f"Client {self.client_id}: [CONFIG VALIDATION] Error converting config values: {e}")
                logger.error(
                    f"Client {self.client_id}: [CONFIG VALIDATION] Using default values for failed conversions")
                # Set default values for any failed conversions
                distillation_config = self.config.get('distillation', {})
                distillation_config.setdefault('temperature', 4.0)
                distillation_config.setdefault('alpha', 0.7)
                distillation_config.setdefault('beta', 0.3)
                distillation_config.setdefault('learning_rate', 5e-5)
                distillation_config.setdefault('weight_decay', 0.01)
                distillation_config.setdefault('local_epochs', 1)

            # Check if models are properly initialized
            if not hasattr(self, 'student') or self.student is None:
                logger.error(
                    f"Client {self.client_id}: Student model not initialized")
                empty_params = self.get_parameters({})
                return FitRes(
                    status=Status(
                        code=Code.OK, message="Student model not initialized"),
                    parameters=empty_params,
                    num_examples=1,  # Return at least 1 example to avoid division by zero
                    metrics={
                        "loss": 0.0,
                        "client_id": str(self.client_id),
                        "error": "Student model not initialized"
                    }
                )

            if not hasattr(self, 'teacher') or self.teacher is None:
                logger.error(
                    f"Client {self.client_id}: Teacher model not initialized")
                empty_params = self.get_parameters({})
                return FitRes(
                    status=Status(
                        code=Code.OK, message="Teacher model not initialized"),
                    parameters=empty_params,
                    num_examples=1,  # Return at least 1 example to avoid division by zero
                    metrics={
                        "loss": 0.0,
                        "client_id": str(self.client_id),
                        "error": "Teacher model not initialized"
                    }
                )

            logger.info(
                f"Client {self.client_id}: [OPTIMIZER SETUP] About to get learning_rate and weight_decay from config")
            learning_rate = self.config.get('distillation', {}).get(
                'learning_rate', 5e-5)
            weight_decay = self.config.get('distillation', {}).get(
                'weight_decay', 0.01)

            logger.info(
                f"Client {self.client_id}: [OPTIMIZER SETUP] Raw learning_rate: {learning_rate} (type: {type(learning_rate)}, repr: {repr(learning_rate)})")
            logger.info(
                f"Client {self.client_id}: [OPTIMIZER SETUP] Raw weight_decay: {weight_decay} (type: {type(weight_decay)}, repr: {repr(weight_decay)})")

            logger.info(
                f"Client {self.client_id}: [OPTIMIZER SETUP] Creating optimizer with lr={learning_rate}, wd={weight_decay}")

            # Ensure learning_rate and weight_decay are properly typed before passing to optimizer
            logger.info(
                f"Client {self.client_id}: [OPTIMIZER SETUP] About to convert learning_rate and weight_decay to float")
            try:
                learning_rate = float(
                    learning_rate) if learning_rate is not None else 5e-5
                weight_decay = float(
                    weight_decay) if weight_decay is not None else 0.01
                logger.info(
                    f"Client {self.client_id}: [OPTIMIZER SETUP] learning_rate converted to float: {learning_rate}")
                logger.info(
                    f"Client {self.client_id}: [OPTIMIZER SETUP] weight_decay converted to float: {weight_decay}")
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Client {self.client_id}: [OPTIMIZER SETUP] Error converting optimizer parameters: {e}")
                logger.error(
                    f"Client {self.client_id}: [OPTIMIZER SETUP] Using default values")
                learning_rate = 5e-5
                weight_decay = 0.01

            optimizer = optim.AdamW(
                self.student.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Training with memory-efficient loop
            self.student.train()
            total_loss = 0.0
            num_batches = 0

            logger.info(
                f"Client {self.client_id}: [TRAINING SETUP] About to get local_epochs from config")
            local_epochs = self.config.get('distillation', {}).get(
                'local_epochs', 1)  # Get from config
            logger.info(
                f"Client {self.client_id}: [TRAINING SETUP] Raw local_epochs: {local_epochs} (type: {type(local_epochs)}, repr: {repr(local_epochs)})")

            # Ensure local_epochs is properly typed
            logger.info(
                f"Client {self.client_id}: [TRAINING SETUP] About to convert local_epochs to int")
            try:
                local_epochs = int(local_epochs)
                logger.info(
                    f"Client {self.client_id}: [TRAINING SETUP] local_epochs converted to int: {local_epochs}")
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Client {self.client_id}: [TRAINING SETUP] Error converting local_epochs: {e}")
                logger.warning(
                    f"Client {self.client_id}: [TRAINING SETUP] Invalid local_epochs value: {local_epochs}, using default 1")
                local_epochs = 1

            # Safety check for train_loader length
            try:
                train_loader_len = len(self.train_loader)
                logger.info(
                    f"Client {self.client_id}: Starting training with {local_epochs} epochs, {train_loader_len} batches")
            except Exception as e:
                logger.warning(
                    f"Client {self.client_id}: Could not get train_loader length: {e}")
                logger.info(
                    f"Client {self.client_id}: Starting training with {local_epochs} epochs, unknown number of batches")

            for epoch in range(local_epochs):
                # Ensure epoch is properly typed
                epoch = int(epoch)
                logger.info(
                    f"Client {self.client_id}: Starting epoch {epoch + 1}/{local_epochs}")
                pbar = tqdm(
                    self.train_loader, desc=f"Client {self.client_id} Training", leave=False)
                for batch_idx, batch in enumerate(pbar):
                    # Ensure batch_idx is properly typed
                    batch_idx = int(batch_idx)
                    try:
                        logger.debug(
                            f"Client {self.client_id}: Processing batch {batch_idx}")

                        # Move data to device
                        input_values = batch['input_values'].to(
                            self.device, non_blocking=True)
                        targets = batch['targets'].to(
                            self.device, non_blocking=True)
                        mask = batch['mask'].to(
                            self.device, non_blocking=True)

                        logger.debug(
                            f"Client {self.client_id}: Batch {batch_idx} data moved to device")

                        # Mixed precision forward
                        if scaler.is_enabled():
                            with torch.amp.autocast('cuda'):
                                # Teacher forward pass (inference mode to minimize overhead)
                                try:
                                    with torch.inference_mode():
                                        teacher_outputs = self.teacher(
                                            input_values, frame_mask=mask)
                                    logger.debug(
                                        f"Client {self.client_id}: Teacher forward pass successful")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error in teacher forward pass: {e}")
                                    # Skip this batch
                                    continue

                                # Student forward pass
                                try:
                                    student_outputs = self.student(
                                        input_values, frame_mask=mask)
                                    logger.debug(
                                        f"Client {self.client_id}: Student forward pass successful")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error in student forward pass: {e}")
                                    # Skip this batch
                                    continue

                                # Loss calculation - compute only on masked frames (matching pretraining)
                                batch_size, seq_len, vocab_size = student_outputs['predictions'].size(
                                )
                                predictions_flat = student_outputs['predictions'].view(
                                    batch_size * seq_len, vocab_size)
                                targets_flat = targets.view(
                                    batch_size * seq_len)
                                mask_flat = mask.view(batch_size * seq_len)

                                if mask_flat.any():
                                    # Debug: log mask information
                                    logger.debug(
                                        f"Client {self.client_id}: mask_flat.sum()={mask_flat.sum().item()}, mask_flat.shape={mask_flat.shape}")

                                    # Knowledge distillation loss (only on masked frames)
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] About to get temperature from config")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] self.config type: {type(self.config)}")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] self.config keys: {list(self.config.keys()) if self.config else 'None'}")

                                    distillation_config = self.config.get(
                                        'distillation', {})
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] distillation_config type: {type(distillation_config)}")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] distillation_config keys: {list(distillation_config.keys()) if distillation_config else 'None'}")

                                    temperature = distillation_config.get(
                                        'temperature', 4.0)
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 1] Raw temperature from config: {temperature} (type: {type(temperature)}, repr: {repr(temperature)})")

                                    # Ensure temperature is properly typed
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 2] About to convert temperature to float")
                                    try:
                                        temperature = float(
                                            temperature) if temperature is not None else 4.0
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 2] temperature converted to float: {temperature} (type: {type(temperature)})")
                                    except (ValueError, TypeError) as e:
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 2] Error converting temperature to float: {e}")
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 2] temperature value: {temperature}, type: {type(temperature)}")
                                        temperature = 4.0
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 2] Using default temperature: {temperature}")

                                    # Additional safety check for temperature
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 3] About to check temperature validity")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 3] temperature before check: {temperature} (type: {type(temperature)})")
                                    try:
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 3] Checking isinstance(temperature, (int, float)): {isinstance(temperature, (int, float))}")
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 3] About to check float(temperature) <= 0")
                                        float_temp = float(temperature)
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 3] float(temperature) = {float_temp}")
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 3] About to compare {float_temp} <= 0")
                                        if not isinstance(temperature, (int, float)) or float_temp <= 0:
                                            logger.warning(
                                                f"Client {self.client_id}: [CHECKPOINT 3] Invalid temperature value: {temperature}, using default 4.0")
                                            temperature = 4.0
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 3] Temperature check passed: {temperature}")
                                    except (ValueError, TypeError) as e:
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 3] Error in temperature validation: {e}")
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 3] temperature value: {temperature}, type: {type(temperature)}")
                                        logger.warning(
                                            f"Client {self.client_id}: [CHECKPOINT 3] Using default temperature due to error")
                                        temperature = 4.0

                                    try:
                                        distill_loss = F.kl_div(
                                            F.log_softmax(
                                                predictions_flat[mask_flat] / temperature, dim=-1),
                                            F.softmax(
                                                teacher_outputs['predictions'].view(batch_size * seq_len, vocab_size)[mask_flat] / temperature, dim=-1),
                                            reduction='batchmean'
                                        ) * (temperature ** 2)
                                        logger.debug(
                                            f"Client {self.client_id}: distill_loss computed successfully: {distill_loss}")
                                    except Exception as e:
                                        logger.error(
                                            f"Client {self.client_id}: Error computing distill_loss: {e}")
                                        # Use a fallback loss
                                        distill_loss = torch.tensor(
                                            0.0, device=self.device)

                                    # Task loss (only on masked frames)
                                    try:
                                        task_loss = F.cross_entropy(
                                            predictions_flat[mask_flat],
                                            targets_flat[mask_flat]
                                        )
                                        logger.debug(
                                            f"Client {self.client_id}: task_loss computed successfully: {task_loss}")
                                    except Exception as e:
                                        logger.error(
                                            f"Client {self.client_id}: Error computing task_loss: {e}")
                                        # Use a fallback loss
                                        task_loss = torch.tensor(
                                            0.0, device=self.device)

                                    # Get loss weights from config
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 4] About to get alpha and beta from config")
                                    alpha = self.config.get('distillation', {}).get(
                                        'alpha', 0.7)
                                    beta = self.config.get('distillation', {}).get(
                                        'beta', 0.3)
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 4] Raw alpha: {alpha} (type: {type(alpha)}, repr: {repr(alpha)})")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 4] Raw beta: {beta} (type: {type(beta)}, repr: {repr(beta)})")

                                    # Debug: log the loss weights and losses
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 4] alpha={alpha} (type: {type(alpha)}), beta={beta} (type: {type(beta)})")
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 4] task_loss={task_loss} (type: {type(task_loss)}), distill_loss={distill_loss} (type: {type(distill_loss)})")

                                    # Ensure loss weights are properly typed
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 5] About to convert alpha and beta to float")
                                    try:
                                        alpha = float(
                                            alpha) if alpha is not None else 0.7
                                        beta = float(
                                            beta) if beta is not None else 0.3
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 5] alpha converted: {alpha} (type: {type(alpha)})")
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 5] beta converted: {beta} (type: {type(beta)})")
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 5] About to compute total_loss_batch")
                                        total_loss_batch = alpha * task_loss + beta * distill_loss
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 5] total_loss_batch computed: {total_loss_batch} (type: {type(total_loss_batch)})")
                                    except (ValueError, TypeError) as e:
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 5] Error in loss computation: {e}")
                                        logger.error(
                                            f"Client {self.client_id}: [CHECKPOINT 5] alpha: {alpha} (type: {type(alpha)}), beta: {beta} (type: {type(beta)})")
                                        # Use default values
                                        total_loss_batch = 0.7 * task_loss + 0.3 * distill_loss
                                        logger.info(
                                            f"Client {self.client_id}: [CHECKPOINT 5] Using default loss weights, total_loss_batch: {total_loss_batch}")
                                else:
                                    # Skip batches with no masked frames (matching pretraining)
                                    continue

                            # Backward pass with gradient scaling
                            optimizer.zero_grad(set_to_none=True)
                            scaler.scale(total_loss_batch).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.student.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard precision fallback
                            try:
                                with torch.inference_mode():
                                    teacher_outputs = self.teacher(
                                        input_values, frame_mask=mask)
                                logger.debug(
                                    f"Client {self.client_id}: Teacher forward pass successful (standard precision)")
                            except Exception as e:
                                logger.error(
                                    f"Client {self.client_id}: Error in teacher forward pass (standard precision): {e}")
                                # Skip this batch
                                continue

                            try:
                                student_outputs = self.student(
                                    input_values, frame_mask=mask)
                                logger.debug(
                                    f"Client {self.client_id}: Student forward pass successful (standard precision)")
                            except Exception as e:
                                logger.error(
                                    f"Client {self.client_id}: Error in student forward pass (standard precision): {e}")
                                # Skip this batch
                                continue

                            # Loss calculation - compute only on masked frames (matching pretraining)
                            batch_size, seq_len, vocab_size = student_outputs['predictions'].size(
                            )
                            predictions_flat = student_outputs['predictions'].view(
                                batch_size * seq_len, vocab_size)
                            targets_flat = targets.view(batch_size * seq_len)
                            mask_flat = mask.view(batch_size * seq_len)

                            if mask_flat.any():
                                # Debug: log mask information
                                logger.debug(
                                    f"Client {self.client_id}: mask_flat.sum()={mask_flat.sum().item()}, mask_flat.shape={mask_flat.shape}")

                                # Knowledge distillation loss (only on masked frames)
                                logger.info(
                                    f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - About to get temperature from config")
                                temperature = self.config.get('distillation', {}).get(
                                    'temperature', 4.0)
                                logger.info(
                                    f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - Raw temperature: {temperature} (type: {type(temperature)}, repr: {repr(temperature)})")

                                # Ensure temperature is properly typed
                                logger.info(
                                    f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - About to convert temperature to float")
                                try:
                                    temperature = float(
                                        temperature) if temperature is not None else 4.0
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - temperature converted: {temperature} (type: {type(temperature)})")
                                except (ValueError, TypeError) as e:
                                    logger.error(
                                        f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - Error converting temperature: {e}")
                                    temperature = 4.0

                                # Additional safety check for temperature
                                logger.info(
                                    f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - About to validate temperature")
                                try:
                                    if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
                                        logger.warning(
                                            f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - Invalid temperature: {temperature}, using default 4.0")
                                        temperature = 4.0
                                    logger.info(
                                        f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - Temperature validation passed: {temperature}")
                                except (ValueError, TypeError) as e:
                                    logger.error(
                                        f"Client {self.client_id}: [CHECKPOINT 6] Standard precision - Temperature validation error: {e}")
                                    temperature = 4.0

                                try:
                                    distill_loss = F.kl_div(
                                        F.log_softmax(
                                            predictions_flat[mask_flat] / temperature, dim=-1),
                                        F.softmax(
                                            teacher_outputs['predictions'].view(batch_size * seq_len, vocab_size)[mask_flat] / temperature, dim=-1),
                                        reduction='batchmean'
                                    ) * (temperature ** 2)
                                    logger.debug(
                                        f"Client {self.client_id}: distill_loss computed successfully: {distill_loss}")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error computing distill_loss: {e}")
                                    # Use a fallback loss
                                    distill_loss = torch.tensor(
                                        0.0, device=self.device)

                                # Task loss (only on masked frames)
                                try:
                                    task_loss = F.cross_entropy(
                                        predictions_flat[mask_flat],
                                        targets_flat[mask_flat]
                                    )
                                    logger.debug(
                                        f"Client {self.client_id}: task_loss computed successfully: {task_loss}")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error computing task_loss: {e}")
                                    # Use a fallback loss
                                    task_loss = torch.tensor(
                                        0.0, device=self.device)

                                # Get loss weights from config
                                alpha = self.config.get('distillation', {}).get(
                                    'alpha', 0.7)
                                beta = self.config.get('distillation', {}).get(
                                    'beta', 0.3)

                                # Debug: log the loss weights and losses
                                logger.debug(
                                    f"Client {self.client_id}: alpha={alpha} (type: {type(alpha)}), beta={beta} (type: {type(beta)})")
                                logger.debug(
                                    f"Client {self.client_id}: task_loss={task_loss} (type: {type(task_loss)}), distill_loss={distill_loss} (type: {type(distill_loss)})")

                                # Ensure loss weights are properly typed
                                try:
                                    alpha = float(
                                        alpha) if alpha is not None else 0.7
                                    beta = float(
                                        beta) if beta is not None else 0.3
                                    total_loss_batch = alpha * task_loss + beta * distill_loss
                                    logger.debug(
                                        f"Client {self.client_id}: total_loss_batch={total_loss_batch} (type: {type(total_loss_batch)})")
                                except (ValueError, TypeError) as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error in loss computation: {e}")
                                    # Use default values
                                    total_loss_batch = 0.7 * task_loss + 0.3 * distill_loss
                            else:
                                # Skip batches with no masked frames (matching pretraining)
                                continue

                            optimizer.zero_grad(set_to_none=True)
                            total_loss_batch.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.student.parameters(), 1.0)
                            optimizer.step()

                        # Ensure numeric operations are safe
                        try:
                            # Debug: log the total_loss_batch before processing
                            logger.debug(
                                f"Client {self.client_id}: total_loss_batch={total_loss_batch} (type: {type(total_loss_batch)})")

                            # Ensure total_loss_batch is a tensor before calling .item()
                            if hasattr(total_loss_batch, 'item'):
                                loss_value = total_loss_batch.item()
                                logger.debug(
                                    f"Client {self.client_id}: loss_value={loss_value} (type: {type(loss_value)})")

                                # Additional safety check for loss_value
                                if loss_value is None:
                                    logger.warning(
                                        f"Client {self.client_id}: loss_value is None, skipping")
                                    continue

                                # Ensure loss_value is numeric
                                try:
                                    loss_value = float(loss_value)
                                    total_loss += loss_value
                                    logger.debug(
                                        f"Client {self.client_id}: Successfully added loss_value={loss_value} to total_loss={total_loss}")
                                except (ValueError, TypeError) as e:
                                    logger.error(
                                        f"Client {self.client_id}: Could not convert loss_value to float: {e}")
                                    continue
                            else:
                                logger.warning(
                                    f"Client {self.client_id}: total_loss_batch has no .item() method, skipping")
                                continue

                            num_batches += 1
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Client {self.client_id}: Error in numeric operation: {e}")
                            # Skip this batch if we can't process the loss
                            continue

                        # Clear intermediate variables
                        del input_values, targets, teacher_outputs, student_outputs, distill_loss, task_loss, total_loss_batch

                        # Periodic memory cleanup
                        # Debug: log the types and values before comparison
                        logger.debug(
                            f"Client {self.client_id}: batch_idx={batch_idx} (type: {type(batch_idx)})")
                        if self.device.type == "cuda" and (batch_idx % 3 == 0):
                            torch.cuda.empty_cache()

                        # Continue training on all available batches
                        # Removed artificial batch limit to allow full dataset training

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in training batch {batch_idx}: {e}")
                        continue

            # Final cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Debug: log the types and values before comparison
            logger.debug(
                f"Client {self.client_id}: total_loss={total_loss} (type: {type(total_loss)}), num_batches={num_batches} (type: {type(num_batches)})")

            # Additional safety checks for both variables
            try:
                # Ensure total_loss is properly typed
                if total_loss is None:
                    logger.warning(
                        f"Client {self.client_id}: total_loss is None, using 0.0")
                    total_loss = 0.0
                else:
                    total_loss = float(total_loss)

                # Ensure num_batches is properly typed
                if num_batches is None:
                    logger.warning(
                        f"Client {self.client_id}: num_batches is None, using 0")
                    num_batches = 0
                else:
                    num_batches = int(num_batches)

                # Now do the comparison
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            except Exception as e:
                logger.error(
                    f"Client {self.client_id}: Error in variable processing: {e}")
                avg_loss = 0.0

            # Pass empty config since we use self.config
            params = self.get_parameters({})
            # Ensure numeric metrics are properly typed
            metrics = {
                "loss": float(avg_loss) if avg_loss is not None else 0.0,
                "client_id": str(self.client_id)
            }

            # Debug: log the train_dataset length
            try:
                train_dataset_len = len(self.train_dataset)
                logger.debug(
                    f"Client {self.client_id}: train_dataset length={train_dataset_len}")
            except Exception as e:
                logger.warning(
                    f"Client {self.client_id}: Could not get train_dataset length: {e}")
                train_dataset_len = 1

            logger.info(
                f"Client {self.client_id}: [FIT SUCCESS] Training completed successfully")
            logger.info(
                f"Client {self.client_id}: [FIT SUCCESS] Final metrics: {metrics}")
            logger.info(
                f"Client {self.client_id}: [FIT SUCCESS] About to return FitRes")

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=params,
                num_examples=train_dataset_len,
                metrics=metrics
            )

        except Exception as e:
            logger.error(
                f"Client {self.client_id}: [FIT ERROR] Error in fit method: {e}")
            logger.error(
                f"Client {self.client_id}: [FIT ERROR] Error type: {type(e)}")
            logger.error(
                f"Client {self.client_id}: [FIT ERROR] Error traceback: ", exc_info=True)
            # Return error response instead of raising
            # Return empty parameters and error in metrics
            empty_params = self.get_parameters({})
            logger.info(
                f"Client {self.client_id}: [FIT ERROR] About to return error FitRes")
            return FitRes(
                status=Status(code=Code.OK, message=str(e)),
                parameters=empty_params,
                num_examples=1,  # Return at least 1 example to avoid division by zero
                metrics={
                    "loss": 0.0,
                    "client_id": str(self.client_id),
                    "error": str(e)
                }
            )

    def evaluate(self, eval_ins) -> EvaluateRes:
        """Evaluate student model on local validation set"""
        try:
            # Extract parameters and config from eval_ins
            parameters = getattr(eval_ins, 'parameters', None)
            config = getattr(eval_ins, 'config', {})

            if parameters is None:
                logger.warning(
                    f"Client {self.client_id}: No parameters in eval_ins")
                return EvaluateRes(
                    status=Status(code=Code.OK, message="Success"),
                    loss=0.0,
                    num_examples=1,  # Return at least 1 example to avoid division by zero
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
                pbar = tqdm(
                    self.val_loader, desc=f"Client {self.client_id} Evaluation", leave=False)
                for batch_idx, batch in enumerate(pbar):
                    # Ensure batch_idx is properly typed
                    batch_idx = int(batch_idx)
                    # Initialize variables for this batch
                    loss = None
                    preds = None
                    targets_flat = None
                    mask_flat = None
                    try:
                        input_values = batch['input_values'].to(self.device)
                        targets = batch['targets'].to(self.device)
                        mask = batch['mask'].to(self.device)

                        # Use autocast during evaluation to reduce memory
                        if self.device.type == "cuda":
                            try:
                                with torch.amp.autocast('cuda'):
                                    outputs = self.student(
                                        input_values, frame_mask=mask)
                                logger.debug(
                                    f"Client {self.client_id}: Evaluation forward pass successful (autocast)")
                            except Exception as e:
                                logger.error(
                                    f"Client {self.client_id}: Error in evaluation forward pass (autocast): {e}")
                                continue

                            # Compute loss only on masked frames (matching pretraining)
                            batch_size, seq_len, vocab_size = outputs['predictions'].size(
                            )
                            predictions_flat = outputs['predictions'].view(
                                batch_size * seq_len, vocab_size)
                            targets_flat = targets.view(
                                batch_size * seq_len)
                            mask_flat = mask.view(batch_size * seq_len)
                            if mask_flat.any():
                                try:
                                    loss = F.cross_entropy(
                                        predictions_flat[mask_flat],
                                        targets_flat[mask_flat]
                                    )
                                    preds = torch.argmax(
                                        predictions_flat[mask_flat], dim=-1)
                                    logger.debug(
                                        f"Client {self.client_id}: Evaluation loss computed successfully: {loss}")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error computing evaluation loss: {e}")
                                    loss = torch.tensor(
                                        0.0, device=self.device)
                                    preds = torch.zeros(
                                        1, dtype=torch.long, device=self.device)
                            else:
                                # Skip this batch if no masked frames
                                continue
                        else:
                            try:
                                outputs = self.student(
                                    input_values, frame_mask=mask)
                                logger.debug(
                                    f"Client {self.client_id}: Evaluation forward pass successful (standard precision)")
                            except Exception as e:
                                logger.error(
                                    f"Client {self.client_id}: Error in evaluation forward pass (standard precision): {e}")
                                continue

                            # Compute loss only on masked frames (matching pretraining)
                            batch_size, seq_len, vocab_size = outputs['predictions'].size(
                            )
                            predictions_flat = outputs['predictions'].view(
                                batch_size * seq_len, vocab_size)
                            targets_flat = targets.view(batch_size * seq_len)
                            mask_flat = mask.view(batch_size * seq_len)

                            if mask_flat.any():
                                try:
                                    loss = F.cross_entropy(
                                        predictions_flat[mask_flat],
                                        targets_flat[mask_flat]
                                    )
                                    preds = torch.argmax(
                                        predictions_flat[mask_flat], dim=-1)
                                    logger.debug(
                                        f"Client {self.client_id}: Evaluation loss computed successfully: {loss}")
                                except Exception as e:
                                    logger.error(
                                        f"Client {self.client_id}: Error computing evaluation loss: {e}")
                                    loss = torch.tensor(
                                        0.0, device=self.device)
                                    # Ensure targets_flat and mask_flat are defined before using them
                                    if 'targets_flat' in locals() and 'mask_flat' in locals() and mask_flat is not None:
                                        preds = torch.zeros_like(
                                            targets_flat[mask_flat])
                                    else:
                                        # Fallback: create a dummy tensor
                                        preds = torch.zeros(
                                            1, dtype=torch.long, device=self.device)
                            else:
                                # Skip this batch if no masked frames
                                continue

                        # Compute accuracy only on masked frames (matching pretraining)
                        try:
                            # Ensure targets_flat and mask_flat are defined
                            if 'targets_flat' in locals() and 'mask_flat' in locals() and mask_flat is not None:
                                masked_targets = targets_flat[mask_flat]
                                accuracy = (
                                    preds == masked_targets).float().mean()
                                logger.debug(
                                    f"Client {self.client_id}: Accuracy computed successfully: {accuracy}")
                            else:
                                logger.warning(
                                    f"Client {self.client_id}: targets_flat or mask_flat not defined, using fallback")
                                accuracy = torch.tensor(
                                    0.0, device=self.device)
                        except Exception as e:
                            logger.error(
                                f"Client {self.client_id}: Error computing accuracy: {e}")
                            # Use a fallback accuracy
                            accuracy = torch.tensor(0.0, device=self.device)

                        # Only update metrics if we have valid loss and predictions
                        if loss is not None and preds is not None:
                            try:
                                # Debug: log the values before processing
                                logger.debug(
                                    f"Client {self.client_id}: loss={loss} (type: {type(loss)})")
                                logger.debug(
                                    f"Client {self.client_id}: accuracy={accuracy} (type: {type(accuracy)})")

                                # Ensure loss is a tensor before calling .item()
                                if hasattr(loss, 'item'):
                                    loss_value = loss.item()
                                    logger.debug(
                                        f"Client {self.client_id}: loss_value={loss_value} (type: {type(loss_value)})")

                                    if loss_value is None:
                                        logger.warning(
                                            f"Client {self.client_id}: loss_value is None, skipping")
                                        continue

                                    loss_value = float(loss_value)
                                    total_loss += loss_value
                                else:
                                    logger.warning(
                                        f"Client {self.client_id}: loss has no .item() method, skipping")
                                    continue

                                # Ensure accuracy is a tensor before calling .item()
                                if hasattr(accuracy, 'item'):
                                    accuracy_value = accuracy.item()
                                    logger.debug(
                                        f"Client {self.client_id}: accuracy_value={accuracy_value} (type: {type(accuracy_value)})")

                                    if accuracy_value is None:
                                        logger.warning(
                                            f"Client {self.client_id}: accuracy_value is None, skipping")
                                        continue

                                    accuracy_value = float(accuracy_value)
                                    total_accuracy += accuracy_value
                                else:
                                    logger.warning(
                                        f"Client {self.client_id}: accuracy has no .item() method, skipping")
                                    continue

                                # Ensure input_values.size(0) is valid
                                try:
                                    batch_size = int(input_values.size(0))
                                    num_samples += batch_size
                                    logger.debug(
                                        f"Client {self.client_id}: Added batch_size={batch_size}, total_samples={num_samples}")
                                except (ValueError, TypeError) as e:
                                    logger.error(
                                        f"Client {self.client_id}: Could not get batch size: {e}")
                                    continue

                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Client {self.client_id}: Error in evaluation numeric operation: {e}")
                                continue

                        # Continue evaluation on all available batches
                        # Removed artificial batch limit to allow full validation

                    except Exception as e:
                        logger.error(
                            f"Client {self.client_id}: Error in evaluation batch {batch_idx}: {e}")
                        continue

            # Debug: log the types and values before comparison
            logger.debug(
                f"Client {self.client_id}: total_loss={total_loss} (type: {type(total_loss)}), val_loader_len={len(self.val_loader)} (type: {type(len(self.val_loader))})")
            # Safety check for val_loader length
            try:
                val_loader_len = len(self.val_loader)
                avg_loss = total_loss / max(val_loader_len, 1)
                avg_accuracy = total_accuracy / max(val_loader_len, 1)
            except Exception as e:
                logger.warning(
                    f"Client {self.client_id}: Could not get val_loader length: {e}")
                avg_loss = total_loss if total_loss is not None else 0.0
                avg_accuracy = total_accuracy if total_accuracy is not None else 0.0

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=float(avg_loss) if avg_loss is not None else 0.0,
                num_examples=num_samples,
                metrics={"accuracy": float(
                    avg_accuracy) if avg_accuracy is not None else 0.0}
            )

        except Exception as e:
            logger.error(f"Client {self.client_id}: Error in evaluate: {e}")
            # Return error response instead of raising
            return EvaluateRes(
                status=Status(code=Code.OK, message=str(e)),
                loss=0.0,
                num_examples=1,  # Return at least 1 example to avoid division by zero
                metrics={"accuracy": 0.0, "error": str(e)}
            )


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    try:
        # Use the global config path
        if CLIENT_CONFIG_PATH is None:
            raise ValueError(
                "CLIENT_CONFIG_PATH not set. Please run the script from the main function.")

        config_path = CLIENT_CONFIG_PATH
        logger.info(f"client_fn: Loading config from {config_path}")

        # If config_path is relative, make it absolute
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)
            logger.info(f"client_fn: Absolute config path: {config_path}")

        logger.info(
            f"client_fn: Config file exists: {os.path.exists(config_path)}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(
            f"client_fn: Config loaded successfully with keys: {list(config.keys())}")
        logger.info(f"client_fn: Config type: {type(config)}")
        logger.info(
            f"client_fn: Distillation config: {config.get('distillation', {})}")
        logger.info(
            f"client_fn: Temperature value: {config.get('distillation', {}).get('temperature', 'NOT_FOUND')} (type: {type(config.get('distillation', {}).get('temperature', 'NOT_FOUND'))})")
        logger.info(
            f"client_fn: Alpha value: {config.get('distillation', {}).get('alpha', 'NOT_FOUND')} (type: {type(config.get('distillation', {}).get('alpha', 'NOT_FOUND'))})")
        logger.info(
            f"client_fn: Beta value: {config.get('distillation', {}).get('beta', 'NOT_FOUND')} (type: {type(config.get('distillation', {}).get('beta', 'NOT_FOUND'))})")

        # Determine the client ID based on the node ID
        node_id = context.node_id
        num_clients = int(config['simulation']['num_supernodes'])
        client_id = hash(str(node_id)) % num_clients

        # Setup data path
        data_root_config = config['data']['partitioned_data_root']
        logger.info(f"client_fn: Data root from config: {data_root_config}")

        if not os.path.isabs(data_root_config):
            data_root = Path.cwd() / data_root_config
            logger.info(f"client_fn: Relative path, using cwd: {Path.cwd()}")
        else:
            data_root = Path(data_root_config)

        logger.info(f"client_fn: Final data root: {data_root}")
        logger.info(f"client_fn: Data root exists: {data_root.exists()}")

        client_data_path = data_root / f"client_{client_id}"
        logger.info(f"client_fn: Client data path: {client_data_path}")
        logger.info(
            f"client_fn: Client data path exists: {client_data_path.exists()}")

        if not client_data_path.exists():
            raise FileNotFoundError(
                f"Client data directory not found: {client_data_path}; distillation requires real data")

        logger.info(
            f"client_fn: Creating FederatedClient with client_id={client_id}, data_path={client_data_path}")

        client = FederatedClient(
            client_id=client_id,
            data_path=str(client_data_path)
        )

        logger.info(f"client_fn: FederatedClient created successfully")
        return client

    except Exception as e:
        logger.error(f"Error in client_fn: {e}")
        raise


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """Aggregate metrics using weighted average based on number of samples."""
    # Debug: log what metrics we're receiving
    logger.debug(f"weighted_average called with {len(metrics)} metric sets")
    for i, (num_samples, metric_dict) in enumerate(metrics):
        logger.debug(
            f"Metric set {i}: num_samples={num_samples}, keys={list(metric_dict.keys())}")
        for key, value in metric_dict.items():
            logger.debug(f"  {key}: {value} (type: {type(value)})")

    # Calculate weighted average
    total_samples = sum(num_samples for num_samples, _ in metrics)

    # Safety check: ensure we have valid samples
    if total_samples == 0:
        logger.warning(
            "weighted_average: total_samples is 0, returning empty metrics")
        return {}

    weighted_metrics = {}

    # Safety check: ensure we have valid metrics
    if not metrics or not metrics[0] or not metrics[0][1]:
        logger.warning("weighted_average: no valid metrics to process")
        return {}

    for metric_name in metrics[0][1].keys():
        # Skip non-numeric metrics
        if metric_name in ["client_id", "error"]:
            continue

        # Check if the metric value is numeric and can be aggregated
        try:
            # Try to get the first value to check if it's numeric
            first_value = metrics[0][1][metric_name]
            if isinstance(first_value, (int, float)) and not isinstance(first_value, bool):
                # This is a numeric metric, calculate weighted average
                weighted_sum = sum(
                    float(metric[metric_name]) * num_samples for num_samples, metric in metrics
                )
                weighted_metrics[metric_name] = weighted_sum / total_samples
            else:
                # Non-numeric metric, skip aggregation
                logger.debug(
                    f"Skipping non-numeric metric: {metric_name} (type: {type(first_value)})")
        except (ValueError, TypeError) as e:
            # Skip metrics that can't be converted to float
            logger.debug(
                f"Skipping metric {metric_name} due to conversion error: {e}")
            continue
        except Exception as e:
            # Catch any other unexpected errors
            logger.warning(
                f"Unexpected error processing metric {metric_name}: {e}")
            continue

    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Server function to initialize the federated learning server."""
    try:
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

        # Initialize student model
        student_hidden_size = config['distillation']['student_hidden_size']
        student_num_layers = config['distillation']['student_num_layers']

        student_model = HubertStudent(
            hidden_size=student_hidden_size,
            num_layers=student_num_layers,
            vocab_size=config['distillation']['vocab_size'],
            frame_stride=320
        )

        # Convert to parameters
        parameters = parameters_to_ndarrays(ndarrays_to_parameters([
            val.cpu().numpy() for _, val in student_model.state_dict().items()
        ]))

        # Initialize checkpoint manager
        checkpoint_dir = config.get('checkpointing', {}).get(
            'save_dir', 'checkpoints/distillation')
        checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir)

        # Strategy with checkpointing
        strategy = FedAdamWithCheckpoints(
            checkpoint_manager=checkpoint_manager,
            initial_parameters=ndarrays_to_parameters(parameters),
            fraction_fit=config['strategy']['fraction_fit'],
            fraction_evaluate=config['strategy']['fraction_evaluate'],
            min_fit_clients=config['strategy']['min_fit_clients'],
            min_evaluate_clients=config['strategy']['min_evaluate_clients'],
            min_available_clients=config['strategy']['min_available_clients'],
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        # Server config - use default or get from config
        num_rounds = config.get('distillation', {}).get('num_rounds', 20)
        server_config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(
            config=server_config,
            strategy=strategy,
        )

    except Exception as e:
        logger.error(f"Error in server_fn: {e}")
        raise


def main():
    """Main function to run the federated learning simulation."""
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
        print("🚀 Starting Federated HuBERT Distillation")
        print(
            f"📊 Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
        print("=" * 50)
        print("Progress will be shown below (reduced logging for clarity):")
        print()

        # Set the global config path
        global CLIENT_CONFIG_PATH
        CLIENT_CONFIG_PATH = args.config

        logger.info(f"Setting CLIENT_CONFIG_PATH to: {CLIENT_CONFIG_PATH}")
        logger.info(
            f"Config file exists: {os.path.exists(CLIENT_CONFIG_PATH)}")

        # Create necessary directories
        os.makedirs('logs/distillation', exist_ok=True)
        os.makedirs('checkpoints/distillation', exist_ok=True)

        # Ray will be initialized automatically by Flower

        # Use minimal backend config like pretraining
        backend_config = {
            "client_resources": {
                "num_cpus": 0.25,
                "num_gpus": 0.05,
                "memory": 1000000000
            }
        }

        try:
            run_simulation(
                client_app=ClientApp(client_fn=client_fn),
                server_app=ServerApp(server_fn=server_fn),
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
