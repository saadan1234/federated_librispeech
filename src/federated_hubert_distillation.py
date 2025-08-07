#!/usr/bin/env python3
"""
Simplified Federated HuBERT Knowledge Distillation
10 clients with teacher/student models, server aggregation via FedAdam
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio.transforms as T
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from flwr.common.typing import NDArrays, Config
from flwr.client import NumPyClient
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from transformers import Wav2Vec2FeatureExtractor
import yaml
import argparse
import json
import time
import os
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Enable CUDA allocator expandable segments to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# Prefer higher matmul precision setting (may improve perf)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HubertTeacher(nn.Module):
    """Teacher model"""

    def __init__(self, hidden_size=768, num_layers=12, vocab_size=504):
        super().__init__()
        logger.info(
            f"Initializing teacher model with {hidden_size} hidden size, {num_layers} layers")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

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

        logger.info(
            f"Teacher model created with {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def forward(self, input_values):
        logger.debug(f"Teacher forward: input shape {input_values.shape}")

        # Project input to hidden size
        x = input_values.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.input_projection(x)    # [batch, seq_len, hidden_size]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Project to vocab
        output = self.output_projection(x)

        logger.debug(f"Teacher forward: output shape {output.shape}")
        return {"predictions": output}


class HubertStudent(nn.Module):
    """Student model (smaller than teacher)"""

    def __init__(self, hidden_size=384, num_layers=3, vocab_size=504, use_gradient_checkpointing: bool = True):
        super().__init__()
        logger.info(
            f"Initializing student model with {hidden_size} hidden size, {num_layers} layers")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Smaller transformer layers with proper configuration
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=6,
                dim_feedforward=1536,
                batch_first=True,  # Important for proper input format
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.input_projection = nn.Linear(1, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        logger.info(
            f"Student model created with {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def forward(self, input_values):
        logger.debug(f"Student forward: input shape {input_values.shape}")

        # Project input to hidden size
        x = input_values.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.input_projection(x)    # [batch, seq_len, hidden_size]

        # Pass through transformer layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(layer, x)
            else:
                x = layer(x)

        # Project to vocab
        output = self.output_projection(x)

        logger.debug(f"Student forward: output shape {output.shape}")
        return {"predictions": output}


class LibriSpeechDistillationDataset(Dataset):
    """Real LibriSpeech dataset for distillation training"""

    def __init__(
        self,
        manifest_file: str,
        audio_root: str,
        split: str = "train",  # "train" or "validation"
        max_length: int = 80000,  # 5 seconds at 16kHz
        sample_rate: int = 16000,
        mask_prob: float = 0.15,
        mask_length: int = 10,
        vocab_size: int = 504
    ):
        # Robust manifest file reading with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not os.path.exists(manifest_file):
                    raise FileNotFoundError(
                        f"Manifest file not found: {manifest_file}")

                if os.path.getsize(manifest_file) == 0:
                    raise ValueError(
                        f"Manifest file is empty: {manifest_file}")

                self.manifest_df = pd.read_csv(manifest_file)

                if self.manifest_df.empty:
                    raise ValueError(
                        f"Manifest file has no data: {manifest_file}")

                required_cols = ['audio_path', 'duration']
                missing_cols = [
                    col for col in required_cols if col not in self.manifest_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns in manifest: {missing_cols}")

                break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to read manifest file after {max_retries} attempts: {str(e)}")
                else:
                    logger.warning(
                        f"Attempt {attempt + 1} failed to read manifest file: {str(e)}. Retrying...")
                    import time
                    time.sleep(0.5)

        self.audio_root = Path(audio_root)
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.vocab_size = vocab_size
        self.split = split

        # Filter manifest by split
        if 'split' in self.manifest_df.columns:
            self.manifest_df = self.manifest_df[self.manifest_df['split'] == split]
            logger.info(
                f"Filtered manifest to {split} split: {len(self.manifest_df)} samples")
        else:
            # If no split column, assume all data is for the requested split
            logger.warning(
                f"No 'split' column found in manifest, using all {len(self.manifest_df)} samples for {split}")

        # Resampler for different sample rates
        self.resampler = T.Resample(
            orig_freq=16000, new_freq=sample_rate) if sample_rate != 16000 else None

        logger.info(
            f"Dataset initialized with {len(self.manifest_df)} {split} samples")

    def __len__(self) -> int:
        return len(self.manifest_df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Official HuBERT: non-overlapping span masking"""
        mask = torch.zeros(seq_len, dtype=torch.bool)

        num_mask = int(seq_len * self.mask_prob)
        span_starts = []
        attempts = 0

        while len(span_starts) * self.mask_length < num_mask and attempts < seq_len:
            start = torch.randint(
                0, seq_len - self.mask_length + 1, (1,)).item()
            if not mask[start:start+self.mask_length].any():
                mask[start:start+self.mask_length] = True
                span_starts.append(start)
            attempts += 1

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 10

        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.manifest_df)
                row = self.manifest_df.iloc[current_idx]

                audio_path = self.audio_root / row['audio_path']

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
                audio = torch.tensor(audio, dtype=torch.float32)

                # Resample if needed
                if self.resampler is not None:
                    audio = self.resampler(audio)

                # Normalize audio
                if audio.abs().max() > 0:
                    audio = audio / audio.abs().max()

                # Truncate or pad to max_length
                if len(audio) > self.max_length:
                    start = torch.randint(
                        0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    padding = self.max_length - len(audio)
                    audio = torch.nn.functional.pad(audio, (0, padding))

                # Generate random targets for now (in real implementation, these would be k-means clusters)
                targets = torch.randint(0, self.vocab_size, (len(audio),))

                # Apply span masking
                mask = self._span_mask(len(audio))
                masked_audio = audio.clone()
                masked_audio[mask] = 0.0  # Simple masking

                return {
                    "input_values": masked_audio,
                    "targets": targets,
                    "mask": mask
                }

            except Exception as e:
                logger.warning(f"Error loading sample {current_idx}: {e}")
                if attempt == max_retries - 1:
                    # Return a fallback sample
                    fallback_audio = torch.randn(self.max_length)
                    fallback_targets = torch.randint(
                        0, self.vocab_size, (self.max_length,))
                    fallback_mask = torch.zeros(
                        self.max_length, dtype=torch.bool)

                    return {
                        "input_values": fallback_audio,
                        "targets": fallback_targets,
                        "mask": fallback_mask
                    }

        raise RuntimeError(
            f"Failed to load sample after {max_retries} attempts")


class CheckpointManager:
    """Manages model checkpointing for federated learning"""

    def __init__(self, save_dir: str = "checkpoints/distillation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Checkpoint manager initialized with save directory: {self.save_dir}")

    def save_global_model(self, parameters: NDArrays, round_num: int, metrics: Dict[str, float] = None):
        """Save global model parameters after each round"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / \
            f"global_model_round_{round_num}_{timestamp}.pt"

        # Save parameters as torch tensors
        checkpoint = {
            'round': round_num,
            'timestamp': timestamp,
            'parameters': [torch.tensor(param) for param in parameters],
            'metrics': metrics or {}
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Global model saved to {checkpoint_path}")

        # Save latest checkpoint
        latest_path = self.save_dir / "latest_global_model.pt"
        torch.save(checkpoint, latest_path)

        return checkpoint_path

    def save_client_model(self, client_id: int, parameters: NDArrays, round_num: int, metrics: Dict[str, float] = None):
        """Save client model parameters"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / \
            f"client_{client_id}_round_{round_num}_{timestamp}.pt"

        checkpoint = {
            'client_id': client_id,
            'round': round_num,
            'timestamp': timestamp,
            'parameters': [torch.tensor(param) for param in parameters],
            'metrics': metrics or {}
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Client {client_id} model saved to {checkpoint_path}")

        return checkpoint_path

    def save_training_history(self, history: Dict[str, List], round_num: int):
        """Save training history"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        history_path = self.save_dir / \
            f"training_history_round_{round_num}_{timestamp}.json"

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        logger.info(f"Training history saved to {history_path}")
        return history_path

    def load_latest_global_model(self):
        """Load the latest global model checkpoint"""
        latest_path = self.save_dir / "latest_global_model.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path)
            logger.info(f"Loaded latest global model from {latest_path}")
            return checkpoint
        else:
            logger.warning("No latest global model checkpoint found")
            return None

    def load_client_model(self, client_id: int, round_num: int = None):
        """Load a specific client model checkpoint"""
        if round_num is not None:
            # Look for specific round
            pattern = f"client_{client_id}_round_{round_num}_*.pt"
            checkpoints = list(self.save_dir.glob(pattern))
            if checkpoints:
                # Get latest timestamp
                checkpoint_path = sorted(checkpoints)[-1]
                checkpoint = torch.load(checkpoint_path)
                logger.info(
                    f"Loaded client {client_id} model from {checkpoint_path}")
                return checkpoint

        # Look for any client checkpoint
        pattern = f"client_{client_id}_*.pt"
        checkpoints = list(self.save_dir.glob(pattern))
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]  # Get latest
            checkpoint = torch.load(checkpoint_path)
            logger.info(
                f"Loaded client {client_id} model from {checkpoint_path}")
            return checkpoint

        logger.warning(f"No checkpoint found for client {client_id}")
        return None


class FedAdamWithCheckpoints(FedAdam):
    """FedAdam strategy with checkpointing"""

    def __init__(self, checkpoint_manager: CheckpointManager, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_manager = checkpoint_manager
        self.current_round = 0

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | Tuple[ClientProxy, Exception]]):
        """Aggregate fit results and save checkpoints"""
        self.current_round = server_round

        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        # Save global model
        if aggregated_parameters is not None:
            parameters = parameters_to_ndarrays(aggregated_parameters)
            self.checkpoint_manager.save_global_model(
                parameters, server_round, aggregated_metrics)

        # Save individual client models
        for client_proxy, fit_res in results:
            if fit_res.parameters is not None:
                client_parameters = parameters_to_ndarrays(fit_res.parameters)
                client_id = getattr(client_proxy, 'cid', 'unknown')
                self.checkpoint_manager.save_client_model(
                    client_id, client_parameters, server_round, fit_res.metrics)

        return aggregated_parameters, aggregated_metrics


class FederatedClient(NumPyClient):
    """Federated client with teacher/student models using real data"""

    def __init__(self, client_id: int, data_path: str):
        logger.info(f"Initializing client {client_id}")

        self.client_id = client_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Client {client_id} using device: {self.device}")

        # GPU memory management
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(
                0.09)  # Allow up to 9% of GPU memory per client
            # Enable TF32 for speed on Ampere+ GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            logger.info(f"Client {client_id}: GPU memory fraction set to 0.09")

        # Initialize models
        self.teacher = HubertTeacher().to(self.device)
        self.student = HubertStudent().to(self.device)

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Setup data with real LibriSpeech partitions
        self._setup_data(data_path)

        logger.info(
            f"Client {client_id} initialized with {len(self.train_dataset)} train samples, {len(self.val_dataset)} val samples")

    def _setup_data(self, data_path: str):
        """Setup real LibriSpeech data loading"""
        data_path = Path(data_path)

        # Load manifest file
        manifest_path = data_path / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}")

        # Create train dataset with reduced sequence length
        self.train_dataset = LibriSpeechDistillationDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="train",
            max_length=20000,  # Reduced further to ~1.25 seconds
            sample_rate=16000,
            mask_prob=0.15,
            mask_length=10,
            vocab_size=504
        )

        # Create validation dataset with reduced sequence length
        self.val_dataset = LibriSpeechDistillationDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="validation",
            max_length=20000,  # Reduced further to ~1.25 seconds
            sample_rate=16000,
            mask_prob=0.15,
            mask_length=10,
            vocab_size=504
        )

        # Create data loaders with smaller batch size
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Reduced from 2 to 1
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Reduced from 2 to 1
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def get_parameters(self, config: Config) -> NDArrays:
        """Get student model parameters"""
        logger.info(f"Client {self.client_id}: get_parameters called")
        params = [val.cpu().numpy()
                  for _, val in self.student.state_dict().items()]
        logger.info(
            f"Client {self.client_id}: returning {len(params)} parameter arrays")
        return params

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set student model parameters"""
        logger.info(
            f"Client {self.client_id}: set_parameters called with {len(parameters)} parameters")

        if not parameters:
            logger.warning(f"Client {self.client_id}: No parameters provided")
            return

        state_dict = self.student.state_dict()
        param_keys = list(state_dict.keys())

        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            if i >= len(parameters):
                break
            param_tensor = torch.tensor(param_array)
            if param_tensor.shape == state_dict[key].shape:
                state_dict[key] = param_tensor
            else:
                logger.warning(
                    f"Client {self.client_id}: Shape mismatch for {key}")

        self.student.load_state_dict(state_dict, strict=False)
        logger.info(f"Client {self.client_id}: parameters set successfully")

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """Train with memory-efficient mixed precision and checkpointing"""
        logger.info(f"Client {self.client_id}: fit called")

        # Aggressive memory cleanup before training
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Set parameters
        self.set_parameters(parameters)

        # Use mixed precision to save memory
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        # Setup optimizer
        optimizer = optim.AdamW(self.student.parameters(), lr=5e-5, weight_decay=0.01)

        # Training with memory-efficient loop
        self.student.train()
        total_loss = 0.0
        num_batches = 0
        local_epochs = 2  # Reduce local epochs to lower memory/time per round

        for epoch in range(local_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                input_values = batch['input_values'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)

                # Mixed precision forward
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        # Teacher forward pass (inference mode to minimize overhead)
                        with torch.inference_mode():
                            teacher_outputs = self.teacher(input_values)

                        # Student forward pass
                        student_outputs = self.student(input_values)

                        # Loss calculation
                        distill_loss = F.kl_div(
                            F.log_softmax(student_outputs['predictions'] / 4.0, dim=-1),
                            F.softmax(teacher_outputs['predictions'] / 4.0, dim=-1),
                            reduction='batchmean'
                        ) * (4.0 ** 2)

                        task_loss = F.cross_entropy(
                            student_outputs['predictions'].view(-1, 504),
                            targets.view(-1)
                        )

                        total_loss_batch = 0.7 * task_loss + 0.3 * distill_loss

                    # Backward pass with gradient scaling
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(total_loss_batch).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision fallback
                    with torch.inference_mode():
                        teacher_outputs = self.teacher(input_values)

                    student_outputs = self.student(input_values)

                    distill_loss = F.kl_div(
                        F.log_softmax(student_outputs['predictions'] / 4.0, dim=-1),
                        F.softmax(teacher_outputs['predictions'] / 4.0, dim=-1),
                        reduction='batchmean'
                    ) * (4.0 ** 2)

                    task_loss = F.cross_entropy(
                        student_outputs['predictions'].view(-1, 504),
                        targets.view(-1)
                    )

                    total_loss_batch = 0.7 * task_loss + 0.3 * distill_loss

                    optimizer.zero_grad(set_to_none=True)
                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()

                total_loss += float(total_loss_batch.item())
                num_batches += 1

                # Clear intermediate variables
                del input_values, targets, teacher_outputs, student_outputs, distill_loss, task_loss, total_loss_batch

                # Periodic memory cleanup
                if self.device.type == "cuda" and (batch_idx % 5 == 0):
                    torch.cuda.empty_cache()

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        params = self.get_parameters(config)
        metrics = {"loss": avg_loss, "client_id": self.client_id}

        return params, len(self.train_dataset), metrics

    def _save_client_checkpoint(self, config: Config):
        """Save client model checkpoint"""
        try:
            checkpoint_dir = Path("checkpoints/distillation/clients")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / \
                f"client_{self.client_id}_{timestamp}.pt"

            checkpoint = {
                'client_id': self.client_id,
                'timestamp': timestamp,
                'model_state_dict': self.student.state_dict(),
                'config': config
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(
                f"Client {self.client_id} checkpoint saved to {checkpoint_path}")

        except Exception as e:
            logger.warning(
                f"Failed to save client {self.client_id} checkpoint: {e}")

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate student model on local validation set"""
        logger.info(f"Client {self.client_id}: evaluate called")

        # Set parameters
        self.set_parameters(parameters)

        # Evaluation
        self.student.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)

                # Use autocast during evaluation to reduce memory
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.student(input_values)
                        loss = F.cross_entropy(
                            outputs['predictions'].view(-1, 504),
                            targets.view(-1)
                        )
                        preds = torch.argmax(outputs['predictions'], dim=-1)
                else:
                    outputs = self.student(input_values)
                    loss = F.cross_entropy(
                        outputs['predictions'].view(-1, 504),
                        targets.view(-1)
                    )
                    preds = torch.argmax(outputs['predictions'], dim=-1)

                accuracy = (preds == targets).float().mean()

                total_loss += float(loss.item())
                total_accuracy += float(accuracy.item())
                num_samples += input_values.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)

        logger.info(
            f"Client {self.client_id}: evaluation completed, loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

        return avg_loss, num_samples, {"accuracy": avg_accuracy}


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    config_path = "configs/distillation_config.yaml"

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
    )


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics using weighted average based on number of samples."""
    logger.info(f"weighted_average called with {len(metrics)} client results")

    # Calculate weighted average
    total_samples = sum(num_samples for num_samples, _ in metrics)
    weighted_metrics = {}

    for metric_name in metrics[0][1].keys():
        if metric_name == "client_id":
            continue
        weighted_sum = sum(
            metric[metric_name] * num_samples for num_samples, metric in metrics
        )
        weighted_metrics[metric_name] = weighted_sum / total_samples

    logger.info(f"weighted_average returning {weighted_metrics}")
    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Server function to initialize the federated learning server."""
    logger.info("server_fn called")

    config_path = "configs/distillation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize student model
    student_hidden_size = config['distillation']['student_hidden_size']
    student_num_layers = config['distillation']['student_num_layers']

    logger.info(
        f"Initializing student model with {student_hidden_size} hidden size, {student_num_layers} layers")

    student_model = HubertStudent(
        hidden_size=student_hidden_size,
        num_layers=student_num_layers,
        vocab_size=config['distillation']['vocab_size']
    )

    logger.info(
        f"Student model created with {sum(p.numel() for p in student_model.parameters())/1e6:.1f}M parameters")

    # Convert to parameters
    parameters = parameters_to_ndarrays(ndarrays_to_parameters([
        val.cpu().numpy() for _, val in student_model.state_dict().items()
    ]))

    logger.info(f"Created initial parameters with {len(parameters)} arrays")

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

    logger.info(f"Server config: {num_rounds} rounds")
    logger.info("server_fn returning ServerAppComponents")
    return ServerAppComponents(
        config=server_config,
        strategy=strategy,
    )


def main():
    """Main function to run the federated learning simulation."""
    logger.info("main function called")

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
                        default=10, help="Number of clients")
    parser.add_argument("--num-rounds", type=int,
                        default=3, help="Number of rounds")

    args = parser.parse_args()

    logger.info(
        f"Arguments: config={args.config}, simulation={args.simulation}, num_clients={args.num_clients}, num_rounds={args.num_rounds}")

    if args.simulation:
        logger.info(
            f"Starting federated learning simulation with {args.num_clients} clients")

        # Run simulation with adjusted resources to prevent OOM
        backend_config = {
            "client_resources": {
                "num_cpus": 0.4,
                "num_gpus": 0.2,  # Reduce concurrency: at most 5 clients per GPU
                "memory": 2000000000  # 2GB per client
            },
            "init_args": {
                "log_to_driver": False,
                "configure_logging": True,
                "local_mode": False,
                "logging_level": 30
            }
        }

        run_simulation(
            client_app=ClientApp(client_fn=client_fn),
            server_app=ServerApp(server_fn=server_fn),
            num_supernodes=args.num_clients,
            backend_config=backend_config
        )

        logger.info("Federated learning simulation completed")

    logger.info("main function returning None")
    return None


if __name__ == "__main__":
    main()
