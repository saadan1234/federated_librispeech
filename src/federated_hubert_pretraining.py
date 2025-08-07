#!/usr/bin/env python3
"""
Simplified Federated HuBERT Pretraining
10 clients with HubertBase model, server aggregation via FedAdam
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
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from transformers import Wav2Vec2FeatureExtractor
import yaml
import argparse
import json
import time
import os
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HubertBase(nn.Module):
    """HubertBase model for pretraining"""

    def __init__(self, hidden_size=768, num_layers=12, vocab_size=504):
        super().__init__()
        logger.info(
            f"Initializing HubertBase model with {hidden_size} hidden size, {num_layers} layers")

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
            f"HubertBase model created with {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def forward(self, input_values):
        logger.debug(f"HubertBase forward: input shape {input_values.shape}")

        # Project input to hidden size
        x = input_values.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.input_projection(x)    # [batch, seq_len, hidden_size]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Project to vocab
        output = self.output_projection(x)

        logger.debug(f"HubertBase forward: output shape {output.shape}")
        return {"predictions": output}


class LibriSpeechPretrainingDataset(Dataset):
    """Dataset for LibriSpeech pretraining with masking"""

    def __init__(
        self,
        manifest_file: str,
        audio_root: str,
        split: str = "train",  # "train" or "validation"
        max_length: int = 40000,  # 2.5 seconds at 16kHz
        sample_rate: int = 16000,
        mask_prob: float = 0.08,
        mask_length: int = 10,
        vocab_size: int = 504
    ):
        # Robust manifest file reading with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.df = pd.read_csv(manifest_file)
                logger.info(f"Loaded manifest with {len(self.df)} samples")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load manifest after {max_retries} attempts: {e}")
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        self.audio_root = Path(audio_root)
        self.split = split
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.vocab_size = vocab_size

        # Filter by split if specified
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            logger.info(f"Filtered to {split} split: {len(self.df)} samples")

        # Generate random targets for pretraining (simplified approach)
        # Approximate sequence length after feature extraction
        self.targets = np.random.randint(
            0, vocab_size, size=(len(self.df), max_length // 320))

        logger.info(f"Dataset initialized with {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        """Generate span mask for pretraining"""
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
                audio_path = self.audio_root / row['audio_file']

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

                # Generate mask
                mask = self._span_mask(len(audio))

                # Get targets (truncate to match sequence length)
                # Approximate feature sequence length
                target_seq_len = len(audio) // 320
                targets = torch.tensor(
                    self.targets[idx][:target_seq_len], dtype=torch.long)

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
                        self.max_length // 320, dtype=torch.long)
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

    def __init__(self, save_dir: str = "checkpoints/pretraining"):
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
        pattern = f"client_{client_id}_round_*.pt"
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
    """Federated client with HubertBase model using real data"""

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
                0.08)  # Further reduced to 8% of GPU memory per client
            logger.info(f"Client {client_id}: GPU memory fraction set to 0.08")

        # Initialize model
        self.model = HubertBase().to(self.device)

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
        self.train_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="train",
            max_length=40000,  # Reduced from 80000 to 40000 (2.5 seconds)
            sample_rate=16000,
            mask_prob=0.08,
            mask_length=10,
            vocab_size=504
        )

        # Create validation dataset with reduced sequence length
        self.val_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(manifest_path),
            audio_root=str(data_path),
            split="validation",
            max_length=40000,  # Reduced from 80000 to 40000 (2.5 seconds)
            sample_rate=16000,
            mask_prob=0.08,
            mask_length=10,
            vocab_size=504
        )

        # Create data loaders with smaller batch size
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=2,  # Reduced batch size
            shuffle=True,
            num_workers=0,  # Reduced from 2 to 0 to avoid thread issues
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=2,  # Reduced batch size
            shuffle=False,
            num_workers=0,  # Reduced from 2 to 0 to avoid thread issues
            pin_memory=True,
            drop_last=True
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

        for epoch in range(5):  # 5 local epochs
            epoch_loss = 0.0
            epoch_samples = 0

            for batch in self.train_loader:
                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_values)
                predictions = outputs['predictions']

                # Apply mask and compute loss
                masked_predictions = predictions[mask]
                masked_targets = targets[mask]

                if masked_predictions.size(0) > 0:
                    loss = F.cross_entropy(
                        masked_predictions.view(-1, 504),
                        masked_targets.view(-1)
                    )

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_samples += input_values.size(0)

            total_loss += epoch_loss
            num_samples += epoch_samples

            logger.info(
                f"Client {self.client_id}: Epoch {epoch + 1}, Loss = {epoch_loss / len(self.train_loader):.4f}")

        # Average over epochs
        avg_loss = total_loss / (5 * len(self.train_loader))

        logger.info(
            f"Client {self.client_id}: training completed, avg_loss={avg_loss:.4f}")

        return self.get_parameters(config={}), num_samples, {"pretrain_loss": avg_loss}

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model using the provided parameters."""
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_values = batch['input_values'].to(self.device)
                targets = batch['targets'].to(self.device)

                outputs = self.model(input_values)

                # Compute loss and accuracy
                loss = F.cross_entropy(
                    outputs['predictions'].view(-1, 504),
                    targets.view(-1)
                )

                preds = torch.argmax(outputs['predictions'], dim=-1)
                accuracy = (preds == targets).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_samples += input_values.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)

        logger.info(
            f"Client {self.client_id}: evaluation completed, loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

        return avg_loss, num_samples, {"accuracy": avg_accuracy}


def client_fn(context: Context) -> FederatedClient:
    """Client function to initialize the federated learning client."""
    config_path = "configs/pretraining_config.yaml"

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
    config_path = "configs/pretraining_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Create initial parameters from a dummy model
    dummy_model = HubertBase()
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in dummy_model.state_dict().items()])

    # Create strategy
    strategy = FedAdamWithCheckpoints(
        checkpoint_manager=checkpoint_manager,
        fraction_fit=config['strategy']['fraction_fit'],
        fraction_evaluate=config['strategy']['fraction_evaluate'],
        min_fit_clients=config['strategy']['min_fit_clients'],
        min_evaluate_clients=config['strategy']['min_evaluate_clients'],
        min_available_clients=config['strategy']['min_available_clients'],
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    # Create server config
    server_config = ServerConfig(num_rounds=20)

    # Create server app components
    server_app_components = ServerAppComponents(
        strategy=strategy,
    )

    return server_app_components


def main():
    """Main function to run federated HuBERT pretraining."""
    parser = argparse.ArgumentParser(
        description="Federated HuBERT Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretraining_config.yaml",
                        help="Path to config file")
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode")
    parser.add_argument("--num-clients", type=int,
                        default=10, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=20,
                        help="Number of federated rounds")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/federated_pretraining.log'),
            logging.StreamHandler()
        ]
    )

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
