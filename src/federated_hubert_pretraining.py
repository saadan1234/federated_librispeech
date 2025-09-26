#!/usr/bin/env python3
"""
Research-Grade Federated HuBERT Pretraining with Flower (FedAdam) - S3PRL Compatible.

This implementation maintains full compatibility with s3prl's HuBERT pretraining logic
while enabling federated learning through Flower framework. All training procedures,
loss computation, and model architectures follow s3prl standards for research comparison.
"""

import os
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
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

# Add s3prl to Python path
sys.path.append('/home/saadan/scratch/federated_librispeech/s3prl')
from s3prl.upstream.hubert.hubert_model import HubertModel, HubertConfig, HubertPretrainingConfig
from s3prl.dataio.dataset.load_audio import LoadAudio
from s3prl.utility.audio import extract_feature
from s3prl.dataio.encoder.tokenizer import Tokenizer
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLibriSpeechDataset(Dataset):
    """
    Dataset class fully compatible with s3prl's HuBERT pretraining dataset structure.
    Uses the partitioned data structure created by s3prl_aligned_partitioner.py
    """

    def __init__(self, manifest_file: str, kmeans_targets_path: str, split: str = "train",
                 max_length: int = 250000, sample_rate: int = 16000, 
                 label_rate: int = 50, vocab_size: int = 504):
        """
        Initialize dataset compatible with s3prl partitioned structure.
        
        Args:
            manifest_file: Path to train.csv or validation.csv
            kmeans_targets_path: Path to kmeans_targets.npy for this client
            split: "train" or "validation"
            max_length: Maximum audio length in samples (s3prl default: 250000)
            sample_rate: Audio sample rate (16000 for LibriSpeech)
            label_rate: Label frame rate (50Hz for HuBERT)
            vocab_size: KMeans vocabulary size (504 for HuBERT base)
        """
        self.manifest = pd.read_csv(manifest_file)
        self.split = split
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.vocab_size = vocab_size
        
        # Load client-specific kmeans targets
        self.kmeans_targets = np.load(kmeans_targets_path, allow_pickle=True)
        logger.info(f"Loaded {len(self.kmeans_targets)} kmeans targets for {split}")
        
        # Initialize s3prl audio loader for consistency
        self.audio_loader = LoadAudio(sample_rate=sample_rate, channels_first=False)
        
        # Verify data alignment
        assert len(self.manifest) == len(self.kmeans_targets), \
            f"Manifest length {len(self.manifest)} != targets length {len(self.kmeans_targets)}"
            
        logger.info(f"Initialized {split} dataset with {len(self.manifest)} samples")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item following s3prl's expected format."""
        row = self.manifest.iloc[idx]
        
        # Load audio using s3prl's standard approach
        audio_path = row['file_path']  # Created by s3prl_aligned_partitioner.py
        
        # Use s3prl's audio loader for consistency
        wav = self.audio_loader(audio_path)
        
        # Ensure correct tensor format (1D tensor for s3prl)
        if wav.dim() > 1:
            wav = wav.squeeze()
            
        # Truncate if too long (following s3prl logic)
        if len(wav) > self.max_length:
            wav = wav[:self.max_length]
            
        # Get corresponding kmeans targets
        targets = self.kmeans_targets[idx]
        
        # Convert to tensor and ensure proper dtype
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).long()
        else:
            targets = torch.tensor(targets, dtype=torch.long)
            
        # Calculate expected target length based on audio length and label rate
        # Following s3prl's frame calculation logic
        expected_target_len = len(wav) // (self.sample_rate // self.label_rate)
        
        # Align target length with audio (pad or truncate as needed)
        if len(targets) > expected_target_len:
            targets = targets[:expected_target_len]
        elif len(targets) < expected_target_len:
            # Pad with last value (following s3prl convention)
            pad_len = expected_target_len - len(targets)
            last_val = targets[-1] if len(targets) > 0 else 0
            padding = torch.full((pad_len,), last_val, dtype=torch.long)
            targets = torch.cat([targets, padding])

        return {
            'wav': wav,              # Raw audio waveform (s3prl format)
            'target': targets,       # Frame-level cluster assignments
            'wav_len': len(wav),     # Audio length for batching
            'target_len': len(targets)  # Target length for batching
        }

def collate_fn(batch):
    """
    Collate function compatible with s3prl's batching logic.
    Handles variable-length sequences with proper padding.
    """
    # Sort batch by wav length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x['wav_len'], reverse=True)
    
    # Extract components
    wavs = [item['wav'] for item in batch]
    targets = [item['target'] for item in batch]
    wav_lens = torch.tensor([item['wav_len'] for item in batch])
    target_lens = torch.tensor([item['target_len'] for item in batch])
    
    # Pad wavs to same length
    max_wav_len = max(wav_lens)
    padded_wavs = []
    for wav in wavs:
        if len(wav) < max_wav_len:
            padding = torch.zeros(max_wav_len - len(wav))
            padded_wav = torch.cat([wav, padding])
        else:
            padded_wav = wav
        padded_wavs.append(padded_wav)
    
    # Pad targets to same length
    max_target_len = max(target_lens)
    padded_targets = []
    for target in targets:
        if len(target) < max_target_len:
            # Pad with -100 (ignore index for loss computation)
            padding = torch.full((max_target_len - len(target),), -100, dtype=torch.long)
            padded_target = torch.cat([target, padding])
        else:
            padded_target = target
        padded_targets.append(padded_target)
    
    return {
        'wavs': torch.stack(padded_wavs),           # [batch, max_wav_len]
        'targets': torch.stack(padded_targets),     # [batch, max_target_len]
        'wav_lens': wav_lens,                       # [batch]
        'target_lens': target_lens                  # [batch]
    }

class S3PRLCompatibleClient(NumPyClient):
    """
    Federated client that maintains full s3prl compatibility for research comparison.
    All training logic follows s3prl's HuBERT pretraining procedures exactly.
    """

    def __init__(self, client_id: int, train_dataset, val_dataset, model, device=None):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_id = client_id % gpu_count
                self.device = torch.device(f"cuda:{gpu_id}")
                # Remove memory fraction setting for simulation
                # torch.cuda.set_per_process_memory_fraction(0.8, gpu_id)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Load s3prl compatible configuration
        config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.config = cfg.get('pretraining', {})
        
        # S3PRL-compatible dataloader settings
        batch_size = self.config.get('batch_size', 16)  # Smaller for memory efficiency
        num_workers = min(self.config.get('num_workers', 4), 8)  # Limit workers
        
        # Create dataloaders with s3prl-compatible settings
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=collate_fn,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def get_parameters(self, config: Config) -> NDArrays:
        """Get model parameters as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """
        Train using s3prl's exact HuBERT pretraining logic.
        This ensures research compatibility with centralized s3prl training.
        """
        self.set_parameters(parameters)
        
        # S3PRL-compatible optimizer (following HuBERT paper exactly)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 0.0005),      # 5e-4 as in HuBERT paper
            weight_decay=0.01,                # Standard weight decay
            betas=(0.9, 0.98),               # HuBERT paper betas
            eps=1e-6                         # Numerical stability
        )
        
        # Learning rate scheduler (following s3prl approach)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1e-6, 
            end_factor=1.0, 
            total_iters=len(self.train_loader) * config.get("local_epochs", 1) // 10
        )

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        num_samples = 0

        local_epochs = int(config.get("local_epochs", 1))

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_predictions = 0

            pbar = tqdm(
                self.train_loader,
                desc=f"Client {self.client_id} Epoch {epoch + 1}/{local_epochs}",
                leave=False
            )

            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                wavs = batch['wavs'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                wav_lens = batch['wav_lens'].to(self.device, non_blocking=True)
                target_lens = batch['target_lens'].to(self.device, non_blocking=True)

                # Create padding mask for s3prl model
                batch_size, max_wav_len = wavs.shape
                padding_mask = torch.zeros(batch_size, max_wav_len, dtype=torch.bool, device=self.device)
                for i, wav_len in enumerate(wav_lens):
                    if wav_len < max_wav_len:
                        padding_mask[i, wav_len:] = True

                # Prepare target_list in s3prl format
                target_list = [targets]

                # Forward pass using s3prl HubertModel API exactly
                try:
                    net_output = self.model(
                        source=wavs,
                        target_list=target_list,
                        padding_mask=padding_mask,
                        mask=True,          # Enable masking for pretraining
                        features_only=False # Get logits for loss computation
                    )

                    # Extract logits following s3prl format
                    logit_m_list = net_output.get("logit_m_list", [])
                    target_m_list = net_output.get("target_m_list", [])
                    
                    if not logit_m_list or not target_m_list:
                        logger.warning(f"Empty logits/targets in batch {batch_idx}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Forward pass failed in batch {batch_idx}: {e}")
                    continue

                # Compute loss exactly as s3prl does
                loss = 0.0
                correct_predictions = 0
                total_pred_count = 0
                
                for logits, targets_masked in zip(logit_m_list, target_m_list):
                    # logits: [batch * masked_frames, vocab_size]
                    # targets_masked: [batch * masked_frames]
                    
                    # Filter out padding tokens (-100)
                    valid_mask = targets_masked != -100
                    if valid_mask.sum() == 0:
                        continue
                        
                    valid_logits = logits[valid_mask]
                    valid_targets = targets_masked[valid_mask]
                    
                    # Cross entropy loss
                    batch_loss = F.cross_entropy(valid_logits, valid_targets)
                    loss += batch_loss
                    
                    # Accuracy computation
                    predictions = torch.argmax(valid_logits, dim=-1)
                    correct_predictions += (predictions == valid_targets).sum().item()
                    total_pred_count += len(valid_targets)

                # Normalize loss by number of target layers
                if len(logit_m_list) > 0:
                    loss = loss / len(logit_m_list)
                else:
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (following s3prl approach)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                # Update metrics
                epoch_loss += loss.item()
                epoch_correct += correct_predictions
                epoch_predictions += total_pred_count

                # Update progress bar
                if batch_idx % 10 == 0:
                    current_acc = correct_predictions / max(1, total_pred_count) * 100
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.2f}%',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_predictions += epoch_predictions
            num_samples += len(self.train_dataset)

        # Calculate final metrics
        avg_loss = total_loss / max(1, local_epochs * len(self.train_loader))
        avg_accuracy = total_correct / max(1, total_predictions) * 100

        return self.get_parameters(config={}), num_samples, {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "learning_rate": scheduler.get_last_lr()[0]
        }

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate using s3prl's exact evaluation logic for research comparison.
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        num_samples = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Client {self.client_id} Validation",
                leave=False
            )

            for batch in pbar:
                # Move data to device
                wavs = batch['wavs'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                wav_lens = batch['wav_lens'].to(self.device, non_blocking=True)

                # Create padding mask
                batch_size, max_wav_len = wavs.shape
                padding_mask = torch.zeros(batch_size, max_wav_len, dtype=torch.bool, device=self.device)
                for i, wav_len in enumerate(wav_lens):
                    if wav_len < max_wav_len:
                        padding_mask[i, wav_len:] = True

                # Forward pass
                target_list = [targets]
                net_output = self.model(
                    source=wavs,
                    target_list=target_list,
                    padding_mask=padding_mask,
                    mask=True,
                    features_only=False
                )

                # Extract and process logits
                logit_m_list = net_output.get("logit_m_list", [])
                target_m_list = net_output.get("target_m_list", [])

                if not logit_m_list or not target_m_list:
                    continue

                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0

                for logits, targets_masked in zip(logit_m_list, target_m_list):
                    valid_mask = targets_masked != -100
                    if valid_mask.sum() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_targets = targets_masked[valid_mask]

                    # Loss
                    loss = F.cross_entropy(valid_logits, valid_targets)
                    batch_loss += loss.item()

                    # Accuracy
                    predictions = torch.argmax(valid_logits, dim=-1)
                    batch_correct += (predictions == valid_targets).sum().item()
                    batch_predictions += len(valid_targets)

                if len(logit_m_list) > 0:
                    batch_loss /= len(logit_m_list)

                total_loss += batch_loss
                total_correct += batch_correct
                total_predictions += batch_predictions
                num_samples += batch_size

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_accuracy = total_correct / max(1, total_predictions) * 100

        return avg_loss, num_samples, {
            "val_accuracy": avg_accuracy,
            "val_predictions": total_predictions
        }

def client_fn(context: Context) -> S3PRLCompatibleClient:
    """
    Create client function that uses the exact partitioned data structure.
    """
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get client ID
    node_id = context.node_id
    num_clients = config.get('simulation', {}).get('num_supernodes', 2)
    client_id = hash(str(node_id)) % num_clients

    # Use the exact partitioned data structure
    base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
    client_path = base_path / f"client_{client_id}"

    if not client_path.exists():
        raise FileNotFoundError(f"Client data not found: {client_path}")

    # Load datasets using partitioned structure
    train_manifest = client_path / "train.csv"
    val_manifest = client_path / "validation.csv"
    kmeans_targets = client_path / "kmeans_targets.npy"

    if not all([train_manifest.exists(), val_manifest.exists(), kmeans_targets.exists()]):
        raise FileNotFoundError(f"Required files missing in {client_path}")

    # Load kmeans metadata for vocab size
    with open(base_path / "kmeans_metadata.json", 'r') as f:
        kmeans_meta = json.load(f)
    vocab_size = kmeans_meta.get('n_clusters', 504)

    pre_cfg = config.get('pretraining', {})

    # Create datasets
    train_dataset = FederatedLibriSpeechDataset(
        manifest_file=str(train_manifest),
        kmeans_targets_path=str(kmeans_targets),
        split="train",
        max_length=pre_cfg.get('max_audio_length', 250000),  # S3PRL default
        sample_rate=pre_cfg.get('sample_rate', 16000),
        label_rate=pre_cfg.get('label_rate', 50),
        vocab_size=vocab_size
    )

    # For validation, we need to slice the kmeans targets appropriately
    train_len = len(train_dataset)
    val_targets_slice = np.load(str(kmeans_targets), allow_pickle=True)[train_len:]
    
    # Save temporary validation targets
    val_targets_path = client_path / "validation_kmeans_targets.npy"
    if not val_targets_path.exists():
        # Load and slice from combined targets
        all_targets = np.load(str(kmeans_targets), allow_pickle=True)
        train_len = len(train_dataset)
        
        # Verify we have enough targets
        if len(all_targets) < train_len:
            raise ValueError(f"Not enough kmeans targets: {len(all_targets)} < {train_len}")
        
        # Slice validation targets
        val_targets_slice = all_targets[train_len:]
        
        # Save for future use
        np.save(val_targets_path, val_targets_slice)
        logger.info(f"Created validation targets: {len(val_targets_slice)} samples")

    val_dataset = FederatedLibriSpeechDataset(
        manifest_file=str(val_manifest),
        kmeans_targets_path=str(val_targets_path),
        split="validation",
        max_length=pre_cfg.get('max_audio_length', 250000),
        sample_rate=pre_cfg.get('sample_rate', 16000),
        label_rate=pre_cfg.get('label_rate', 50),
        vocab_size=vocab_size
    )

    # Create s3prl HubertModel with exact configuration
    model_cfg = HubertConfig(
        label_rate=pre_cfg.get('label_rate', 50),
        extractor_mode=pre_cfg.get('extractor_mode', "default"),
        encoder_layers=pre_cfg.get('num_hidden_layers', 12),
        encoder_embed_dim=pre_cfg.get('hidden_size', 768),
        encoder_ffn_embed_dim=pre_cfg.get('intermediate_size', 3072),
        encoder_attention_heads=pre_cfg.get('num_attention_heads', 12),
        activation_fn=pre_cfg.get('activation_fn', "gelu"),
        dropout=pre_cfg.get('dropout', 0.1),
        attention_dropout=pre_cfg.get('attention_dropout', 0.1),
        activation_dropout=pre_cfg.get('activation_dropout', 0.0),
        final_dim=pre_cfg.get('final_dim', 256),
        layer_norm_first=pre_cfg.get('layer_norm_first', True),
        conv_feature_layers=pre_cfg.get('conv_feature_layers', "[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2"),
        logit_temp=pre_cfg.get('logit_temp', 0.1),
        mask_prob=pre_cfg.get('mask_prob', 0.08),
        mask_selection=pre_cfg.get('mask_selection', "static"),
        mask_other=pre_cfg.get('mask_other', 0),
        mask_length=pre_cfg.get('mask_length', 10),
        no_mask_overlap=pre_cfg.get('no_mask_overlap', False),
        mask_min_space=pre_cfg.get('mask_min_space', 1),
        # ... (rest of the checkpointing and server functions remain similar but with proper s3prl integration)
        conv_bias=pre_cfg.get('conv_bias', False),
        encoder_layerdrop=pre_cfg.get('encoder_layerdrop', 0.0),
        dropout_input=pre_cfg.get('dropout_input', 0.0),
        dropout_features=pre_cfg.get('dropout_features', 0.0),
        feature_grad_mult=pre_cfg.get('feature_grad_mult', 0.1),
        untie_final_proj=pre_cfg.get('untie_final_proj', True),
    )

    task_cfg = HubertPretrainingConfig(
        label_rate=pre_cfg.get('label_rate', 50),
        sample_rate=pre_cfg.get('sample_rate', 16000),
        normalize=pre_cfg.get('normalize', False),
        enable_padding=pre_cfg.get('enable_padding', False),
        max_keep_size=pre_cfg.get('max_keep_size', None),
        max_sample_size=pre_cfg.get('max_audio_length', 250000),
        min_sample_size=pre_cfg.get('min_sample_size', None),
        random_crop=pre_cfg.get('random_crop', True),
        pad_audio=pre_cfg.get('pad_audio', False)
    )

    # Create dictionaries for HuBERT model
    dictionaries = [list(range(vocab_size))]

    model = HubertModel(model_cfg, task_cfg, dictionaries)

    return S3PRLCompatibleClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model
    ).to_client()

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics using weighted average for federated learning."""
    if not metrics:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for _, client_metrics in metrics:
        metric_names.update(client_metrics.keys())
    
    aggregated = {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    for metric_name in metric_names:
        weighted_sum = 0.0
        for num_samples, client_metrics in metrics:
            if metric_name in client_metrics:
                weighted_sum += client_metrics[metric_name] * num_samples
        aggregated[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    return aggregated

def fit_config(server_round: int) -> Dict[str, Union[str, int, float]]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round > 5 else 1,  # Adjust local epochs
        "lr": max(0.0001, 0.0005 * (0.98 ** (server_round - 1))),  # Decay learning rate
        "batch_size": 16,
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, Union[str, int, float]]:
    """Return evaluation configuration dict for each round."""
    return {
        "server_round": server_round,
        "batch_size": 16,
    }

def get_evaluate_fn(model_cfg, task_cfg, dictionaries):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Union[str, int, float]]):
        """Evaluate the global model on a global test set (optional)."""
        # For now, return None to skip server-side evaluation
        # In a full implementation, you could load a global test set here
        return None
    return evaluate

def server_fn(context: Context) -> ServerAppComponents:
    """Construct server components for federated learning."""
    
    # Load configuration
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    simulation_cfg = config.get('simulation', {})
    pretraining_cfg = config.get('pretraining', {})

    # Load model configuration for parameter initialization
    base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
    with open(base_path / "kmeans_metadata.json", 'r') as f:
        kmeans_meta = json.load(f)
    vocab_size = kmeans_meta.get('n_clusters', 504)
    
    # Create s3prl HubertModel for initial parameters
    model_cfg = HubertConfig(
        label_rate=pretraining_cfg.get('label_rate', 50),
        extractor_mode=pretraining_cfg.get('extractor_mode', "default"),
        encoder_layers=pretraining_cfg.get('num_hidden_layers', 12),
        encoder_embed_dim=pretraining_cfg.get('hidden_size', 768),
        encoder_ffn_embed_dim=pretraining_cfg.get('intermediate_size', 3072),
        encoder_attention_heads=pretraining_cfg.get('num_attention_heads', 12),
        activation_fn=pretraining_cfg.get('activation_fn', "gelu"),
        dropout=pretraining_cfg.get('dropout', 0.1),
        attention_dropout=pretraining_cfg.get('attention_dropout', 0.1),
        activation_dropout=pretraining_cfg.get('activation_dropout', 0.0),
        final_dim=pretraining_cfg.get('final_dim', 256),
        layer_norm_first=pretraining_cfg.get('layer_norm_first', True),
        conv_feature_layers=pretraining_cfg.get('conv_feature_layers', "[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2"),
        logit_temp=pretraining_cfg.get('logit_temp', 0.1),
        mask_prob=pretraining_cfg.get('mask_prob', 0.08),
        mask_selection=pretraining_cfg.get('mask_selection', "static"),
        mask_other=pretraining_cfg.get('mask_other', 0),
        mask_length=pretraining_cfg.get('mask_length', 10),
        no_mask_overlap=pretraining_cfg.get('no_mask_overlap', False),
        mask_min_space=pretraining_cfg.get('mask_min_space', 1),
        # Additional HuBERT configuration parameters
        conv_bias=pretraining_cfg.get('conv_bias', False),
        encoder_layerdrop=pretraining_cfg.get('encoder_layerdrop', 0.0),
        dropout_input=pretraining_cfg.get('dropout_input', 0.0),
        dropout_features=pretraining_cfg.get('dropout_features', 0.0),
        feature_grad_mult=pretraining_cfg.get('feature_grad_mult', 0.1),
        untie_final_proj=pretraining_cfg.get('untie_final_proj', True),
    )

    task_cfg = HubertPretrainingConfig(
        label_rate=pretraining_cfg.get('label_rate', 50),
        sample_rate=pretraining_cfg.get('sample_rate', 16000),
        normalize=pre_cfg.get('normalize', False),
        enable_padding=pre_cfg.get('enable_padding', False),
        max_keep_size=pre_cfg.get('max_keep_size', None),
        max_sample_size=pre_cfg.get('max_audio_length', 250000),
        min_sample_size=pre_cfg.get('min_sample_size', None),
        random_crop=pre_cfg.get('random_crop', True),
        pad_audio=pre_cfg.get('pad_audio', False)
    )
    
    # Create dictionaries for HuBERT model
    dictionaries = [list(range(vocab_size))]
    
    # Initialize model for getting initial parameters
    model = HubertModel(model_cfg, task_cfg, dictionaries)
    initial_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    # Create FedAdam strategy with s3prl-compatible settings
    strategy = FedAdam(
        fraction_fit=simulation_cfg.get('fraction_fit', 1.0),
        fraction_evaluate=simulation_cfg.get('fraction_evaluate', 1.0),
        min_fit_clients=simulation_cfg.get('min_fit_clients', 2),
        min_evaluate_clients=simulation_cfg.get('min_evaluate_clients', 2),
        min_available_clients=simulation_cfg.get('min_available_clients', 2),
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_fn=get_evaluate_fn(model_cfg, task_cfg, dictionaries),
        # FedAdam specific parameters
        eta=simulation_cfg.get('eta', 1e-3),           # Server learning rate
        eta_l=simulation_cfg.get('eta_l', 1e-3),       # Client learning rate
        beta_1=simulation_cfg.get('beta_1', 0.9),      # First moment decay
        beta_2=simulation_cfg.get('beta_2', 0.99),     # Second moment decay
        tau=simulation_cfg.get('tau', 1e-9),           # Controls the algorithm's degree of adaptability
    )
    
    return ServerAppComponents(strategy=strategy)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, client_id=None):
    """Save training checkpoint with s3prl compatibility."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
        'timestamp': time.time()
    }
    
    if client_id is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_client_{client_id}_epoch_{epoch}.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint with s3prl compatibility."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def validate_s3prl_compatibility():
    """Validate that s3prl imports and configurations are working correctly."""
    try:
        # Test s3prl imports
        from s3prl.upstream.hubert.hubert_model import HubertModel, HubertConfig, HubertPretrainingConfig
        from s3prl.dataio.dataset.load_audio import LoadAudio
        
        # Test basic model creation
        model_cfg = HubertConfig()
        task_cfg = HubertPretrainingConfig()
        dictionaries = [list(range(504))]
        
        model = HubertModel(model_cfg, task_cfg, dictionaries)
        logger.info("✅ S3PRL compatibility validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3PRL compatibility validation failed: {e}")
        return False

def setup_logging(log_dir: str, client_id: Optional[int] = None):
    """Setup logging with proper file handlers for federated training."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    if client_id is not None:
        log_file = os.path.join(log_dir, f"client_{client_id}.log")
    else:
        log_file = os.path.join(log_dir, "federated_training.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def main():
    """Main function with proper error handling and validation."""
    parser = argparse.ArgumentParser(description="S3PRL-Compatible Federated HuBERT Pretraining")
    parser.add_argument("--config", type=str,
                        default="/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml",
                        help="Path to config file")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--log-dir", type=str, 
                        default="/home/saadan/scratch/federated_librispeech/src/logs",
                        help="Directory for log files")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)

    try:
        # Validate s3prl compatibility first
        if not validate_s3prl_compatibility():
            raise RuntimeError("S3PRL compatibility check failed")

        if args.simulation:
            print("🚀 Starting S3PRL-Compatible Federated HuBERT Pretraining")
            print(f"📊 Configuration: {args.num_clients} clients, {args.num_rounds} rounds")
            
            # Load configuration
            if not os.path.exists(args.config):
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
            
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update config with command line arguments
            simulation_config = config.get('simulation', {})
            simulation_config['num_supernodes'] = args.num_clients
            simulation_config['num_rounds'] = args.num_rounds
            config['simulation'] = simulation_config
            
            # Validate partitioned data exists
            base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
            
            if not base_path.exists():
                raise FileNotFoundError(f"Data directory not found: {base_path}")
            
            # Check required metadata files
            metadata_files = ["kmeans_metadata.json", "partition_metadata.json"]
            for metadata_file in metadata_files:
                metadata_path = base_path / metadata_file
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Required metadata file missing: {metadata_path}")
            
            # Validate client data
            clients_found = []
            for i in range(args.num_clients):
                client_path = base_path / f"client_{i}"
                required_files = ["train.csv", "validation.csv", "kmeans_targets.npy"]
                
                if client_path.exists():
                    missing_files = []
                    for req_file in required_files:
                        file_path = client_path / req_file
                        if not file_path.exists():
                            missing_files.append(str(file_path))
                    
                    if missing_files:
                        logger.warning(f"Client {i} missing files: {missing_files}")
                    else:
                        clients_found.append(i)
                        logger.info(f"✅ Client {i} data validated")
                else:
                    logger.warning(f"Client {i} directory not found: {client_path}")
            
            if len(clients_found) < args.num_clients:
                logger.warning(f"Only found {len(clients_found)} clients, requested {args.num_clients}")
                if len(clients_found) < 2:
                    raise ValueError("Need at least 2 clients for federated learning")
                # Adjust num_clients to available clients
                args.num_clients = len(clients_found)
                simulation_config['num_supernodes'] = args.num_clients
            
            print(f"✅ All {args.num_clients} client(s) data validated")
            
            # Create client and server apps
            client_app = ClientApp(client_fn=client_fn)
            server_app = ServerApp(server_fn=server_fn)
            
            # Run federated simulation
            print("🔄 Starting federated simulation...")
            start_time = time.time()
            
            try:
                history = run_simulation(
                    server_app=server_app,
                    client_app=client_app,
                    num_supernodes=args.num_clients,
                    backend_config={
                        "client_resources": {
                            "num_cpus": simulation_config.get('num_cpus', 4),
                            "num_gpus": simulation_config.get('num_gpus', 0.25)  # Share GPUs
                        }
                    },
                    run_config=ServerConfig(
                        num_rounds=args.num_rounds,
                        round_timeout=simulation_config.get('round_timeout', 1800),  # 30 minutes
                    ),
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"🎉 Federated training completed!")
                print(f"⏱️  Total training time: {total_time:.2f} seconds")
                print(f"📈 Training rounds completed: {len(history.losses_distributed)}")
                
                # Save training history
                history_path = os.path.join(args.log_dir, "training_history.json")
                history_dict = {
                    "losses_distributed": [(round_num, loss) for round_num, loss in history.losses_distributed],
                    "losses_centralized": [(round_num, loss) for round_num, loss in history.losses_centralized],
                    "metrics_distributed_fit": history.metrics_distributed_fit,
                    "metrics_distributed": history.metrics_distributed,
                    "metrics_centralized": history.metrics_centralized,
                    "total_time": total_time,
                    "num_rounds": args.num_rounds,
                    "num_clients": args.num_clients
                }
                
                with open(history_path, 'w') as f:
                    json.dump(history_dict, f, indent=2)
                
                logger.info(f"Training history saved to: {history_path}")
                
                # Print final metrics
                if history.losses_distributed:
                    final_loss = history.losses_distributed[-1][1]
                    print(f"🏁 Final loss: {final_loss:.4f}")
                
                if history.metrics_distributed_fit:
                    final_metrics = history.metrics_distributed_fit[-1][1]
                    for metric_name, metric_value in final_metrics.items():
                        print(f"📊 Final {metric_name}: {metric_value:.4f}")
                
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                raise
                
        else:
            # Non-simulation mode (for distributed deployment)
            print("⚠️  Non-simulation mode not implemented yet")
            print("Use --simulation flag for now")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)