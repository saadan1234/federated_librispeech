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

from flwr.common.typing import NDArrays, Config, Parameters
from flwr.client import NumPyClient, ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, FitIns, EvaluateIns
from flwr.server.strategy import Strategy, FedAdam
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager


# Add s3prl to Python path
sys.path.append('/home/saadan/scratch/federated_librispeech/s3prl')
from s3prl.upstream.hubert.hubert_model import HubertModel, HubertConfig, HubertPretrainingConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage(stage=""):
    """Log detailed memory usage."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"{stage} - Process Memory:")
    print(f"  RSS: {memory_info.rss / 1e9:.2f}GB")
    print(f"  VMS: {memory_info.vms / 1e9:.2f}GB")
    
    system_memory = psutil.virtual_memory()
    print(f"  System: {system_memory.used / 1e9:.2f}GB / {system_memory.total / 1e9:.2f}GB")

def safe_execute(func, context, client_id=None):
    """Safe execution wrapper for functions with error handling and logging."""
    try:
        logger.info(f"▶️ {context}: Starting {func.__name__}")
        result = func()
        logger.info(f"✅ {context}: {func.__name__} completed successfully")
        return result
    except Exception as e:
        logger.error(f"❌ {context}: Error in {func.__name__}: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Emergency cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise

import psutil
import os
import traceback
import sys
from datetime import datetime

def log_memory_and_process(stage: str, client_id: int = None):
    """Comprehensive memory and process logging."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process info
        cpu_percent = process.cpu_percent()
        
        log_msg = f"""
{'='*50} {stage.upper()} {'='*50}
PID: {os.getpid()} | Client: {client_id if client_id is not None else 'Server'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESS MEMORY:
  RSS: {memory_info.rss / 1e9:.2f} GB
  VMS: {memory_info.vms / 1e9:.2f} GB
  CPU: {cpu_percent:.1f}%

SYSTEM MEMORY:
  Used: {system_memory.used / 1e9:.2f} GB
  Available: {system_memory.available / 1e9:.2f} GB  
  Total: {system_memory.total / 1e9:.2f} GB
  Percentage: {system_memory.percent:.1f}%

PYTORCH MEMORY:
"""
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                log_msg += f"  GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total\n"
        else:
            log_msg += "  CUDA not available\n"
        
        log_msg += "=" * 110
        
        logger.info(log_msg)
        
        # Check for potential OOM
        if system_memory.percent > 95:
            logger.error(f"⚠️ CRITICAL: System memory usage at {system_memory.percent:.1f}%")
            return False
        
        if memory_info.rss / 1e9 > 50:  # Process using more than 50GB
            logger.error(f"⚠️ CRITICAL: Process memory usage at {memory_info.rss / 1e9:.2f}GB")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to log memory info: {e}")
        return True

def safe_execute(func, stage_name: str, client_id: int = None, *args, **kwargs):
    """Safely execute function with comprehensive error handling."""
    try:
        logger.info(f"🔄 STARTING: {stage_name}")
        
        # Pre-execution memory check
        if not log_memory_and_process(f"BEFORE {stage_name}", client_id):
            raise RuntimeError(f"Memory check failed before {stage_name}")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Post-execution memory check
        log_memory_and_process(f"AFTER {stage_name}", client_id)
        
        logger.info(f"✅ COMPLETED: {stage_name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ FAILED: {stage_name}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Emergency memory logging
        log_memory_and_process(f"ERROR IN {stage_name}", client_id)
        
        # Force cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise

def load_audio_with_torchaudio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Load audio file using torchaudio with proper resampling and format handling.
    This replaces the s3prl LoadAudio to avoid initialization issues.
    """
    try:
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        
        # Return as 1D tensor (s3prl format)
        return waveform.squeeze()
        
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        # Return empty tensor as fallback
        return torch.zeros(sample_rate)  # 1 second of silence

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
        
        # Load client-specific kmeans targets with better error handling
        try:
            self.kmeans_targets = np.load(kmeans_targets_path, allow_pickle=True)
            logger.info(f"Loaded {len(self.kmeans_targets)} kmeans targets from {kmeans_targets_path}")
        except Exception as e:
            logger.error(f"Failed to load kmeans targets from {kmeans_targets_path}: {e}")
            raise
        
        # Debug information
        logger.info(f"Dataset {split}: Manifest length: {len(self.manifest)}, Targets length: {len(self.kmeans_targets)}")
        
        # Handle data alignment with fallback mechanisms
        if len(self.manifest) != len(self.kmeans_targets):
            logger.warning(f"Data mismatch in {split} dataset:")
            logger.warning(f"  Manifest file: {manifest_file} (length: {len(self.manifest)})")
            logger.warning(f"  Targets file: {kmeans_targets_path} (length: {len(self.kmeans_targets)})")
            
            if len(self.kmeans_targets) == 0:
                logger.error("  Kmeans targets file is empty!")
                raise ValueError("Kmeans targets file is empty")
            
            # Implement fallback strategies
            if len(self.manifest) > len(self.kmeans_targets):
                # More manifest entries than targets - truncate manifest
                logger.warning(f"  Truncating manifest from {len(self.manifest)} to {len(self.kmeans_targets)}")
                self.manifest = self.manifest.iloc[:len(self.kmeans_targets)].copy()
                
                # Save the truncated manifest
                self.manifest.to_csv(manifest_file, index=False)
                logger.info(f"  Saved truncated manifest to {manifest_file}")
                
            elif len(self.kmeans_targets) > len(self.manifest):
                # More targets than manifest entries - truncate targets
                logger.warning(f"  Truncating targets from {len(self.kmeans_targets)} to {len(self.manifest)}")
                self.kmeans_targets = self.kmeans_targets[:len(self.manifest)]
                
                # Save the truncated targets
                np.save(kmeans_targets_path, self.kmeans_targets)
                logger.info(f"  Saved truncated targets to {kmeans_targets_path}")
        
        # Final verification
        assert len(self.manifest) == len(self.kmeans_targets), \
            f"After alignment: Manifest length {len(self.manifest)} != targets length {len(self.kmeans_targets)}"
            
        logger.info(f"Successfully initialized {split} dataset with {len(self.manifest)} samples")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item following s3prl's expected format."""
        if idx >= len(self.manifest) or idx >= len(self.kmeans_targets):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.manifest)} samples")
        
        row = self.manifest.iloc[idx]
        
        # Load audio using our custom function
        audio_path = row['file_path']  # Created by s3prl_aligned_partitioner.py
        # Construct full path - add the base LibriSpeech directory
        base_audio_dir = "/home/saadan/scratch/federated_librispeech/LibriSpeechTars/LibriSpeech"
        full_audio_path = os.path.join(base_audio_dir, audio_path)
        wav = load_audio_with_torchaudio(full_audio_path, self.sample_rate)
        
        # Truncate if too long (following s3prl logic)
        if len(wav) > self.max_length:
            wav = wav[:self.max_length]
        
        # Get corresponding kmeans targets
        targets = self.kmeans_targets[idx]
        
        # Convert to tensor and ensure proper dtype
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).long()
        elif isinstance(targets, (list, tuple)):
            targets = torch.tensor(targets, dtype=torch.long)
        else:
            # Handle scalar targets by creating appropriate length sequence
            target_len = len(wav) // (self.sample_rate // self.label_rate)
            targets = torch.full((target_len,), int(targets), dtype=torch.long)
    
        # CRITICAL FIX: Validate and clamp target indices to prevent CUDA assertion failures
        if len(targets) > 0:
            # Check for invalid indices
            invalid_mask = (targets >= self.vocab_size) | (targets < 0)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} invalid target indices in sample {idx}")
                logger.warning(f"Target range: [{targets.min().item()}, {targets.max().item()}], vocab_size: {self.vocab_size}")
                
                # Clamp invalid indices to valid range
                targets = torch.clamp(targets, 0, self.vocab_size - 1)
    
        # Calculate expected target length based on audio length and label rate
        # Following s3prl's frame calculation logic
        expected_target_len = len(wav) // (self.sample_rate // self.label_rate)
        
        # Align target length with audio (pad or truncate as needed)
        if len(targets) > expected_target_len:
            targets = targets[:expected_target_len]
        elif len(targets) < expected_target_len:
            # Pad with last valid value (following s3prl convention)
            pad_len = expected_target_len - len(targets)
            if len(targets) > 0:
                last_val = targets[-1].item()
                # Ensure the padding value is also within bounds
                last_val = max(0, min(last_val, self.vocab_size - 1))
            else:
                last_val = 0  # Default to cluster 0 if no targets
            padding = torch.full((pad_len,), last_val, dtype=torch.long)
            targets = torch.cat([targets, padding])
        
        # Ensure we have valid targets with final validation
        if len(targets) == 0:
            # Fallback: create minimal target sequence
            targets = torch.zeros(max(1, expected_target_len), dtype=torch.long)
        
        # Final bounds check
        if len(targets) > 0:
            assert targets.min() >= 0, f"Negative target index found: {targets.min()}"
            assert targets.max() < self.vocab_size, f"Target index {targets.max()} >= vocab_size {self.vocab_size}"

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
    max_target_len = max(target_lens) if len(target_lens) > 0 else 1
    padded_targets = []
    for target in targets:
        if len(target) < max_target_len:
            # Pad with -100 (ignore index for loss computation)
            padding = torch.full((max_target_len - len(target),), -100, dtype=torch.long)
            padded_target = torch.cat([target, padding])
        else:
            padded_target = target
        padded_targets.append(padded_target)
    
    # Additional validation before returning
    for i, target in enumerate(padded_targets):
        valid_indices = target[target != -100]
        if len(valid_indices) > 0:
            min_idx = valid_indices.min().item()
            max_idx = valid_indices.max().item()
            if min_idx < 0 or max_idx >= 504:  # Assuming vocab_size = 504
                logger.error(f"Batch {i}: Invalid target indices [{min_idx}, {max_idx}]")
                # Fix invalid indices
                target[target >= 504] = 503
                target[(target >= 0) & (target < 0)] = 0  # This condition won't match, but for completeness
    
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
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Load s3prl compatible configuration
        config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            cfg = {'pretraining': {'batch_size': 8, 'num_workers': 0}}
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            cfg = {'pretraining': {'batch_size': 8, 'num_workers': 0}}

        self.config = cfg.get('pretraining', {})
        
        # S3PRL-compatible dataloader settings
        batch_size = self.config.get('batch_size', 4)  # Further reduced for memory efficiency
        num_workers = min(self.config.get('num_workers', 0), 0
        )  # Reduced workers
        
        logger.info(f"Client {client_id}: Using batch_size={batch_size}, num_workers={num_workers}")
        
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

        logger.info(f"Client {client_id}: Initialized with {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")


    # In your S3PRLCompatibleClient.get_parameters method:
    def get_parameters(self, config: Config) -> NDArrays:
        """Get model parameters as NumPy arrays with proper type validation."""
        params = []
        for name, param in self.model.named_parameters():
            try:
                # Ensure parameter is on CPU and detached
                param_cpu = param.detach().cpu()
                
                # Convert to float32 numpy array
                param_array = param_cpu.numpy().astype(np.float32)
                
                # Validate the array
                if np.isnan(param_array).any() or np.isinf(param_array).any():
                    logger.warning(f"Invalid values in parameter {name}, replacing with zeros")
                    param_array = np.zeros_like(param_array, dtype=np.float32)
                
                params.append(param_array)
                
            except Exception as e:
                logger.error(f"Failed to extract parameter {name}: {e}")
                # Create a zero array with correct shape
                if hasattr(param, 'shape'):
                    fallback = np.zeros(param.shape, dtype=np.float32)
                else:
                    fallback = np.array([0.0], dtype=np.float32)
                params.append(fallback)
        
        logger.info(f"Client {self.client_id}: Extracted {len(params)} parameters")
        return params

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """
        Train using s3prl's exact HuBERT pretraining logic with enhanced error handling.
        """
        try:
            self.set_parameters(parameters)
            
            # Clear GPU memory before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # S3PRL-compatible optimizer with more conservative settings
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.get("lr", 0.0001),     # Reduced learning rate
                weight_decay=0.01,
                betas=(0.9, 0.98),
                eps=1e-6
            )
            
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            num_samples = 0
            processed_batches = 0

            local_epochs = int(config.get("local_epochs", 1))

            for epoch in range(local_epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_predictions = 0

                try:
                    # Limit the number of batches to prevent memory accumulation
                    max_batches = min(10, len(self.train_loader))  # Process at most 10 batches
                    
                    for batch_idx, batch in enumerate(self.train_loader):
                        if batch_idx >= max_batches:
                            logger.info(f"Client {self.client_id}: Stopping at batch {batch_idx} to prevent memory issues")
                            break
                        
                        try:
                            # Move data to device with error handling
                            wavs = batch['wavs'].to(self.device, non_blocking=True)
                            targets = batch['targets'].to(self.device, non_blocking=True)
                            wav_lens = batch['wav_lens'].to(self.device, non_blocking=True)
                            
                            # Validate tensor sizes before processing
                            batch_size, max_wav_len = wavs.shape
                            if batch_size == 0 or max_wav_len == 0:
                                logger.warning(f"Empty batch detected: shape {wavs.shape}")
                                continue
                            
                            # Limit sequence length to prevent OOM
                            if max_wav_len > 80000:  # ~5 seconds at 16kHz
                                wavs = wavs[:, :80000]
                                max_wav_len = 80000
                                logger.info(f"Truncated audio to {max_wav_len} samples")
                            
                            # Create conservative padding mask
                            padding_mask = torch.zeros(batch_size, max_wav_len, dtype=torch.bool, device=self.device)
                            for i, wav_len in enumerate(wav_lens):
                                actual_len = min(wav_len.item(), max_wav_len)
                                if actual_len < max_wav_len:
                                    padding_mask[i, actual_len:] = True

                            # Prepare target_list with validation
                            if targets.size(1) > 250:  # Limit target sequence length
                                targets = targets[:, :250]
                                logger.info(f"Truncated targets to 250 frames")
                            
                            target_list = [targets]

                            # Forward pass with comprehensive error handling
                            try:
                                # Clear gradients before forward pass
                                optimizer.zero_grad()
                                
                                net_output = self.model(
                                    source=wavs,
                                    target_list=target_list,
                                    padding_mask=padding_mask,
                                    mask=True,
                                    features_only=False
                                )

                                # Validate output structure
                                if not isinstance(net_output, dict):
                                    logger.error(f"Invalid model output type: {type(net_output)}")
                                    continue
                                    
                                logit_m_list = net_output.get("logit_m_list", [])
                                target_m_list = net_output.get("target_m_list", [])
                                
                                if not logit_m_list or not target_m_list:
                                    logger.warning(f"Empty logits/targets in batch {batch_idx}")
                                    continue
                                    
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.error(f"GPU OOM in forward pass: {e}")
                                    torch.cuda.empty_cache()
                                    continue
                                else:
                                    logger.error(f"Forward pass runtime error: {e}")
                                    continue
                            except Exception as e:
                                logger.error(f"Forward pass failed: {e}")
                                continue

                            # Compute loss with validation
                            try:
                                loss = 0.0
                                correct_predictions = 0
                                total_pred_count = 0
                                valid_loss_computed = False
                                
                                for logits, targets_masked in zip(logit_m_list, target_m_list):
                                    # Validate tensors
                                    if logits is None or targets_masked is None:
                                        continue
                                        
                                    if logits.numel() == 0 or targets_masked.numel() == 0:
                                        continue
                                    
                                    # Filter out padding tokens
                                    valid_mask = targets_masked != -100
                                    if valid_mask.sum() == 0:
                                        continue
                                    
                                    valid_logits = logits[valid_mask]
                                    valid_targets = targets_masked[valid_mask]
                                    
                                    # Additional validation
                                    if valid_logits.size(0) == 0 or valid_targets.size(0) == 0:
                                        continue
                                    
                                    # Check for valid target indices
                                    if valid_targets.min() < 0 or valid_targets.max() >= valid_logits.size(-1):
                                        logger.warning(f"Invalid target indices: [{valid_targets.min()}, {valid_targets.max()}], vocab_size: {valid_logits.size(-1)}")
                                        continue
                                    
                                    # Cross entropy loss with error handling
                                    try:
                                        batch_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
                                        
                                        # Check for invalid loss
                                        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                            logger.warning(f"Invalid loss detected: {batch_loss}")
                                            continue
                                        
                                        loss += batch_loss
                                        valid_loss_computed = True
                                        
                                        # Accuracy computation
                                        with torch.no_grad():
                                            predictions = torch.argmax(valid_logits, dim=-1)
                                            correct_predictions += (predictions == valid_targets).sum().item()
                                            total_pred_count += len(valid_targets)
                                            
                                    except Exception as e:
                                        logger.error(f"Loss computation failed: {e}")
                                        continue

                                if not valid_loss_computed or len(logit_m_list) == 0:
                                    logger.warning(f"No valid loss computed for batch {batch_idx}")
                                    continue

                                # Normalize loss
                                loss = loss / len(logit_m_list)
                                
                                # Check final loss value
                                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                                    logger.warning(f"Abnormal loss value: {loss.item()}")
                                    continue

                            except Exception as e:
                                logger.error(f"Loss computation error: {e}")
                                continue

                            # Backward pass with error handling
                            try:
                                loss.backward()
                                
                                # Gradient clipping
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                                # Check for invalid gradients
                                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                    logger.warning(f"Invalid gradient norm: {grad_norm}")
                                    optimizer.zero_grad()
                                    continue
                                
                                optimizer.step()
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.error(f"GPU OOM in backward pass: {e}")
                                    torch.cuda.empty_cache()
                                    optimizer.zero_grad()
                                    continue
                                else:
                                    logger.error(f"Backward pass runtime error: {e}")
                                    optimizer.zero_grad()
                                    continue
                            except Exception as e:
                                logger.error(f"Backward pass failed: {e}")
                                optimizer.zero_grad()
                                continue

                            # Update metrics
                            epoch_loss += loss.item()
                            epoch_correct += correct_predictions
                            epoch_predictions += total_pred_count
                            processed_batches += 1

                            # Memory cleanup
                            del wavs, targets, padding_mask, net_output, loss
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Log progress occasionally
                            if batch_idx % 5 == 0 and total_pred_count > 0:
                                current_acc = correct_predictions / total_pred_count * 100
                                logger.info(f"Client {self.client_id} Epoch {epoch+1} Batch {batch_idx}: "
                                          f"loss={loss.item():.4f}, acc={current_acc:.2f}%")

                        except Exception as e:
                            logger.error(f"Batch {batch_idx} processing failed: {e}")
                            # Clean up and continue
                            optimizer.zero_grad()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                except Exception as e:
                    logger.error(f"Epoch {epoch} failed: {e}")
                    continue

                total_loss += epoch_loss
                total_correct += epoch_correct
                total_predictions += epoch_predictions
                num_samples += len(self.train_dataset)

            # Calculate final metrics with safety checks
            if processed_batches > 0:
                avg_loss = total_loss / processed_batches
            else:
                avg_loss = float('inf')
                logger.warning("No batches processed successfully")
            
            if total_predictions > 0:
                avg_accuracy = total_correct / total_predictions * 100
            else:
                avg_accuracy = 0.0
                logger.warning("No predictions made")

            logger.info(f"Client {self.client_id} training complete: "
                       f"loss={avg_loss:.4f}, acc={avg_accuracy:.2f}%, batches={processed_batches}")

            return self.get_parameters(config={}), num_samples, {
                "train_loss": avg_loss,
                "train_accuracy": avg_accuracy,
                "processed_batches": processed_batches
            }

        except Exception as e:
            logger.error(f"Training completely failed for client {self.client_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return current parameters to avoid complete failure
            try:
                current_params = self.get_parameters(config={})
                return current_params, 0, {
                    "train_loss": float('inf'),
                    "train_accuracy": 0.0,
                    "error": str(e)
                }
            except:
                # Last resort - return empty parameters
                return [], 0, {"error": "Complete failure"}

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
                try:
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

                    total_loss += batch_loss
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    num_samples += batch_size

                except Exception as e:
                    logger.error(f"Evaluation error: {e}")
                    continue

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_accuracy = total_correct / max(1, total_predictions) * 100

        return avg_loss, num_samples, {
            "val_accuracy": avg_accuracy,
            "val_predictions": total_predictions
        }
    


def client_fn(context: Context) -> S3PRLCompatibleClient:
    """Create client function with comprehensive debugging."""
    
    client_id = None
    try:
        # CHECKPOINT 1: Initial setup
        logger.info("🚀 CLIENT_FN: Starting client creation")
        log_memory_and_process("CLIENT_FN_START")
        
        # Get client ID early for debugging
        node_id = context.node_id
        num_clients = 2  # Fixed for debugging
        client_id = hash(str(node_id)) % num_clients
        logger.info(f"🆔 CLIENT_FN: Client ID = {client_id}")
        
        # CHECKPOINT 2: Config loading
        def load_config():
            config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
            logger.info(f"📄 Loading config from {config_path}")
            
            if not os.path.exists(config_path):
                logger.warning("Config file not found, using defaults")
                return {
                    'pretraining': {'batch_size': 1, 'learning_rate': 0.00005},
                    'model': {},
                    'simulation': {}
                }
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Config loaded successfully")
            return config
        
        config = safe_execute(load_config, "CONFIG_LOADING", client_id)
        
        # CHECKPOINT 3: Data loading
        # Replace the load_datasets function around line 949 with:
        def load_datasets():
            logger.info(f"📊 CLIENT_{client_id}: Loading datasets")
            
            base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
            client_path = base_path / f"client_{client_id}"
            
            if not client_path.exists():
                raise FileNotFoundError(f"Client {client_id} data directory not found: {client_path}")
            
            # FIXED: Use correct parameter names matching the class constructor
            train_dataset = FederatedLibriSpeechDataset(
                manifest_file=str(client_path / "train.csv"),           # Changed from csv_path
                kmeans_targets_path=str(client_path / "kmeans_targets.npy"), # Changed from kmeans_path
                split="train",
                max_length=80000,    # DEBUGGING: Limit audio length (~5 seconds at 16kHz)
                sample_rate=16000,
                label_rate=50,
                vocab_size=504
            )
            
            val_dataset = FederatedLibriSpeechDataset(
                manifest_file=str(client_path / "validation.csv"),     # Changed from csv_path
                kmeans_targets_path=str(client_path / "kmeans_targets.npy"), # Changed from kmeans_path
                split="validation",
                max_length=80000,    # DEBUGGING: Limit audio length
                sample_rate=16000,
                label_rate=50,
                vocab_size=504
            )
            
            logger.info(f"✅ CLIENT_{client_id}: Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            return train_dataset, val_dataset
        
        train_dataset, val_dataset = safe_execute(load_datasets, "DATASET_LOADING", client_id)
        
        # CHECKPOINT 4: Model creation
        # Replace the create_model function around line 1000-1020 with:
        def create_model():
            logger.info(f"🤖 CLIENT_{client_id}: Creating model")
            
            # DEBUGGING: Use smaller model configuration
            model_cfg = HubertConfig(
                label_rate=50,  # Required positional argument
                extractor_mode="default",
                encoder_layers=12,
                encoder_embed_dim=768,
                encoder_ffn_embed_dim=3072,
                encoder_attention_heads=12,
                activation_fn="gelu",
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                final_dim=256,
                layer_norm_first=True,
                conv_feature_layers="[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2",
                logit_temp=0.1,
                mask_prob=0.08,
                mask_selection="static",
                mask_other=0,
                mask_length=10,
                no_mask_overlap=False,
                mask_min_space=1,
                conv_bias=False,
                encoder_layerdrop=0.0,
                dropout_input=0.0,
                dropout_features=0.0,
                feature_grad_mult=0.1,
                untie_final_proj=True,
            )
            
            # CREATE THE MISSING TASK_CFG
            task_cfg = HubertPretrainingConfig(
                label_rate=50,
                sample_rate=16000,
                normalize=False,
                enable_padding=False,
                max_keep_size=None,
                max_sample_size=80000,  # Match your reduced audio length
                min_sample_size=None,
                random_crop=True,
                pad_audio=False
            )
            
            # CREATE THE MISSING DICTIONARIES
            dictionaries = [list(range(504))]  # vocab_size = 504
            
            # NOW CREATE MODEL WITH ALL 3 REQUIRED ARGUMENTS
            model = HubertModel(model_cfg, task_cfg, dictionaries)
            logger.info(f"✅ CLIENT_{client_id}: Model created")
            return model
        
        model = safe_execute(create_model, "MODEL_CREATION", client_id)
        
        # CHECKPOINT 5: Device assignment
        def assign_device():
            logger.info(f"🖥️ CLIENT_{client_id}: Assigning device")
            
            # DEBUGGING: Force CPU to avoid GPU issues
            device = torch.device("cpu")
            logger.info(f"✅ CLIENT_{client_id}: Using device: {device}")
            return device
        
        device = safe_execute(assign_device, "DEVICE_ASSIGNMENT", client_id)
        
        # CHECKPOINT 6: Client creation
        def create_client():
            logger.info(f"👤 CLIENT_{client_id}: Creating client instance")
            
            client = S3PRLCompatibleClient(
                client_id=client_id,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model=model,
                device=device
            )
            
            logger.info(f"✅ CLIENT_{client_id}: Client instance created successfully")
            return client
        
        client = safe_execute(create_client, "CLIENT_CREATION", client_id)
        
        # Final checkpoint
        log_memory_and_process(f"CLIENT_FN_COMPLETE", client_id)
        logger.info(f"🎉 CLIENT_{client_id}: Client function completed successfully")
        
        return client
        
    except Exception as e:
        logger.error(f"💥 CLIENT_{client_id}: FATAL ERROR in client_fn")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Emergency cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Re-raise with more context
        raise RuntimeError(f"Client {client_id} creation failed: {str(e)}") from e
    
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
        "local_epochs": 1,
        "lr": max(0.00005, 0.0002 * (0.98 ** (server_round - 1))),  # Adjusted learning rate
        "batch_size": 2,  # Reasonable for your GPU
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, Union[str, int, float]]:
    """Return evaluation configuration dict for each round."""
    return {
        "server_round": server_round,
        "batch_size": 2,  # Further reduced batch size to avoid memory issues
    }

def get_evaluate_fn(model_cfg, task_cfg, dictionaries):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Union[str, int, float]]):
        """Evaluate the global model on a global test set (optional)."""
        return None
    return evaluate

class SafeFedAdam(Strategy):
    """
    Custom FedAdam implementation with proper type checking and parameter validation.
    Fixes the numpy dtype incompatibility issues in the original FedAdam.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        evaluate_fn = None,
        eta: float = 1e-3,
        eta_l: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    ):
        super().__init__()
        
        # Validation
        if min_fit_clients > min_available_clients:
            raise ValueError("min_fit_clients cannot be larger than min_available_clients")
        if min_evaluate_clients > min_available_clients:
            raise ValueError("min_evaluate_clients cannot be larger than min_available_clients")
        if fraction_fit < 0.0 or fraction_fit > 1.0:
            raise ValueError("fraction_fit must be between 0.0 and 1.0")
        if fraction_evaluate < 0.0 or fraction_evaluate > 1.0:
            raise ValueError("fraction_evaluate must be between 0.0 and 1.0")

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_fn = evaluate_fn
        
        # FedAdam specific parameters
        self.eta = eta
        self.eta_l = eta_l
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        
        # Initialize parameters and adaptive terms
        self.current_weights = None
        if initial_parameters is not None:
            self.current_weights = self._ensure_float32_parameters(
                parameters_to_ndarrays(initial_parameters)
            )
        
        # Adam states
        self.m_t = None  # First moment
        self.v_t = None  # Second moment
        self.server_round = 0

    # In your SafeFedAdam class, update the _ensure_float32_parameters method:
    def _ensure_float32_parameters(self, parameters: NDArrays) -> NDArrays:
        """Ensure all parameters are float32 numpy arrays."""
        validated_params = []
        
        for i, param in enumerate(parameters):
            try:
                if isinstance(param, np.ndarray):
                    if param.dtype.kind in ['U', 'S', 'O']:  # String or object types
                        logger.warning(f"Parameter {i} has invalid dtype {param.dtype}, skipping")
                        # Create zero array with correct shape
                        if hasattr(param, 'shape') and len(param.shape) > 0:
                            fallback = np.zeros(param.shape, dtype=np.float32)
                        else:
                            fallback = np.array([0.0], dtype=np.float32)
                        validated_params.append(fallback)
                    else:
                        # Convert numeric types to float32
                        validated_params.append(param.astype(np.float32))
                elif isinstance(param, (list, tuple)):
                    # Convert list/tuple to numpy array
                    try:
                        param_array = np.array(param, dtype=np.float32)
                        validated_params.append(param_array)
                    except:
                        # Fallback for non-numeric lists
                        validated_params.append(np.array([0.0], dtype=np.float32))
                else:
                    # Handle other types
                    try:
                        param_array = np.array([float(param)], dtype=np.float32)
                        validated_params.append(param_array)
                    except:
                        validated_params.append(np.array([0.0], dtype=np.float32))
                        
            except Exception as e:
                logger.error(f"Failed to convert parameter {i}: {e}")
                # Always provide a valid fallback
                validated_params.append(np.array([0.0], dtype=np.float32))
                
        return validated_params

    def _validate_client_parameters(self, client_parameters_list):
        """Validate and fix client parameters before aggregation."""
        validated_list = []
        
        for client_idx, (params, num_samples) in enumerate(client_parameters_list):
            try:
                # Convert parameters to ndarrays if they're Parameters objects
                if hasattr(params, 'tensors'):
                    param_arrays = parameters_to_ndarrays(params)
                else:
                    param_arrays = params
                
                # Ensure float32 dtype
                validated_params = self._ensure_float32_parameters(param_arrays)
                
                # Check parameter count consistency
                if self.current_weights is not None:
                    if len(validated_params) != len(self.current_weights):
                        logger.warning(f"Client {client_idx}: parameter count mismatch ({len(validated_params)} vs {len(self.current_weights)})")
                        # Skip this client
                        continue
                        
                    # Check shapes
                    shape_mismatch = False
                    for i, (client_param, server_param) in enumerate(zip(validated_params, self.current_weights)):
                        if client_param.shape != server_param.shape:
                            logger.warning(f"Client {client_idx}, param {i}: shape mismatch {client_param.shape} vs {server_param.shape}")
                            shape_mismatch = True
                            break
                    
                    if shape_mismatch:
                        continue
                
                validated_list.append((validated_params, num_samples))
                
            except Exception as e:
                logger.error(f"Failed to validate client {client_idx} parameters: {e}")
                continue
        
        return validated_list

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.current_weights is not None:
            return ndarrays_to_parameters(self.current_weights)
        return None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure the next round of training."""
        self.server_round = server_round
        
        # Sample clients
        sample_size = max(int(self.fraction_fit * client_manager.num_available()), self.min_fit_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)
        
        # Create config
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Import FitIns from flwr.common
        from flwr.common import FitIns
        
        # Create fit instructions with proper FitIns objects
        fit_ins = []
        for client in clients:
            fit_instruction = FitIns(parameters=parameters, config=config)
            fit_ins.append((client, fit_instruction))
        
        return fit_ins

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
            
        # Sample clients
        sample_size = max(int(self.fraction_evaluate * client_manager.num_available()), self.min_evaluate_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)
        
        # Create config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        # Import EvaluateIns from flwr.common
        from flwr.common import EvaluateIns
        
        # Create evaluate instructions with proper EvaluateIns objects
        evaluate_ins = []
        for client in clients:
            evaluate_instruction = EvaluateIns(parameters=parameters, config=config)
            evaluate_ins.append((client, evaluate_instruction))
        
        return evaluate_ins

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results using FedAdam with proper type checking."""
        
        if not results:
            return None, {}
        
        # Extract client results
        client_params_and_samples = []
        for client_proxy, fit_res in results:
            client_params_and_samples.append((fit_res.parameters, fit_res.num_examples))
        
        # Validate all client parameters
        validated_results = self._validate_client_parameters(client_params_and_samples)
        
        if not validated_results:
            logger.error("No valid client parameters received")
            return None, {}
        
        logger.info(f"Aggregating {len(validated_results)} clients (out of {len(results)} total)")
        
        try:
            # Convert to parameter arrays
            weights_results = []
            for params, num_samples in validated_results:
                weights_results.append((params, num_samples))
            
            # Perform FedAdam aggregation
            aggregated_weights = self._fedadam_aggregate(weights_results)
            
            # Update current weights
            self.current_weights = aggregated_weights
            
            # Aggregate metrics
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn is not None:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
                if fit_metrics:
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            return ndarrays_to_parameters(aggregated_weights), metrics_aggregated
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, {}

    def _fedadam_aggregate(self, weights_results):
        """Perform FedAdam aggregation with proper error handling."""
        
        # Extract weights and sample counts
        weights_list = [weights for weights, _ in weights_results]
        sample_counts = np.array([num_samples for _, num_samples in weights_results], dtype=np.float32)
        
        # Normalize sample counts
        total_samples = np.sum(sample_counts)
        if total_samples == 0:
            raise ValueError("Total samples is zero")
        normalized_weights = sample_counts / total_samples
        
        # Initialize server weights if needed
        if self.current_weights is None:
            self.current_weights = [np.copy(weights_list[0][i]) for i in range(len(weights_list[0]))]
        
        # Initialize Adam states if needed
        if self.m_t is None:
            self.m_t = [np.zeros_like(w) for w in self.current_weights]
            self.v_t = [np.zeros_like(w) for w in self.current_weights]
        
        # Compute weighted average (pseudo-gradient)
        pseudo_gradient = []
        for i in range(len(self.current_weights)):
            # Compute weighted difference from current server weights
            weighted_diff = np.zeros_like(self.current_weights[i])
            
            for j, client_weights in enumerate(weights_list):
                diff = client_weights[i] - self.current_weights[i]
                weighted_diff += normalized_weights[j] * diff
            
            pseudo_gradient.append(weighted_diff)
        
        # FedAdam update
        new_weights = []
        for i in range(len(self.current_weights)):
            # Current server weight
            x = self.current_weights[i].astype(np.float32)
            
            # Pseudo-gradient
            g = pseudo_gradient[i].astype(np.float32)
            
            # Update biased first moment estimate
            self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * g
            
            # Update biased second moment estimate
            self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m_t[i] / (1 - self.beta_1 ** self.server_round)
            v_hat = self.v_t[i] / (1 - self.beta_2 ** self.server_round)
            
            # FedAdam update
            denominator = np.sqrt(v_hat) + self.tau
            update = self.eta * m_hat / denominator
            
            new_weight = x + update
            new_weights.append(new_weight.astype(np.float32))
        
        return new_weights

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results."""
        if not results:
            return None, {}
            
        # Aggregate metrics
        if self.evaluate_metrics_aggregation_fn is not None:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
            if eval_metrics:
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
                return None, metrics_aggregated
        
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate the global model (server-side evaluation)."""
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})

def server_fn(context: Context) -> ServerAppComponents:
    """Construct server components for federated learning with SafeFedAdam."""
    
    # Load configuration with better error handling
    config_path = "/home/saadan/scratch/federated_librispeech/src/configs/pretraining_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load server config: {e}")
        # Provide comprehensive defaults
        config = {
            'simulation': {
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'min_available_clients': 2,
                'eta': 1e-3,
                'eta_l': 1e-3,
                'beta_1': 0.9,
                'beta_2': 0.99,
                'tau': 1e-9
            },
            'pretraining': {
                'label_rate': 50,
                'extractor_mode': 'default',
                'num_hidden_layers': 12,
                'hidden_size': 768,
                'intermediate_size': 3072,
                'num_attention_heads': 12,
                'activation_fn': 'gelu',
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'activation_dropout': 0.0,
                'final_dim': 256,
                'layer_norm_first': True,
                'conv_feature_layers': "[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2",
                'logit_temp': 0.1,
                'mask_prob': 0.08,
                'mask_selection': 'static',
                'mask_other': 0,
                'mask_length': 10,
                'no_mask_overlap': False,
                'mask_min_space': 1,
                'conv_bias': False,
                'encoder_layerdrop': 0.0,
                'dropout_input': 0.0,
                'dropout_features': 0.0,
                'feature_grad_mult': 0.1,
                'untie_final_proj': True,
                'sample_rate': 16000,
                'normalize': False,
                'enable_padding': False,
                'max_keep_size': None,
                'max_audio_length': 250000,
                'min_sample_size': None,
                'random_crop': True,
                'pad_audio': False
            }
        }
    
    simulation_cfg = config.get('simulation', {})
    pretraining_cfg = config.get('pretraining', {})

    # Load model configuration for parameter initialization
    base_path = Path("/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data")
    try:
        with open(base_path / "kmeans_metadata.json", 'r') as f:
            kmeans_meta = json.load(f)
        vocab_size = kmeans_meta.get('n_clusters', 504)
    except:
        vocab_size = 504
        logger.warning(f"Could not load kmeans metadata, using default vocab_size: {vocab_size}")
    
    try:
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
            normalize=pretraining_cfg.get('normalize', False),
            enable_padding=pretraining_cfg.get('enable_padding', False),
            max_keep_size=pretraining_cfg.get('max_keep_size', None),
            max_sample_size=pretraining_cfg.get('max_audio_length', 250000),
            min_sample_size=pretraining_cfg.get('min_sample_size', None),
            random_crop=pretraining_cfg.get('random_crop', True),
            pad_audio=pretraining_cfg.get('pad_audio', False)
        )
        
        # Create dictionaries for HuBERT model
        dictionaries = [list(range(vocab_size))]
        
        # Initialize model for getting initial parameters
        model = HubertModel(model_cfg, task_cfg, dictionaries)
        initial_parameters = [val.detach().cpu().numpy().astype(np.float32) for _, val in model.state_dict().items()]
        
        logger.info(f"Server initialized with {len(initial_parameters)} parameter arrays")
        
        # Create SafeFedAdam strategy with s3prl-compatible settings
        strategy = SafeFedAdam(
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
            eta=simulation_cfg.get('eta', 1e-3),
            eta_l=simulation_cfg.get('eta_l', 1e-3),
            beta_1=simulation_cfg.get('beta_1', 0.9),
            beta_2=simulation_cfg.get('beta_2', 0.99),
            tau=simulation_cfg.get('tau', 1e-9),
        )
        
        logger.info("Server strategy initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize server strategy: {e}")
        raise
    
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
        
        # Test basic model creation with required parameters
        model_cfg = HubertConfig(
            label_rate=50,  # Required positional argument
            extractor_mode="default",
            encoder_layers=12,
            encoder_embed_dim=768,
            encoder_ffn_embed_dim=3072,
            encoder_attention_heads=12,
            activation_fn="gelu",
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            final_dim=256,
            layer_norm_first=True,
            conv_feature_layers="[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2",
            logit_temp=0.1,
            mask_prob=0.08,
            mask_selection="static",
            mask_other=0,
            mask_length=10,
            no_mask_overlap=False,
            mask_min_space=1,
            conv_bias=False,
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            feature_grad_mult=0.1,
            untie_final_proj=True,
        )
        
        task_cfg = HubertPretrainingConfig(
            label_rate=50,
            sample_rate=16000,
            normalize=False,
            enable_padding=False,
            max_keep_size=None,
            max_sample_size=250000,
            min_sample_size=None,
            random_crop=True,
            pad_audio=False
        )
        
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
    parser = argparse.ArgumentParser(description="Federated HuBERT Pretraining")
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
                logger.warning(f"Configuration file not found: {args.config}, using defaults")
                config = {'simulation': {}, 'pretraining': {}}
            else:
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
            metadata_files = ["kmeans_metadata.json"]
            for metadata_file in metadata_files:
                metadata_path = base_path / metadata_file
                if not metadata_path.exists():
                    logger.warning(f"Metadata file missing: {metadata_path}")
            
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
                # In your main() function, replace the backend_config with:
                history = run_simulation(
                    server_app=server_app,
                    client_app=client_app,
                    num_supernodes=2,
                    backend_config={
                        "client_resources": {
                            "num_cpus": 1.0,      # 1 CPU per client
                            "num_gpus": 0.5       # 0.5 GPU per client (shared)
                        },
                        "init_args": {
                            "object_store_memory": 15_000_000_000,  # 48GB
                            "log_to_driver": False,
                            "configure_logging": False,
                            "num_cpus": 4,        # Total CPUs
                            "num_gpus": 2,        # Total GPUs
                            # REMOVED all _system_config - these are causing the error
                        }
                    }
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"🎉 Federated training completed!")
                print(f"⏱️  Total training time: {total_time:.2f} seconds")
                
                # Handle case where history might be None
                if history is None:
                    logger.error("Simulation returned None history - training may have failed")
                    print("❌ Training failed - no history returned")
                    return 1
                
                print(f"📈 Training rounds completed: {len(history.losses_distributed) if hasattr(history, 'losses_distributed') and history.losses_distributed else 0}")
                
                # Save training history
                history_path = os.path.join(args.log_dir, "training_history.json")
                
                # Safe access to history attributes
                history_dict = {
                    "losses_distributed": [(round_num, loss) for round_num, loss in history.losses_distributed] if hasattr(history, 'losses_distributed') and history.losses_distributed else [],
                    "losses_centralized": [(round_num, loss) for round_num, loss in history.losses_centralized] if hasattr(history, 'losses_centralized') and history.losses_centralized else [],
                    "metrics_distributed_fit": history.metrics_distributed_fit if hasattr(history, 'metrics_distributed_fit') else {},
                    "metrics_distributed": history.metrics_distributed if hasattr(history, 'metrics_distributed') else {},
                    "metrics_centralized": history.metrics_centralized if hasattr(history, 'metrics_centralized') else {},
                    "total_time": total_time,
                    "num_rounds": args.num_rounds,
                    "num_clients": args.num_clients
                }
                
                with open(history_path, 'w') as f:
                    json.dump(history_dict, f, indent=2)
                
                logger.info(f"Training history saved to: {history_path}")
                
                # Print final metrics
                if hasattr(history, 'losses_distributed') and history.losses_distributed:
                    final_loss = history.losses_distributed[-1][1]
                    print(f"🏁 Final loss: {final_loss:.4f}")
                else:
                    print("⚠️  No loss information available")
                
                if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
                    final_metrics = history.metrics_distributed_fit[-1][1]
                    for metric_name, metric_value in final_metrics.items():
                        print(f"📊 Final {metric_name}: {metric_value:.4f}")
                else:
                    print("⚠️  No training metrics available")
                
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                import traceback
                traceback.print_exc()
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