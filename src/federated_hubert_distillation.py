#!/usr/bin/env python3
"""
Federated HuBERT Knowledge Distillation System for LibriSpeech
Implements server-side teacher model with client-side student models
Knowledge distillation performed at server after aggregation
Uses flwr for federation with dynamic resource utilization
"""

import os
import json
import yaml
import logging
import argparse
import warnings
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common.typing import NDArrays
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T

import pandas as pd
import numpy as np
from tqdm import tqdm

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation import run_simulation
from flwr.common import Context, Metrics, NDArrays, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Config

from transformers import (
    HubertModel,
    HubertConfig, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)

import soundfile as sf

# Import resource optimizer
from utils.resource_optimizer import ResourceOptimizer

# Import existing components from pretraining
from federated_hubert_pretraining import (
    LibriSpeechPretrainingDataset,
    HuBERTFeatureEncoder,
    HuBERTTransformerEncoder,
    HuBERTTransformerLayer,
    HuBERTAttention,
    HuBERTFeedForward,
    HuBERTPretrainingModel
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/federated_distillation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompactHuBERTConfig:
    """Configuration for compressed student HuBERT model"""
    
    def __init__(self, 
                 hidden_size: int = 384,      # Half of teacher (DistilHuBERT approach)
                 num_hidden_layers: int = 3,  # Quarter of teacher layers
                 num_attention_heads: int = 6,# Half of teacher (divisible by hidden_size)
                 intermediate_size: int = 1536,# Half of teacher
                 vocab_size: int = 504,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 feat_extract_dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5,
                 max_position_embeddings: int = 2048):
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.feat_extract_dropout = feat_extract_dropout
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings

class CompactHuBERTFeatureEncoder(nn.Module):
    """Compact feature encoder for student model"""
    
    def __init__(self, config: CompactHuBERTConfig):
        super().__init__()
        
        # Reduced CNN layers for efficiency
        conv_layers = [
            # Layer 1: 256 channels instead of 512
            nn.Conv1d(1, 256, kernel_size=10, stride=5, bias=False),
            nn.GroupNorm(256, 256),
            nn.GELU(),
            
            # Layer 2: kernel=3, stride=2
            nn.Conv1d(256, 256, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(256, 256),
            nn.GELU(),
            
            # Layer 3: kernel=3, stride=2
            nn.Conv1d(256, 256, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(256, 256),
            nn.GELU(),
            
            # Layer 4: kernel=3, stride=2
            nn.Conv1d(256, 256, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(256, 256),
            nn.GELU(),
            
            # Layer 5: kernel=2, stride=2
            nn.Conv1d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(256, 256),
            nn.GELU(),
        ]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Feature projection to student hidden size
        self.feature_projection = nn.Linear(256, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_extract_dropout)
        
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: [batch_size, seq_len]
        hidden_states = input_values.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply convolutional layers
        hidden_states = self.conv_layers(hidden_states)  # [batch_size, 256, reduced_seq_len]
        
        # Transpose for transformer: [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)
        
        # Project to student hidden size
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class CompactHuBERTTransformerEncoder(nn.Module):
    """Compact transformer encoder for student model"""
    
    def __init__(self, config: CompactHuBERTConfig):
        super().__init__()
        
        self.config = config
        
        # Position embeddings for sequence positioning
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            HuBERTTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Add position embeddings with bounds checking
        if seq_len > self.config.max_position_embeddings:
            logger.warning(f"Sequence length {seq_len} exceeds max_position_embeddings {self.config.max_position_embeddings}. Truncating sequence.")
            hidden_states = hidden_states[:, :self.config.max_position_embeddings, :]
            seq_len = self.config.max_position_embeddings
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        all_hidden_states = []
        
        for layer in self.layers:
            # Use gradient checkpointing for memory efficiency during training
            if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
        
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'all_hidden_states': all_hidden_states
        }

class StudentHuBERTModel(nn.Module):
    """Compact student HuBERT model for knowledge distillation"""
    
    def __init__(self, config: CompactHuBERTConfig):
        super().__init__()
        
        self.config = config
        
        # Model components
        self.feature_extractor = CompactHuBERTFeatureEncoder(config)
        self.encoder = CompactHuBERTTransformerEncoder(config)
        
        # Prediction head for masked tokens (same as teacher)
        self.prediction_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Student HuBERT model initialized: {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through student model
        """
        # Extract features from raw audio
        hidden_states = self.feature_extractor(input_values)
        
        # Create proper attention mask to match feature extractor output length
        batch_size, seq_len = hidden_states.shape[:2]
        if attention_mask is None or attention_mask.shape[-1] != seq_len:
            attention_mask = torch.ones((batch_size, seq_len), device=input_values.device)
        
        # Pass through transformer encoder
        outputs = self.encoder(hidden_states, attention_mask)
        
        # Predict masked tokens
        predictions = self.prediction_head(outputs['last_hidden_state'])
        
        return {
            'predictions': predictions,
            'hidden_states': outputs['last_hidden_state'],
            'all_hidden_states': outputs['all_hidden_states']
        }

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining:
    1. Output-level distillation (soft targets)
    2. Feature-level distillation (intermediate representations)
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 feature_weight: float = 0.5,
                 student_dim: int = 384,
                 teacher_dim: int = 768,
                 layer_mapping: Optional[Dict[int, int]] = None):
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta    # Weight for task loss
        self.feature_weight = feature_weight
        
        # Layer mapping from student to teacher (DistilHuBERT approach)
        # Maps to layers 4, 8, 12 of teacher (0-indexed: 3, 7, 11)
        self.layer_mapping = layer_mapping or {0: 3, 1: 7, 2: 11}
        
        # Layer projectors for feature distillation
        self.layer_projectors = nn.ModuleDict({
            str(student_layer): nn.Linear(student_dim, teacher_dim)
            for student_layer in self.layer_mapping.keys()
        })
        
        logger.info(f"Knowledge distillation loss initialized with temperature={temperature}, alpha={alpha}")
    
    def forward(self, 
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                student_mask: Optional[torch.Tensor] = None,
                teacher_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss
        """
        losses = {}
        
        # 1. Output-level distillation (soft targets)
        student_predictions = student_outputs['predictions']
        teacher_predictions = teacher_outputs['predictions']
        
        output_distill_loss = self._compute_output_distillation(
            student_predictions, teacher_predictions, self.temperature, student_mask, teacher_mask
        )
        losses['output_distillation'] = output_distill_loss
        
        # 2. Feature-level distillation
        student_hidden_states = student_outputs['all_hidden_states']
        teacher_hidden_states = teacher_outputs['all_hidden_states']
        
        feature_distill_loss = self._compute_feature_distillation(
            student_hidden_states, teacher_hidden_states, student_mask, teacher_mask
        )
        losses['feature_distillation'] = feature_distill_loss
        
        # 3. Total distillation loss
        total_loss = output_distill_loss + self.feature_weight * feature_distill_loss
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_output_distillation(self, 
                                   student_logits: torch.Tensor,
                                   teacher_logits: torch.Tensor,
                                   temperature: float,
                                   student_mask: Optional[torch.Tensor] = None,
                                   teacher_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute output-level distillation loss (soft targets)"""
        
        # Align dimensions if needed
        if student_logits.shape != teacher_logits.shape:
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # Flatten for loss computation
        student_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Soften the logits with temperature
        student_soft = F.log_softmax(student_flat / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_flat / temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared (standard practice)
        distillation_loss = kl_loss * (temperature ** 2)
        
        return distillation_loss
    
    def _compute_feature_distillation(self,
                                    student_hidden_states: List[torch.Tensor],
                                    teacher_hidden_states: List[torch.Tensor],
                                    student_mask: Optional[torch.Tensor] = None,
                                    teacher_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute feature-level distillation loss"""
        
        feature_losses = []
        
        for student_layer, teacher_layer in self.layer_mapping.items():
            if (student_layer < len(student_hidden_states) and 
                teacher_layer < len(teacher_hidden_states)):
                
                student_hidden = student_hidden_states[student_layer]
                teacher_hidden = teacher_hidden_states[teacher_layer]
                
                # Align sequence lengths
                min_seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
                student_hidden = student_hidden[:, :min_seq_len, :]
                teacher_hidden = teacher_hidden[:, :min_seq_len, :]
                
                # Project student hidden states to teacher dimension
                student_projected = self.layer_projectors[str(student_layer)](student_hidden)
                
                # Compute MSE loss
                layer_loss = F.mse_loss(student_projected, teacher_hidden.detach())
                feature_losses.append(layer_loss)
        
        # Average over layers
        if feature_losses:
            return sum(feature_losses) / len(feature_losses)
        else:
            return torch.tensor(0.0, device=student_hidden_states[0].device)

class FederatedHuBERTDistillationClient(NumPyClient):
    """
    Flower client for federated HuBERT distillation
    Only trains student model - teacher is on server
    """
    
    def __init__(
        self,
        client_id: int,
        data_path: str,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        self.client_id = client_id
        self.data_path = Path(data_path)
        self.config = config
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Distillation Client {client_id} initialized on device: {self.device}")
        
        # Initialize model components
        self.feature_extractor = None
        self.student_model = None
        self.train_loader = None
        
        # Training parameters
        self.local_epochs = int(config['distillation']['local_epochs'])
        self.batch_size = int(config['distillation']['batch_size'])
        self.learning_rate = float(config['distillation']['learning_rate'])
        self.max_audio_length = int(config['distillation']['max_audio_length'])
        
        # Initialize components
        self._setup_feature_extractor()
        self._setup_data_loader()
        self._setup_student_model()
        
    def _setup_feature_extractor(self):
        """Initialize the feature extractor"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
    def _setup_data_loader(self):
        """Setup training data loader"""
        try:
            # Training data
            train_manifest = self.data_path / "manifest.csv"
            if train_manifest.exists():
                # Robust manifest file reading with validation
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Check file size before reading
                        if train_manifest.stat().st_size == 0:
                            raise ValueError(f"Manifest file is empty: {train_manifest}")
                        
                        # Filter for training data only
                        manifest_df = pd.read_csv(train_manifest)
                        
                        if manifest_df.empty:
                            raise ValueError(f"Manifest file has no data: {train_manifest}")
                        
                        train_df = manifest_df[manifest_df['split'] == 'train']
                        
                        if train_df.empty:
                            raise ValueError(f"No training data found in manifest: {train_manifest}")
                        
                        # Save filtered manifest
                        train_manifest_path = self.data_path / "distill_manifest.csv"
                        train_df.to_csv(train_manifest_path, index=False)
                        
                        # Verify the saved file
                        if train_manifest_path.stat().st_size == 0:
                            raise ValueError(f"Failed to create distill_manifest: {train_manifest_path}")
                        
                        break  # Success, exit retry loop
                        
                    except (pd.errors.EmptyDataError, ValueError, FileNotFoundError) as e:
                        if attempt == max_retries - 1:  # Last attempt
                            raise RuntimeError(f"Failed to process manifest file after {max_retries} attempts: {str(e)}")
                        else:
                            print(f"Attempt {attempt + 1} failed to process manifest file: {str(e)}. Retrying...")
                            import time
                            time.sleep(0.5)  # Wait before retry
                
                # Create dataset
                train_dataset = LibriSpeechPretrainingDataset(
                    manifest_file=str(train_manifest_path),
                    audio_root=str(self.data_path),
                    feature_extractor=self.feature_extractor,
                    max_length=self.max_audio_length,
                    mask_prob=float(self.config['distillation']['mask_prob']),
                    mask_length=int(self.config['distillation']['mask_length'])
                )
                
                # Dynamically determine optimal number of workers
                optimizer = ResourceOptimizer()
                optimal_workers = min(optimizer.optimal_workers, 8)  # Cap at 8 to avoid overhead
                
                # Create data loader
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=optimal_workers,
                    pin_memory=True if self.device.type == "cuda" else False,
                    persistent_workers=True if optimal_workers > 0 else False
                )
                
                logger.info(f"Distillation Client {self.client_id}: Loaded {len(train_dataset)} training samples")
                
            else:
                # Check if any alternative manifest files exist
                alternative_manifests = list(self.data_path.glob("*manifest*.csv"))
                if alternative_manifests:
                    logger.warning(f"Primary manifest not found, but found alternatives: {alternative_manifests}")
                    logger.warning(f"Using first available: {alternative_manifests[0]}")
                    
                    # Try to use the first available manifest
                    train_dataset = LibriSpeechPretrainingDataset(
                        manifest_file=str(alternative_manifests[0]),
                        audio_root=str(self.data_path),
                        feature_extractor=self.feature_extractor,
                        max_length=self.max_audio_length,
                        mask_prob=float(self.config['distillation']['mask_prob']),
                        mask_length=int(self.config['distillation']['mask_length'])
                    )
                    
                    # Dynamically determine optimal number of workers
                    optimizer = ResourceOptimizer()
                    optimal_workers = min(optimizer.optimal_workers, 8)  # Cap at 8 to avoid overhead
                    
                    # Create data loader
                    self.train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=optimal_workers,
                        pin_memory=True if self.device.type == "cuda" else False,
                        persistent_workers=True if optimal_workers > 0 else False
                    )
                    
                    logger.info(f"Distillation Client {self.client_id}: Loaded {len(train_dataset)} training samples from alternative manifest")
                else:
                    raise FileNotFoundError(f"No manifest files found in {self.data_path}. Available files: {list(self.data_path.glob('*'))}")
                
        except Exception as e:
            logger.error(f"Error setting up data loader for distillation client {self.client_id}: {e}")
            # Additional debug information
            logger.error(f"Data path: {self.data_path}")
            logger.error(f"Files in data path: {list(self.data_path.glob('*')) if self.data_path.exists() else 'Directory does not exist'}")
            raise
    
    def _setup_student_model(self):
        """Initialize the student HuBERT model"""
        model_config = self.config['distillation']
        
        # Create compact config for student
        student_config = CompactHuBERTConfig(
            hidden_size=int(model_config.get('student_hidden_size', 384)),
            num_hidden_layers=int(model_config.get('student_num_layers', 3)),
            num_attention_heads=int(model_config.get('student_num_heads', 6)),
            intermediate_size=int(model_config.get('student_intermediate_size', 1536)),
            vocab_size=int(model_config.get('vocab_size', 504)),
            hidden_dropout_prob=float(model_config.get('dropout', 0.1)),
            attention_probs_dropout_prob=float(model_config.get('dropout', 0.1))
        )
        
        self.student_model = StudentHuBERTModel(student_config).to(self.device)
        
        # Ensure all parameters require gradients
        for param in self.student_model.parameters():
            param.requires_grad = True
        
        logger.info(f"Distillation Client {self.client_id}: Student model initialized")
    
    def get_parameters(self, config: Config) -> NDArrays:
        """Extract model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.student_model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays"""
        if not parameters:
            return
            
        current_state = self.student_model.state_dict()
        param_keys = list(current_state.keys())
        
        # Only update parameters that have matching shapes
        state_dict = OrderedDict()
        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            if i >= len(parameters):
                break
                
            param_tensor = torch.tensor(param_array)
            current_shape = current_state[key].shape
            
            # Only update if shapes match
            if param_tensor.shape == current_shape:
                state_dict[key] = param_tensor
            else:
                # Keep existing parameter for mismatched shapes
                state_dict[key] = current_state[key]
                logger.debug(f"Skipping parameter {key}: shape mismatch {param_tensor.shape} vs {current_shape}")
        
        self.student_model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """Train the student model (distillation happens on server)"""
        logger.info(f"Distillation Client {self.client_id}: Starting student training")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.config['distillation'].get('weight_decay', 0.01)),
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=int(self.local_epochs * len(self.train_loader))
        )
        
        # Setup mixed precision training if GPU is available
        use_amp = self.device.type == "cuda" and self.config['client'].get('mixed_precision', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        if use_amp:
            logger.info(f"Client {self.client_id}: Using automatic mixed precision (AMP)")
        
        # Optimize GPU memory usage
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Set memory fraction to be more conservative for federated learning
            torch.cuda.set_per_process_memory_fraction(0.2)  # Use only 20% per client
        
        # Training loop - student self-supervised learning
        self.student_model.train()
        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        
        # Check if data loader has any data
        if len(self.train_loader) == 0:
            logger.warning(f"Distillation Client {self.client_id}: No training data available")
            return self.get_parameters(config={}), 0, {"student_loss": 0.0}
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch in tqdm(self.train_loader, desc=f"Distillation Client {self.client_id} Epoch {epoch+1}"):
                try:
                    # Move batch to device with memory management
                    input_values = batch['input_values'].to(self.device)
                    
                    # Truncate sequences if too long for memory efficiency
                    max_len = min(input_values.shape[-1], 80000)  # 5 seconds at 16kHz
                    input_values = input_values[:, :max_len]
                    
                    # Create proper attention mask based on actual feature extraction
                    # The feature extractor reduces sequence length by factor of 80 (5*2*2*2*2)
                    # So we need to create a proper attention mask for the reduced sequence
                    batch_size = input_values.shape[0]
                    expected_seq_len = max_len // 80  # Based on feature extraction reduction
                    attention_mask = torch.ones((batch_size, expected_seq_len), device=input_values.device)
                    
                    # Forward pass with mixed precision
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.student_model(
                                input_values=input_values,
                                attention_mask=attention_mask
                            )
                            # Simple self-supervised loss for student
                            loss = self._compute_self_supervised_loss(outputs, input_values, attention_mask)
                    else:
                        outputs = self.student_model(
                            input_values=input_values,
                            attention_mask=attention_mask
                        )
                        loss = self._compute_self_supervised_loss(outputs, input_values, attention_mask)
                except Exception as e:
                    logger.warning(f"Distillation Client {self.client_id}: Error processing batch: {e}")
                    continue
                
                if loss is not None and loss.requires_grad:
                    # Backward pass with mixed precision
                    if use_amp:
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping with scaling
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                    
                    scheduler.step()
                elif loss is not None:
                    logger.warning(f"Loss tensor does not require gradients: {loss.requires_grad}, skipping backward pass")
                    
                    # Accumulate metrics
                    batch_size = input_values.size(0)
                    epoch_loss += loss.item() * batch_size
                    epoch_samples += batch_size
                    num_batches += 1
                
            if epoch_samples > 0:
                total_loss += epoch_loss
                num_samples += epoch_samples
                
                avg_epoch_loss = epoch_loss / epoch_samples
                logger.info(f"Distillation Client {self.client_id} Epoch {epoch+1}: Loss = {avg_epoch_loss:.4f}")
                
                # Memory cleanup between epochs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"Distillation Client {self.client_id}: Training completed. Average loss: {avg_loss:.4f}")
        
        return self.get_parameters(config={}), num_samples, {"student_loss": avg_loss}
    
    def _compute_self_supervised_loss(self, outputs: Dict[str, torch.Tensor], 
                                    input_values: torch.Tensor, 
                                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute self-supervised loss for student model"""
        
        # Use hidden states for self-supervised learning
        if 'last_hidden_state' in outputs:
            hidden_states = outputs['last_hidden_state']
        elif 'hidden_states' in outputs:
            hidden_states = outputs['hidden_states']
        else:
            # Fallback: create a dummy loss that requires grad
            batch_size = input_values.shape[0]
            return torch.zeros(1, requires_grad=True, device=input_values.device)
        
        # Compute self-supervised loss (contrastive learning style)
        # Pool the hidden states
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Encourage non-zero representations (prevents collapse)
        # Use L2 norm and encourage diversity
        norms = torch.norm(pooled, dim=1)  # [batch_size]
        
        # Loss: encourage non-zero norms while preventing explosion
        # This encourages the model to produce meaningful representations
        target_norm = 1.0
        loss = F.mse_loss(norms, torch.full_like(norms, target_norm))
        
        return loss
    
    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the student model"""
        logger.info(f"Distillation Client {self.client_id}: Evaluation")
        
        try:
            # Set parameters
            self.set_parameters(parameters)
        except Exception as e:
            logger.warning(f"Failed to set parameters during evaluation: {e}")
            # Return reasonable defaults if parameter loading fails
            return 0.0, 0, {"eval_student_loss": 0.0}
        
        # Simple evaluation on training data
        self.student_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= 3:  # Evaluate on first 3 batches only for memory efficiency
                    break
                    
                # Move batch to device with memory management
                input_values = batch['input_values'].to(self.device)
                
                # Truncate sequences if too long for memory efficiency
                max_len = min(input_values.shape[-1], 80000)  # 5 seconds at 16kHz
                input_values = input_values[:, :max_len]
                
                # Create proper attention mask based on actual feature extraction
                # The feature extractor reduces sequence length by factor of 80 (5*2*2*2*2)
                batch_size = input_values.shape[0]
                expected_seq_len = max_len // 80  # Based on feature extraction reduction
                attention_mask = torch.ones((batch_size, expected_seq_len), device=input_values.device)
                
                # Forward pass
                outputs = self.student_model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                
                loss = self._compute_self_supervised_loss(outputs, input_values, attention_mask)
                
                if loss is not None:
                    batch_size = input_values.size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size
        
        # Calculate metrics
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"Distillation Client {self.client_id}: Evaluation completed. Loss: {avg_loss:.4f}")
        
        return avg_loss, num_samples, {"eval_student_loss": avg_loss}

class ServerSideDistillationStrategy(FedAvg):
    """
    Custom Flower strategy with server-side knowledge distillation
    Maintains teacher model on server and performs distillation after aggregation
    """
    
    def __init__(self, 
                 teacher_model: HuBERTPretrainingModel,
                 device: torch.device,
                 distillation_config: Dict[str, Any],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.teacher_model = teacher_model
        self.device = device
        self.distillation_config = distillation_config
        
        # Move teacher to device and freeze
        self.teacher_model.to(device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Server-side distillation parameters
        self.distillation_temperature = float(distillation_config.get('temperature', 4.0))
        self.distillation_weight = float(distillation_config.get('weight', 0.5))
        
        # Setup distillation loss
        self.distillation_loss = KnowledgeDistillationLoss(
            temperature=self.distillation_temperature,
            alpha=self.distillation_weight,
            student_dim=int(distillation_config.get('student_hidden_size', 384)),
            teacher_dim=int(distillation_config.get('teacher_hidden_size', 768))
        ).to(device)
        
        logger.info("Server-side distillation strategy initialized")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate client results and perform server-side distillation"""
        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")
        
        # Check if any client has valid examples
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            logger.warning(f"Round {server_round}: All clients returned 0 examples, skipping aggregation")
            # Return the current parameters without aggregation
            if hasattr(self, '_current_parameters') and self._current_parameters is not None:
                return self._current_parameters, {}
            else:
                # If no current parameters, return None to signal failure
                return None, {}
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Store current parameters for potential fallback
        if aggregated_parameters is not None:
            self._current_parameters = aggregated_parameters
        
        if aggregated_parameters is not None:
            # Perform server-side distillation
            distilled_parameters = self._server_side_distillation(aggregated_parameters, server_round)
            
            # Return distilled parameters
            return distilled_parameters, aggregated_metrics
        
        return aggregated_parameters, aggregated_metrics
    
    def _server_side_distillation(self, parameters: Parameters, server_round: int) -> Parameters:
        """
        Perform server-side parameter distillation from teacher to aggregated student
        """
        logger.info(f"Round {server_round}: Starting server-side parameter distillation")
        
        # Convert parameters to numpy arrays
        student_ndarrays = parameters_to_ndarrays(parameters)
        
        # Create temporary student model to perform distillation
        temp_config = CompactHuBERTConfig()
        temp_student = StudentHuBERTModel(temp_config).to(self.device)
        
        # Load aggregated parameters into temporary student
        current_state = temp_student.state_dict()
        param_keys = list(current_state.keys())
        
        state_dict = OrderedDict()
        for i, (key, param_array) in enumerate(zip(param_keys, student_ndarrays)):
            if i >= len(student_ndarrays):
                break
                
            param_tensor = torch.tensor(param_array, device=self.device)
            current_shape = current_state[key].shape
            
            if param_tensor.shape == current_shape:
                state_dict[key] = param_tensor
            else:
                state_dict[key] = current_state[key]
        
        temp_student.load_state_dict(state_dict, strict=False)
        
        # Perform parameter-level distillation
        with torch.no_grad():
            teacher_state = self.teacher_model.state_dict()
            student_state = temp_student.state_dict()
            
            # Apply weighted combination of teacher and student parameters
            for key in student_state.keys():
                if key in teacher_state:
                    teacher_param = teacher_state[key]
                    student_param = student_state[key]
                    
                    # Apply distillation if shapes are compatible
                    if teacher_param.shape == student_param.shape:
                        # Weighted combination: student + α * (teacher - student)
                        student_state[key] = student_param + self.distillation_weight * (teacher_param - student_param)
                    # For different shapes, try dimension-wise distillation
                    elif len(teacher_param.shape) == len(student_param.shape):
                        # Try to align dimensions
                        min_dims = [min(t, s) for t, s in zip(teacher_param.shape, student_param.shape)]
                        if all(d > 0 for d in min_dims):
                            # Create slices for compatible dimensions
                            teacher_slice = teacher_param
                            student_slice = student_param
                            
                            # Apply slicing to match dimensions
                            for dim, size in enumerate(min_dims):
                                teacher_slice = teacher_slice.narrow(dim, 0, size)
                                student_slice = student_slice.narrow(dim, 0, size)
                            
                            # Update only the compatible portion
                            student_state[key].narrow(dim, 0, size).copy_(
                                student_slice + self.distillation_weight * (teacher_slice - student_slice)
                            )
        
        # Extract distilled parameters
        distilled_ndarrays = [param.cpu().numpy() for param in temp_student.state_dict().values()]
        
        logger.info(f"Round {server_round}: Parameter distillation completed")
        
        # Convert back to parameters
        return ndarrays_to_parameters(distilled_ndarrays)

def client_fn(context: Context) -> Client:
    """Create a Flower client instance for distillation"""
    
    # Load configuration - try optimized first, fall back to original
    optimized_config_path = "configs/distillation_config_optimized.yaml"
    original_config_path = "configs/distillation_config.yaml"
    
    config_path = optimized_config_path if Path(optimized_config_path).exists() else original_config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get client ID from context and map to sequential ID
    node_id = context.node_id
    
    # Get the total number of clients from config
    num_clients = int(config['simulation']['num_supernodes'])
    
    # Create a mapping from node_id to sequential client_id
    client_id = hash(str(node_id)) % num_clients
    
    logger.info(f"Mapping node_id {node_id} to client_id {client_id}")
    
    # Setup client data path
    data_root = Path(config['data']['partitioned_data_root'])
    client_data_path = data_root / f"client_{client_id}"
    
    logger.info(f"Client data path: {client_data_path}")
    
    # Check if the client data path exists
    if not client_data_path.exists():
        logger.error(f"Client data path does not exist: {client_data_path}")
        raise FileNotFoundError(f"Client data directory not found: {client_data_path}")
    
    # Check if manifest file exists
    manifest_path = client_data_path / "manifest.csv"
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Create and return client
    client = FederatedHuBERTDistillationClient(
        client_id=client_id,
        data_path=str(client_data_path),
        config=config
    )
    
    return client

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics"""
    
    # Calculate weighted averages
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        weighted_sum = sum(num_examples * m[key] for num_examples, m in metrics)
        weighted_metrics[key] = weighted_sum / total_examples
    
    return weighted_metrics

def server_fn(context: Context) -> ServerAppComponents:
    """Create server app components for distillation"""
    
    # Load configuration - try optimized first, fall back to original
    optimized_config_path = "configs/distillation_config_optimized.yaml"
    original_config_path = "configs/distillation_config.yaml"
    
    config_path = optimized_config_path if Path(optimized_config_path).exists() else original_config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize teacher model (full HuBERT model)
    teacher_config = config['distillation']
    teacher_model = HuBERTPretrainingModel(
        vocab_size=int(teacher_config.get('vocab_size', 504)),
        hidden_size=int(teacher_config.get('teacher_hidden_size', 768)),
        num_hidden_layers=int(teacher_config.get('teacher_num_layers', 12)),
        num_attention_heads=int(teacher_config.get('teacher_num_heads', 12)),
        intermediate_size=int(teacher_config.get('teacher_intermediate_size', 3072))
    )
    
    # Load pretrained teacher weights if available
    teacher_weights_path = config.get('teacher_weights_path', 'models/pretrained_hubert.pt')
    if Path(teacher_weights_path).exists():
        checkpoint = torch.load(teacher_weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Loaded teacher weights from {teacher_weights_path}")
        else:
            logger.warning(f"No model_state_dict found in {teacher_weights_path}")
    else:
        logger.warning(f"Teacher weights not found at {teacher_weights_path}, using random initialization")
    
    strategy_config = config['strategy']
    
    # Create strategy with server-side distillation
    strategy = ServerSideDistillationStrategy(
        teacher_model=teacher_model,
        device=device,
        distillation_config=teacher_config,
        fraction_fit=float(strategy_config['fraction_fit']),
        fraction_evaluate=float(strategy_config['fraction_evaluate']),
        min_fit_clients=int(strategy_config['min_fit_clients']),
        min_evaluate_clients=int(strategy_config['min_evaluate_clients']),
        min_available_clients=int(strategy_config['min_available_clients']),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    # Create server config
    server_config = ServerConfig(num_rounds=int(teacher_config['num_rounds']))
    
    return ServerAppComponents(strategy=strategy, config=server_config)

def main():
    """Main distillation function"""
    parser = argparse.ArgumentParser(description="Federated HuBERT Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/distillation_config.yaml")
    parser.add_argument("--simulation", action="store_true", help="Run simulation")
    parser.add_argument("--num-clients", type=int, help="Override number of clients")
    parser.add_argument("--num-rounds", type=int, help="Override number of rounds")
    parser.add_argument("--no-optimize", action="store_true", help="Skip automatic resource optimization")
    
    args = parser.parse_args()
    
    # Optimize configuration for current hardware
    if not args.no_optimize:
        logger.info("Detecting and optimizing for available hardware resources...")
        optimizer = ResourceOptimizer()
        optimizer.print_resource_summary()
        
        # Create optimized config
        optimized_config_path = args.config.replace('.yaml', '_optimized.yaml')
        optimizer.save_optimized_config(args.config, optimized_config_path, args.num_clients)
        config_to_use = optimized_config_path
        logger.info(f"Using optimized configuration: {config_to_use}")
    else:
        config_to_use = args.config
        logger.info(f"Using original configuration: {config_to_use}")
    
    # Load configuration
    with open(config_to_use, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config parameters if provided
    if args.num_clients:
        config['simulation']['num_supernodes'] = int(args.num_clients)
    if args.num_rounds:
        config['distillation']['num_rounds'] = int(args.num_rounds)
    
    logger.info("Starting Federated HuBERT Knowledge Distillation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Clients: {config['simulation']['num_supernodes']}")
    logger.info(f"Rounds: {config['distillation']['num_rounds']}")
    
    if args.simulation:
        # Run simulation
        logger.info("Running federated distillation simulation...")
        
        # Create apps
        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(server_fn=server_fn)
        
        # Backend configuration
        backend_config = config['simulation']['backend']['config']
        
        # Run simulation
        try:
            history = run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=int(config['simulation']['num_supernodes']),
                backend_config=backend_config,
            )
            
            logger.info("Distillation simulation completed!")
            
            # Log final results
            if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
                total_rounds = len(history.metrics_distributed)
                logger.info(f"Distillation completed after {total_rounds} rounds")
        
        except Exception as e:
            logger.error(f"Distillation simulation failed: {e}")
            raise
    
    else:
        logger.info("Non-simulation mode not implemented. Use --simulation flag.")

if __name__ == "__main__":
    main()