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
        logging.FileHandler(
            '/home/saadan/scratch/federated_librispeech/src/logs/federated_distillation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompactHuBERTConfig:
    """Configuration for compressed student HuBERT model"""

    def __init__(self,
                 # Half of teacher (DistilHuBERT approach)
                 hidden_size: int = 384,
                 num_hidden_layers: int = 3,  # Quarter of teacher layers
                 # Half of teacher (divisible by hidden_size)
                 num_attention_heads: int = 6,
                 intermediate_size: int = 1536,  # Half of teacher
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
        # [batch_size, 256, reduced_seq_len]
        hidden_states = self.conv_layers(hidden_states)

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
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([
            HuBERTTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]

        # Add position embeddings with bounds checking
        if seq_len > self.config.max_position_embeddings:
            logger.warning(
                f"Sequence length {seq_len} exceeds max_position_embeddings {self.config.max_position_embeddings}. Truncating sequence.")
            hidden_states = hidden_states[:,
                                          :self.config.max_position_embeddings, :]
            seq_len = self.config.max_position_embeddings
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]

        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings

        all_hidden_states = []

        for layer in self.layers:
            # Use gradient checkpointing for memory efficiency during training
            if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False)
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

        logger.info(
            f"Student HuBERT model initialized: {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

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
            # Create attention mask based on the feature extractor output length
            attention_mask = torch.ones(
                (batch_size, seq_len), device=input_values.device)
        elif attention_mask.shape[-1] != seq_len:
            # Adjust attention mask to match feature extractor output
            # The feature extractor reduces sequence length significantly
            # We need to downsample the attention mask accordingly
            original_length = attention_mask.shape[-1]
            reduction_factor = original_length // seq_len if seq_len > 0 else 1

            if reduction_factor > 1:
                # Downsample attention mask by taking every reduction_factor-th element
                attention_mask = attention_mask[:, ::reduction_factor]

            # Ensure exact length match
            if attention_mask.shape[-1] > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            elif attention_mask.shape[-1] < seq_len:
                # Pad with ones if needed
                padding = seq_len - attention_mask.shape[-1]
                attention_mask = F.pad(attention_mask, (0, padding), value=1)

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

        logger.info(
            f"Knowledge distillation loss initialized with temperature={temperature}, alpha={alpha}")

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
                min_seq_len = min(student_hidden.size(1),
                                  teacher_hidden.size(1))
                student_hidden = student_hidden[:, :min_seq_len, :]
                teacher_hidden = teacher_hidden[:, :min_seq_len, :]

                # Project student hidden states to teacher dimension
                student_projected = self.layer_projectors[str(
                    student_layer)](student_hidden)

                # Compute MSE loss
                layer_loss = F.mse_loss(
                    student_projected, teacher_hidden.detach())
                feature_losses.append(layer_loss)

        # Average over layers
        if feature_losses:
            return sum(feature_losses) / len(feature_losses)
        else:
            return torch.tensor(0.0, device=student_hidden_states[0].device)


class FederatedHuBERTDistillationClient(NumPyClient):
    """
    Flower client for federated HuBERT distillation
    Follows standard knowledge distillation approach with teacher-student interaction
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            f"Distillation Client {client_id} initialized on device: {self.device}")

        # Initialize model components
        self.feature_extractor = None
        self.student_model = None
        self.teacher_model = None  # Local teacher model for distillation
        self.train_loader = None

        # Training parameters
        self.local_epochs = int(config['distillation']['local_epochs'])
        self.batch_size = int(config['distillation']['batch_size'])
        self.learning_rate = float(config['distillation']['learning_rate'])
        self.max_audio_length = int(config['distillation']['max_audio_length'])

        # Distillation parameters
        self.temperature = float(
            config['distillation'].get('temperature', 4.0))
        self.alpha = float(config['distillation'].get(
            'alpha', 0.7))  # Weight for distillation loss
        self.beta = float(config['distillation'].get(
            'beta', 0.3))    # Weight for task loss

        # Initialize components
        self._setup_feature_extractor()
        self._setup_data_loader()
        self._setup_models()

    def _setup_feature_extractor(self):
        """Initialize the feature extractor"""
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base")

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
                            raise ValueError(
                                f"Manifest file is empty: {train_manifest}")

                        # Filter for training data only
                        manifest_df = pd.read_csv(train_manifest)

                        if manifest_df.empty:
                            raise ValueError(
                                f"Manifest file has no data: {train_manifest}")

                        train_df = manifest_df[manifest_df['split'] == 'train']

                        if train_df.empty:
                            raise ValueError(
                                f"No training data found in manifest: {train_manifest}")

                        # Save filtered manifest
                        train_manifest_path = self.data_path / "distill_manifest.csv"
                        train_df.to_csv(train_manifest_path, index=False)

                        # Verify the saved file
                        if train_manifest_path.stat().st_size == 0:
                            raise ValueError(
                                f"Failed to create distill_manifest: {train_manifest_path}")

                        break  # Success, exit retry loop

                    except (pd.errors.EmptyDataError, ValueError, FileNotFoundError) as e:
                        if attempt == max_retries - 1:  # Last attempt
                            raise RuntimeError(
                                f"Failed to process manifest file after {max_retries} attempts: {str(e)}")
                        else:
                            print(
                                f"Attempt {attempt + 1} failed to process manifest file: {str(e)}. Retrying...")
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
                # Cap at 8 to avoid overhead
                optimal_workers = min(optimizer.optimal_workers, 8)

                # Create data loader
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=optimal_workers,
                    pin_memory=True if self.device.type == "cuda" else False,
                    persistent_workers=True if optimal_workers > 0 else False,
                    collate_fn=custom_collate_fn  # Use custom collate function
                )

                logger.info(
                    f"Distillation Client {self.client_id}: Loaded {len(train_dataset)} training samples")

            else:
                # Check if any alternative manifest files exist
                alternative_manifests = list(
                    self.data_path.glob("*manifest*.csv"))
                if alternative_manifests:
                    logger.warning(
                        f"Primary manifest not found, but found alternatives: {alternative_manifests}")
                    logger.warning(
                        f"Using first available: {alternative_manifests[0]}")

                    # Try to use the first available manifest
                    train_dataset = LibriSpeechPretrainingDataset(
                        manifest_file=str(alternative_manifests[0]),
                        audio_root=str(self.data_path),
                        feature_extractor=self.feature_extractor,
                        max_length=self.max_audio_length,
                        mask_prob=float(
                            self.config['distillation']['mask_prob']),
                        mask_length=int(
                            self.config['distillation']['mask_length'])
                    )

                    # Dynamically determine optimal number of workers
                    optimizer = ResourceOptimizer()
                    # Cap at 8 to avoid overhead
                    optimal_workers = min(optimizer.optimal_workers, 8)

                    # Create data loader
                    self.train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=optimal_workers,
                        pin_memory=True if self.device.type == "cuda" else False,
                        persistent_workers=True if optimal_workers > 0 else False,
                        collate_fn=custom_collate_fn  # Use custom collate function
                    )

                    logger.info(
                        f"Distillation Client {self.client_id}: Loaded {len(train_dataset)} training samples from alternative manifest")
                else:
                    raise FileNotFoundError(
                        f"No manifest files found in {self.data_path}. Available files: {list(self.data_path.glob('*'))}")

        except Exception as e:
            logger.error(
                f"Error setting up data loader for distillation client {self.client_id}: {e}")
            # Additional debug information
            logger.error(f"Data path: {self.data_path}")
            logger.error(
                f"Files in data path: {list(self.data_path.glob('*')) if self.data_path.exists() else 'Directory does not exist'}")
            raise

    def _setup_models(self):
        """Initialize both student and teacher models"""
        model_config = self.config['distillation']

        # Create compact config for student (DistilHuBERT approach)
        student_config = CompactHuBERTConfig(
            hidden_size=int(model_config.get('student_hidden_size', 384)),
            num_hidden_layers=int(model_config.get('student_num_layers', 3)),
            num_attention_heads=int(model_config.get('student_num_heads', 6)),
            intermediate_size=int(model_config.get(
                'student_intermediate_size', 1536)),
            vocab_size=int(model_config.get('vocab_size', 504)),
            hidden_dropout_prob=float(model_config.get('dropout', 0.1)),
            attention_probs_dropout_prob=float(
                model_config.get('dropout', 0.1))
        )

        self.student_model = StudentHuBERTModel(student_config).to(self.device)

        # Create teacher model (full HuBERT)
        teacher_config = model_config
        self.teacher_model = HuBERTPretrainingModel(
            vocab_size=int(teacher_config.get('vocab_size', 504)),
            hidden_size=int(teacher_config.get('teacher_hidden_size', 768)),
            num_hidden_layers=int(
                teacher_config.get('teacher_num_layers', 12)),
            num_attention_heads=int(
                teacher_config.get('teacher_num_heads', 12)),
            intermediate_size=int(teacher_config.get(
                'teacher_intermediate_size', 3072))
        ).to(self.device)

        # Load teacher weights from server or local checkpoint
        self._load_teacher_weights()

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        # Ensure all student parameters require gradients
        for param in self.student_model.parameters():
            param.requires_grad = True

        logger.info(
            f"Distillation Client {self.client_id}: Student and teacher models initialized")

    def _load_teacher_weights(self):
        """Load teacher model weights from server or local checkpoint"""
        try:
            # Try to load from local checkpoint first
            teacher_weights_path = self.config['distillation'].get(
                'teacher_weights_path', None)
            if teacher_weights_path and Path(teacher_weights_path).exists():
                checkpoint = torch.load(
                    teacher_weights_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.teacher_model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    logger.info(
                        f"Loaded teacher weights from local checkpoint: {teacher_weights_path}")
                    return
                else:
                    logger.warning(
                        f"No model_state_dict found in {teacher_weights_path}")

            # Try to load from HuggingFace
            use_pretrained = self.config['distillation'].get(
                'use_pretrained_teacher', True)
            if use_pretrained:
                success = load_huggingface_hubert_teacher(
                    self.teacher_model, self.device)
                if success:
                    logger.info("Loaded teacher weights from HuggingFace")
                    return

            # Initialize with random weights if all else fails
            logger.warning("Using random teacher initialization")

        except Exception as e:
            logger.error(f"Failed to load teacher weights: {e}")
            logger.warning("Using random teacher initialization")

    def get_parameters(self, config: Config) -> NDArrays:
        """Extract student model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.student_model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set student model parameters from numpy arrays"""
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
                logger.debug(
                    f"Skipping parameter {key}: shape mismatch {param_tensor.shape} vs {current_shape}")

        self.student_model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        """Train the student model using knowledge distillation with teacher"""
        logger.info(
            f"Distillation Client {self.client_id}: Starting knowledge distillation training")

        # Set parameters
        self.set_parameters(parameters)

        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(
                self.config['distillation'].get('weight_decay', 0.01)),
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=int(self.local_epochs * len(self.train_loader))
        )

        # Setup mixed precision training if GPU is available
        use_amp = self.device.type == "cuda" and self.config.get(
            'client', {}).get('mixed_precision', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        if use_amp:
            logger.info(
                f"Client {self.client_id}: Using automatic mixed precision (AMP)")

        # Optimize GPU memory usage
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Set memory fraction to be more conservative for federated learning
            torch.cuda.set_per_process_memory_fraction(
                0.3)  # Increased from 20% to 30%

        # Training loop - knowledge distillation
        self.student_model.train()
        self.teacher_model.eval()  # Ensure teacher is in eval mode

        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        num_samples = 0
        num_batches = 0

        # Check if data loader has any data
        if len(self.train_loader) == 0:
            logger.warning(
                f"Distillation Client {self.client_id}: No training data available")
            # Return 1 sample to avoid division by zero
            return self.get_parameters(config={}), 1, {"student_loss": 0.0}

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_task_loss = 0.0
            epoch_samples = 0

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Distillation Client {self.client_id} Epoch {epoch+1}")):
                try:
                    # Skip empty batches (all items were None)
                    if isinstance(batch, dict) and batch.get('empty_batch', False):
                        logger.warning(
                            f"Distillation Client {self.client_id}: Skipping empty batch at index {batch_idx} "
                            f"(0/{batch.get('total_samples', 0)} valid samples)")
                        continue
                    
                    # Log batch composition if available
                    if isinstance(batch, dict) and 'valid_samples' in batch and 'total_samples' in batch:
                        valid_samples = batch['valid_samples']
                        total_samples = batch['total_samples']
                        if valid_samples < total_samples:
                            logger.info(f"Distillation Client {self.client_id}: Processing batch {batch_idx} "
                                      f"with {valid_samples}/{total_samples} valid samples")

                    # Move batch to device with memory management
                    input_values = batch['input_values'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    # Truncate sequences if too long for memory efficiency
                    # 5 seconds at 16kHz
                    max_len = min(input_values.shape[-1], 80000)
                    input_values = input_values[:, :max_len]

                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :max_len]

                    # Zero gradients
                    optimizer.zero_grad()

                    # Mixed precision training
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            # Teacher forward pass (no gradients)
                            with torch.no_grad():
                                teacher_outputs = self.teacher_model(
                                    input_values=input_values,
                                    attention_mask=attention_mask
                                )

                            # Student forward pass
                            student_outputs = self.student_model(
                                input_values=input_values,
                                attention_mask=attention_mask
                            )

                            # Compute losses
                            distill_loss = self._compute_knowledge_distillation_loss(
                                student_outputs, teacher_outputs, attention_mask
                            )

                            # Task-specific loss (masked language modeling)
                            task_loss = self._compute_masked_language_modeling_loss(
                                student_outputs, input_values, attention_mask
                            )

                            # Combined loss: α * task_loss + (1-α) * distill_loss
                            total_loss = self.alpha * task_loss + \
                                (1 - self.alpha) * distill_loss
                    else:
                        # Teacher forward pass (no gradients)
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(
                                input_values=input_values,
                                attention_mask=attention_mask
                            )

                        # Student forward pass
                        student_outputs = self.student_model(
                            input_values=input_values,
                            attention_mask=attention_mask
                        )

                        # Compute losses
                        distill_loss = self._compute_knowledge_distillation_loss(
                            student_outputs, teacher_outputs, attention_mask
                        )

                        # Task-specific loss (masked language modeling)
                        task_loss = self._compute_masked_language_modeling_loss(
                            student_outputs, input_values, attention_mask
                        )

                        # Combined loss: α * task_loss + (1-α) * distill_loss
                        total_loss = self.alpha * task_loss + \
                            (1 - self.alpha) * distill_loss

                    # Ensure loss requires gradients and is valid
                    if total_loss is None or torch.isnan(total_loss) or torch.isinf(total_loss):
                        logger.warning(
                            f"Invalid loss detected, skipping batch {batch_idx}")
                        continue

                    if not total_loss.requires_grad:
                        logger.warning(
                            f"Loss does not require gradients: {total_loss.requires_grad}, skipping batch {batch_idx}")
                        continue

                    # Backward pass
                    if use_amp:
                        scaler.scale(total_loss).backward()

                        # Gradient clipping with scaling
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.parameters(), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.parameters(), max_norm=1.0)

                        optimizer.step()

                    # Update learning rate
                    scheduler.step()

                    # Update metrics
                    batch_size = input_values.size(0)
                    epoch_loss += total_loss.item() * batch_size
                    epoch_distill_loss += distill_loss.item() * batch_size
                    epoch_task_loss += task_loss.item() * batch_size
                    epoch_samples += batch_size

                    # Memory cleanup
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(
                        f"Distillation Client {self.client_id}: Error processing batch {batch_idx}: {e}")
                    continue

            # Log epoch statistics
            if epoch_samples > 0:
                avg_loss = epoch_loss / epoch_samples
                avg_distill_loss = epoch_distill_loss / epoch_samples
                avg_task_loss = epoch_task_loss / epoch_samples

                logger.info(
                    f"Distillation Client {self.client_id} Epoch {epoch+1}: "
                    f"Loss={avg_loss:.4f}, Distill={avg_distill_loss:.4f}, Task={avg_task_loss:.4f}, "
                    f"Samples={epoch_samples}")

                total_loss += epoch_loss
                total_distill_loss += epoch_distill_loss
                total_task_loss += epoch_task_loss
                num_samples += epoch_samples

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                logger.warning(
                    f"Distillation Client {self.client_id} Epoch {epoch+1}: No valid samples processed")

        # Ensure we have processed some samples
        if num_samples == 0:
            logger.error(
                f"Distillation Client {self.client_id}: No samples processed during training")
            # Return 1 to avoid division by zero
            return self.get_parameters(config={}), 1, {"student_loss": 0.0}

        # Calculate average losses
        avg_loss = total_loss / num_samples
        avg_distill_loss = total_distill_loss / num_samples
        avg_task_loss = total_task_loss / num_samples

        logger.info(f"Distillation Client {self.client_id}: Training completed. "
                    f"Total Loss: {avg_loss:.4f}, Distill Loss: {avg_distill_loss:.4f}, "
                    f"Task Loss: {avg_task_loss:.4f}, Total samples: {num_samples}")

        return self.get_parameters(config={}), num_samples, {
            "student_loss": avg_loss,
            "distill_loss": avg_distill_loss,
            "task_loss": avg_task_loss
        }

    def _compute_knowledge_distillation_loss(self,
                                             student_outputs: Dict[str, torch.Tensor],
                                             teacher_outputs: Dict[str, torch.Tensor],
                                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute knowledge distillation loss following DistilHuBERT approach"""

        # Get predictions from both models
        # [batch_size, seq_len, vocab_size]
        student_predictions = student_outputs['predictions']
        # [batch_size, seq_len, vocab_size]
        teacher_predictions = teacher_outputs['predictions']

        # Align dimensions if needed
        if student_predictions.shape != teacher_predictions.shape:
            min_seq_len = min(student_predictions.size(1),
                              teacher_predictions.size(1))
            student_predictions = student_predictions[:, :min_seq_len, :]
            teacher_predictions = teacher_predictions[:, :min_seq_len, :]

        # Flatten for loss computation
        student_flat = student_predictions.view(-1,
                                                student_predictions.size(-1))
        teacher_flat = teacher_predictions.view(-1,
                                                teacher_predictions.size(-1))

        # Apply temperature scaling
        student_logits = student_flat / self.temperature
        teacher_logits = teacher_flat / self.temperature

        # Compute KL divergence loss
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # KL divergence: KL(student || teacher)
        kl_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction='batchmean')

        # Scale by temperature squared (standard practice)
        distillation_loss = kl_loss * (self.temperature ** 2)

        return distillation_loss

    def _compute_masked_language_modeling_loss(self, outputs: Dict[str, torch.Tensor],
                                               input_values: torch.Tensor,
                                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute masked language modeling loss for student model"""

        # Get predictions and hidden states
        if 'predictions' in outputs:
            # [batch_size, seq_len, vocab_size]
            predictions = outputs['predictions']
        else:
            logger.error("No predictions found in model outputs")
            return torch.tensor(0.0, requires_grad=True, device=input_values.device)

        # Get hidden states for target generation
        if 'hidden_states' in outputs:
            # [batch_size, seq_len, hidden_size]
            hidden_states = outputs['hidden_states']
        else:
            logger.error("No hidden states found in model outputs")
            return torch.tensor(0.0, requires_grad=True, device=input_values.device)

        batch_size, seq_len, vocab_size = predictions.shape

        # Create random mask for tokens (similar to BERT masking)
        mask_prob = 0.15  # Standard masking probability
        mask = torch.rand(batch_size, seq_len,
                          device=predictions.device) < mask_prob

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure attention mask matches prediction sequence length
            if attention_mask.size(-1) != seq_len:
                # Interpolate or crop attention mask to match sequence length
                if attention_mask.size(-1) > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                else:
                    # Pad attention mask
                    padding = seq_len - attention_mask.size(-1)
                    attention_mask = F.pad(
                        attention_mask, (0, padding), value=0)

            # Apply attention mask to the random mask
            mask = mask & (attention_mask.bool())

        # Create targets from hidden states (quantized representations)
        with torch.no_grad():
            targets = self._create_discrete_targets(hidden_states, vocab_size)

        # Only compute loss on masked tokens
        if mask.any():
            # Flatten tensors
            # [batch_size * seq_len, vocab_size]
            flat_predictions = predictions.view(-1, vocab_size)
            flat_targets = targets.view(-1)  # [batch_size * seq_len]
            flat_mask = mask.view(-1)  # [batch_size * seq_len]

            # Select only masked positions
            # [num_masked, vocab_size]
            masked_predictions = flat_predictions[flat_mask]
            masked_targets = flat_targets[flat_mask]  # [num_masked]

            if masked_predictions.size(0) > 0:
                # Compute cross-entropy loss
                loss = F.cross_entropy(masked_predictions, masked_targets)
                return loss
            else:
                # No masked tokens, return small regularization loss
                return torch.tensor(0.01, requires_grad=True, device=predictions.device)
        else:
            # No valid mask, return regularization loss to prevent collapse
            # L2 regularization on predictions to encourage non-zero outputs
            l2_loss = torch.mean(predictions ** 2) * 0.01
            return l2_loss

    def _create_discrete_targets(self, hidden_states: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Create discrete targets from hidden states for self-supervised learning"""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Normalize hidden states
        normalized = F.layer_norm(hidden_states, [hidden_size])

        # Simple quantization: use mean pooling and hash to vocab indices
        # Pool across hidden dimension
        pooled = normalized.mean(dim=-1)  # [batch_size, seq_len]

        # Create targets by binning the pooled values
        # Normalize to [0, 1] range
        min_vals = pooled.min(dim=-1, keepdim=True)[0]
        max_vals = pooled.max(dim=-1, keepdim=True)[0]
        # Add small epsilon to avoid division by zero
        range_vals = max_vals - min_vals + 1e-8

        normalized_pooled = (pooled - min_vals) / range_vals

        # Convert to discrete targets
        targets = (normalized_pooled * (vocab_size - 1)).long()
        targets = torch.clamp(targets, 0, vocab_size - 1)

        return targets


class ServerSideDistillationStrategy(FedAvg):
    """
    Custom Flower strategy for federated knowledge distillation
    Uses standard FedAvg aggregation - distillation happens on clients
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

        logger.info(
            "Server strategy initialized for federated knowledge distillation")

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate client results using standard FedAvg"""
        logger.info(
            f"Round {server_round}: Aggregating {len(results)} client updates")

        # Check if any client has valid examples
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            logger.warning(
                f"Round {server_round}: All clients returned 0 examples, skipping aggregation")
            # Return the current parameters without aggregation
            if hasattr(self, '_current_parameters') and self._current_parameters is not None:
                return self._current_parameters, {}
            else:
                # If no current parameters, return None to signal failure
                return None, {}

        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        # Store current parameters for potential fallback
        if aggregated_parameters is not None:
            self._current_parameters = aggregated_parameters

        logger.info(
            f"Round {server_round}: Aggregation completed with {total_examples} total examples")

        return aggregated_parameters, aggregated_metrics


def client_fn(context: Context) -> Client:
    """Create a Flower client instance for distillation"""

    # Load configuration - try optimized first, fall back to original
    optimized_config_path = "configs/distillation_config_optimized.yaml"
    original_config_path = "configs/distillation_config.yaml"

    config_path = optimized_config_path if Path(
        optimized_config_path).exists() else original_config_path
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
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

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
        weighted_sum = sum(num_examples * m[key]
                           for num_examples, m in metrics)
        weighted_metrics[key] = weighted_sum / total_examples

    return weighted_metrics


def load_huggingface_hubert_teacher(teacher_model: HuBERTPretrainingModel, device: torch.device) -> bool:
    """
    Load pretrained HuBERT weights from HuggingFace hub into custom teacher model
    """
    try:
        logger.info("Loading pretrained HuBERT base from HuggingFace...")

        # Load the HuggingFace pretrained model
        hf_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        hf_state_dict = hf_model.state_dict()

        # Get our custom teacher model state dict
        teacher_state_dict = teacher_model.state_dict()

        # Mapping from HuggingFace HuBERT to our custom model
        weight_mapping = {
            # Feature projection
            'feature_projection.projection.weight': 'feature_extractor.feature_projection.weight',
            'feature_projection.projection.bias': 'feature_extractor.feature_projection.bias',

            # Layer norm
            'encoder.layer_norm.weight': 'encoder.layernorm.weight',
            'encoder.layer_norm.bias': 'encoder.layernorm.bias',
        }

        # Copy transformer layer weights
        for layer_idx in range(12):  # HuBERT base has 12 layers
            hf_prefix = f'encoder.layers.{layer_idx}'
            teacher_prefix = f'encoder.layers.{layer_idx}'

            # Attention weights
            weight_mapping.update({
                f'{hf_prefix}.attention.q_proj.weight': f'{teacher_prefix}.attention.query.weight',
                f'{hf_prefix}.attention.q_proj.bias': f'{teacher_prefix}.attention.query.bias',
                f'{hf_prefix}.attention.k_proj.weight': f'{teacher_prefix}.attention.key.weight',
                f'{hf_prefix}.attention.k_proj.bias': f'{teacher_prefix}.attention.key.bias',
                f'{hf_prefix}.attention.v_proj.weight': f'{teacher_prefix}.attention.value.weight',
                f'{hf_prefix}.attention.v_proj.bias': f'{teacher_prefix}.attention.value.bias',
                f'{hf_prefix}.attention.out_proj.weight': f'{teacher_prefix}.attention.output.weight',
                f'{hf_prefix}.attention.out_proj.bias': f'{teacher_prefix}.attention.output.bias',

                # Layer norms
                f'{hf_prefix}.layer_norm.weight': f'{teacher_prefix}.layernorm_before.weight',
                f'{hf_prefix}.layer_norm.bias': f'{teacher_prefix}.layernorm_before.bias',
                f'{hf_prefix}.final_layer_norm.weight': f'{teacher_prefix}.layernorm_after.weight',
                f'{hf_prefix}.final_layer_norm.bias': f'{teacher_prefix}.layernorm_after.bias',

                # Feed forward
                f'{hf_prefix}.feed_forward.intermediate_dense.weight': f'{teacher_prefix}.feed_forward.dense1.weight',
                f'{hf_prefix}.feed_forward.intermediate_dense.bias': f'{teacher_prefix}.feed_forward.dense1.bias',
                f'{hf_prefix}.feed_forward.output_dense.weight': f'{teacher_prefix}.feed_forward.dense2.weight',
                f'{hf_prefix}.feed_forward.output_dense.bias': f'{teacher_prefix}.feed_forward.dense2.bias',
            })

        # Transfer compatible weights
        loaded_weights = 0
        total_weights = 0

        for hf_key, teacher_key in weight_mapping.items():
            total_weights += 1

            if teacher_key is None:  # Skip incompatible layers
                continue

            if hf_key in hf_state_dict and teacher_key in teacher_state_dict:
                hf_param = hf_state_dict[hf_key]
                teacher_param = teacher_state_dict[teacher_key]

                # Check shape compatibility
                if hf_param.shape == teacher_param.shape:
                    teacher_state_dict[teacher_key] = hf_param.clone()
                    loaded_weights += 1
                    logger.debug(
                        f"Loaded: {hf_key} -> {teacher_key} {hf_param.shape}")
                else:
                    logger.debug(
                        f"Shape mismatch for {hf_key}: {hf_param.shape} vs {teacher_param.shape}")
            else:
                if hf_key in hf_state_dict:
                    logger.debug(f"Teacher key not found: {teacher_key}")
                else:
                    logger.debug(f"HF key not found: {hf_key}")
        
        # Note about feature extractor differences
        logger.info("Note: Feature extractor weights not loaded due to architectural differences.")
        logger.info("HuggingFace HuBERT has different conv layer structure than custom implementation.")

        # Load the updated state dict
        teacher_model.load_state_dict(teacher_state_dict, strict=False)

        logger.info(
            f"Successfully loaded {loaded_weights}/{total_weights} compatible weights from HuggingFace HuBERT base")
        logger.info("Teacher model initialized with pretrained HuBERT weights")

        return True

    except Exception as e:
        logger.error(f"Failed to load HuggingFace HuBERT weights: {e}")
        logger.warning("Using random teacher initialization instead")
        return False


def custom_collate_fn(batch):
    """Custom collate function to handle None values in the batch and within sample dictionaries."""
    import logging
    
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    # Log filtering statistics
    total_samples = len(batch)
    initial_valid_samples = len(valid_batch)
    filtered_samples = total_samples - initial_valid_samples
    
    if filtered_samples > 0:
        logging.info(f"Batch filtering: {initial_valid_samples}/{total_samples} valid samples, filtered out {filtered_samples} None items")

    if not valid_batch:
        # If all items are None, return an empty batch structure that won't cause errors
        logging.warning(f"All {total_samples} items in batch were None, returning empty batch structure")
        # Return a minimal batch structure that the training loop can detect and skip
        return {'empty_batch': True, 'valid_samples': 0, 'total_samples': total_samples}

    # Handle None values within dictionary keys
    # Collect all possible keys from valid samples
    all_keys = set()
    for item in valid_batch:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    # Create collated batch manually to handle None values within dictionaries
    collated_batch = {}
    none_key_counts = {}
    
    for key in all_keys:
        values = []
        none_count = 0
        
        for item in valid_batch:
            if isinstance(item, dict) and key in item:
                value = item[key]
                if value is not None:
                    values.append(value)
                else:
                    none_count += 1
            else:
                none_count += 1
        
        if none_count > 0:
            none_key_counts[key] = none_count
            
        # Only collate if we have valid values
        if values:
            try:
                from torch.utils.data._utils.collate import default_collate
                collated_batch[key] = default_collate(values)
            except Exception as e:
                logging.warning(f"Failed to collate key '{key}': {e}")
                # Skip this key or handle it differently
                if len(values) == 1:
                    collated_batch[key] = values[0]
                else:
                    logging.warning(f"Skipping key '{key}' due to collation failure")
        else:
            # All values for this key were None, skip it
            logging.warning(f"All values for key '{key}' were None, skipping")
    
    # Log None value statistics for keys
    if none_key_counts:
        for key, count in none_key_counts.items():
            if count > 0:
                logging.info(f"Key '{key}': {count}/{initial_valid_samples} values were None")
    
    # Add metadata about batch composition
    collated_batch['valid_samples'] = initial_valid_samples
    collated_batch['total_samples'] = total_samples
    
    return collated_batch


def server_fn(context: Context) -> ServerAppComponents:
    """Create server app components for distillation"""

    # Load configuration - try optimized first, fall back to original
    optimized_config_path = "configs/distillation_config_optimized.yaml"
    original_config_path = "configs/distillation_config.yaml"

    config_path = optimized_config_path if Path(
        optimized_config_path).exists() else original_config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Server using device: {device}")

    # Initialize teacher model (full HuBERT model)
    teacher_config = config['distillation']
    teacher_model = HuBERTPretrainingModel(
        vocab_size=int(teacher_config.get('vocab_size', 504)),
        hidden_size=int(teacher_config.get('teacher_hidden_size', 768)),
        num_hidden_layers=int(teacher_config.get('teacher_num_layers', 12)),
        num_attention_heads=int(teacher_config.get('teacher_num_heads', 12)),
        intermediate_size=int(teacher_config.get(
            'teacher_intermediate_size', 3072))
    ).to(device)

    # Try to load pretrained HuggingFace HuBERT weights
    use_pretrained = teacher_config.get('use_pretrained_teacher', True)
    if use_pretrained:
        success = load_huggingface_hubert_teacher(teacher_model, device)
        if not success:
            logger.warning("Continuing with random teacher initialization")
    else:
        logger.info(
            "Using random teacher initialization (use_pretrained_teacher=False)")

    # Alternative: Load from local checkpoint if available
    teacher_weights_path = teacher_config.get('teacher_weights_path', None)
    if teacher_weights_path and Path(teacher_weights_path).exists():
        try:
            checkpoint = torch.load(teacher_weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                teacher_model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False)
                logger.info(
                    f"Loaded teacher weights from local checkpoint: {teacher_weights_path}")
            else:
                logger.warning(
                    f"No model_state_dict found in {teacher_weights_path}")
        except Exception as e:
            logger.error(f"Failed to load local teacher weights: {e}")

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
    parser = argparse.ArgumentParser(
        description="Federated HuBERT Knowledge Distillation")
    parser.add_argument("--config", type=str,
                        default="configs/distillation_config.yaml")
    parser.add_argument("--simulation", action="store_true",
                        help="Run simulation")
    parser.add_argument("--num-clients", type=int,
                        help="Override number of clients")
    parser.add_argument("--num-rounds", type=int,
                        help="Override number of rounds")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip automatic resource optimization")

    args = parser.parse_args()

    # Optimize configuration for current hardware
    if not args.no_optimize:
        logger.info(
            "Detecting and optimizing for available hardware resources...")
        optimizer = ResourceOptimizer()
        optimizer.print_resource_summary()

        # Create optimized config
        optimized_config_path = args.config.replace('.yaml', '_optimized.yaml')
        optimizer.save_optimized_config(
            args.config, optimized_config_path, args.num_clients)
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
                logger.info(
                    f"Distillation completed after {total_rounds} rounds")

        except Exception as e:
            logger.error(f"Distillation simulation failed: {e}")
            raise

    else:
        logger.info(
            "Non-simulation mode not implemented. Use --simulation flag.")


if __name__ == "__main__":
    main()
