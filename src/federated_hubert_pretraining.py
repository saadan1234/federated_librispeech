#!/usr/bin/env python3

import os
import json
import yaml
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

import pandas as pd
import numpy as np
from tqdm import tqdm

from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAdam, FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.simulation import run_simulation
from flwr.common import Context, Metrics, NDArrays, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Config

from transformers import HubertConfig, Wav2Vec2FeatureExtractor
import soundfile as sf

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/federated_pretraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dataset for LibriSpeech pretraining, loading audio files and applying feature extraction.
class LibriSpeechPretrainingDataset(Dataset):
    
    def __init__(
        self,
        manifest_file: str,
        audio_root: str,
        feature_extractor: Wav2Vec2FeatureExtractor,
        max_length: int = 160000,
        sample_rate: int = 16000,
        mask_prob: float = 0.08,
        mask_length: int = 10
    ):
        self.manifest_df = pd.read_csv(manifest_file)
        self.audio_root = Path(audio_root)
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        
        self.resampler = T.Resample(orig_freq=16000, new_freq=sample_rate) if sample_rate != 16000 else None
    
    def __len__(self) -> int:
        return len(self.manifest_df)
    
    # Apply masking to the features based on the specified mask probability and length.
    def _apply_masking(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = features.shape[:2]
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        
        for i in range(batch_size):
            num_masked = int(seq_len * self.mask_prob)
            max_start = max(1, seq_len - self.mask_length)
            mask_starts = torch.randint(0, max_start, (num_masked,))
            
            for start in mask_starts:
                end = min(start + self.mask_length, seq_len)
                mask[i, start:end] = True
        
        masked_features = features.clone()
        masked_features[mask] = 0.0
        
        return masked_features, mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.manifest_df)
                row = self.manifest_df.iloc[current_idx]
                
                audio_path = self.audio_root / row['audio_path']
                audio, _ = sf.read(str(audio_path))
                audio = torch.tensor(audio, dtype=torch.float32)
                
                if self.resampler is not None:
                    audio = self.resampler(audio)
                
                if len(audio) > self.max_length:
                    start = torch.randint(0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    padding = self.max_length - len(audio)
                    audio = torch.nn.functional.pad(audio, (0, padding))
                
                inputs = self.feature_extractor(
                    audio.numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length
                )
                
                input_values = inputs['input_values'].squeeze(0)
                attention_mask = inputs.get('attention_mask', torch.ones_like(input_values)).squeeze(0)
                
                return {
                    'input_values': input_values,
                    'attention_mask': attention_mask,
                    'audio_path': str(audio_path)
                }
                
            except Exception:
                continue
        
        raise RuntimeError(f"Failed to load any valid audio sample after {max_retries} attempts starting from index {idx}")

# HuBERT convolutional feature encoder with 7 layers as specified in research paper.
# First layer: kernel_size=10, stride=5; following layers: smaller kernels, stride=2
class HuBERTFeatureEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Seven convolutional layers as specified in research paper
        conv_layers = [
            # Layer 1: First layer uses kernel_size=10, stride=5 as specified
            nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 2: Smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 3: Smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 4: Smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 5: Smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 6: Smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            
            # Layer 7: Final layer with smaller kernel, stride=2 as specified
            nn.Conv1d(512, 512, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
        ]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        # Project to 768-dimensional space as specified in paper
        self.feature_projection = nn.Linear(512, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_extract_dropout)
    
    # Forward pass through the feature encoder, applying convolutional layers and projection.
    def forward(self, input_values):
        hidden_states = input_values.unsqueeze(1)
        hidden_states = self.conv_layers(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# HuBERT transformer encoder that processes the hidden states from the feature encoder.
class HuBERTTransformerEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            HuBERTTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

# HuBERT transformer layer that includes attention and feed-forward components.
class HuBERTTransformerLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.attention = HuBERTAttention(config)
        self.feed_forward = HuBERTFeedForward(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# HuBERT attention mechanism with multi-head attention.
class HuBERTAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    # Forward pass for the attention mechanism.
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            mask_value = -65504.0 if attention_scores.dtype == torch.float16 else -1e4
            attention_scores = attention_scores.masked_fill(attention_mask == 0, mask_value)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        
        return context

# HuBERT feed-forward layer with GELU activation and dropout.
class HuBERTFeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = F.gelu
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states

# HuBERT pretraining model that combines feature extraction, transformer encoding, and prediction head.
class HuBERTPretrainingModel(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 504,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        mask_prob: float = 0.08,
        mask_length: int = 10
    ):
        super().__init__()
        
        self.config = HubertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            feat_extract_dropout=0.0,
            layer_norm_eps=1e-5,
            mask_time_prob=0.08,
            mask_time_length=mask_length,
            max_position_embeddings=1024,
        )
        
        self.feature_extractor = HuBERTFeatureEncoder(self.config)
        self.encoder = HuBERTTransformerEncoder(self.config)
        self.prediction_head = nn.Linear(hidden_size, vocab_size)
        
        self.apply(self._init_weights)
        
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        
        # Initialize mask token parameter to ensure consistent model structure
        self.mask_token = nn.Parameter(torch.randn(hidden_size) * 0.02)

    # Initialize weights for the model components.
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

    # Create discrete targets for the model based on the hidden states.
    def _create_discrete_targets(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        hidden_size = hidden_states.shape[-1]
        normalized = F.layer_norm(hidden_states, [hidden_size])
        
        projection_dim = min(128, hidden_size // 4)
        projected = F.linear(normalized, 
                           torch.randn(projection_dim, hidden_size, device=hidden_states.device) * 0.02)
        
        features_1 = projected[:, :, :projection_dim//2].mean(dim=-1)
        features_2 = projected[:, :, projection_dim//2:].mean(dim=-1)
        
        quantized_1 = ((features_1 - features_1.min()) / (features_1.max() - features_1.min() + 1e-8) * (self.config.vocab_size // 2)).long()
        quantized_2 = ((features_2 - features_2.min()) / (features_2.max() - features_2.min() + 1e-8) * (self.config.vocab_size // 2)).long()
        
        targets = (quantized_1 + quantized_2) % self.config.vocab_size
        
        return targets

    # Apply masking to the hidden states based on the specified mask probability and length.
    def _apply_masking(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
        
        # Vectorized masking for speed
        valid_lengths = seq_len if attention_mask is None else attention_mask.sum(dim=1)
        
        for i in range(batch_size):
            valid_length = valid_lengths if attention_mask is None else int(valid_lengths[i].item())
            
            if valid_length <= self.mask_length:
                continue
                
            # Apply span masking as specified in research paper
            total_to_mask = max(1, int(valid_length * self.mask_prob))
            num_spans = max(1, total_to_mask // self.mask_length)
            
            for _ in range(num_spans):
                if valid_length > self.mask_length:
                    start_pos = torch.randint(0, valid_length - self.mask_length, (1,)).item()
                    end_pos = start_pos + self.mask_length
                    mask[i, start_pos:end_pos] = True
        
        return mask
 
    # Forward pass through the model, including feature extraction, masking, and prediction.
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        hidden_states = self.feature_extractor(input_values)
        
        if attention_mask is not None:
            seq_reduction_factor = 320
            reduced_length = hidden_states.size(1)
            attention_mask = attention_mask[:, ::seq_reduction_factor][:, :reduced_length]
        
        with torch.no_grad():
            targets = self._create_discrete_targets(hidden_states)
        
        mask = self._apply_masking(hidden_states, attention_mask)
        
        masked_hidden_states = hidden_states.clone()
        if mask.any():
            mask_token_expanded = self.mask_token.to(hidden_states.device).unsqueeze(0).expand_as(hidden_states)
            masked_hidden_states = torch.where(mask.unsqueeze(-1), mask_token_expanded, hidden_states)
        
        encoder_outputs = self.encoder(masked_hidden_states, attention_mask)
        predictions = self.prediction_head(encoder_outputs)
        
        loss = None
        if mask.any():
            flat_predictions = predictions.view(-1, self.config.vocab_size)
            flat_targets = targets.view(-1)
            flat_mask = mask.view(-1)
            
            masked_predictions = flat_predictions[flat_mask]
            masked_targets = flat_targets[flat_mask]
            
            if masked_predictions.size(0) > 0:
                loss = F.cross_entropy(masked_predictions, masked_targets)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'hidden_states': encoder_outputs,
            'targets': targets,
            'mask': mask
        }

# Federated learning client for HuBERT pretraining.
class FederatedHuBERTPretrainingClient(NumPyClient):
    
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
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.feature_extractor = None
        self.model = None
        self.train_loader = None
        
        self.local_epochs = int(config['pretraining']['local_epochs'])
        self.batch_size = int(config['pretraining']['batch_size'])
        self.learning_rate = float(config['pretraining']['learning_rate'])
        self.max_audio_length = int(config['pretraining']['max_audio_length'])
        
        # Checkpoint configuration
        self.checkpoint_dir = Path(config['pretraining'].get('checkpoint_dir', 'checkpoints'))
        self.save_checkpoints = config['pretraining'].get('save_checkpoints', True)
        self.checkpoint_freq = int(config['pretraining'].get('checkpoint_freq', 1))
        self.keep_last_n_checkpoints = int(config['pretraining'].get('keep_last_n_checkpoints', 3))
        
        # Create checkpoint directory
        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.client_checkpoint_dir = self.checkpoint_dir / f"client_{self.client_id}"
            self.client_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.current_round = 0
        
        self._setup_feature_extractor()
        self._setup_data_loader()
        self._setup_model()
        
    # Setup the feature extractor for audio preprocessing. (Not for feature extraction) 
    def _setup_feature_extractor(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
    def _setup_data_loader(self):
        train_manifest = self.data_path / "manifest.csv"
        if not train_manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {train_manifest}")
            
        manifest_df = pd.read_csv(train_manifest)
        train_df = manifest_df[manifest_df['split'] == 'train']
        
        # Limit dataset size for faster training
        if len(train_df) > 2000:
            train_df = train_df.sample(n=2000, random_state=42)
            logger.info(f"Limited training data to 2000 samples for client {self.client_id}")
        
        train_manifest_path = self.data_path / "pretrain_manifest.csv"
        train_df.to_csv(train_manifest_path, index=False)
        
        train_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(train_manifest_path),
            audio_root=str(self.data_path),
            feature_extractor=self.feature_extractor,
            max_length=self.max_audio_length,
            mask_prob=float(self.config['pretraining']['mask_prob']),
            mask_length=int(self.config['pretraining']['mask_length'])
        )
        
        # Optimized number of workers
        num_workers = min(4, os.cpu_count() // 2)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=3 if num_workers > 0 else 3,
            drop_last=True  # Ensure consistent batch sizes
        )
    
    def _setup_model(self):
        model_config = self.config['pretraining']
        
        self.model = HuBERTPretrainingModel(
            vocab_size=int(model_config.get('vocab_size', 504)),
            hidden_size=int(model_config.get('hidden_size', 768)),
            num_hidden_layers=int(model_config.get('num_hidden_layers', 12)),
            num_attention_heads=int(model_config.get('num_attention_heads', 12)),
            intermediate_size=int(model_config.get('intermediate_size', 3072)),
            mask_prob=float(model_config.get('mask_prob', 0.05)),
            mask_length=int(model_config.get('mask_length', 10))
        ).to(self.device)
    
    def save_checkpoint(self, loss: float, is_best: bool = False, round_num: int = None):
        """Save model checkpoint with training state."""
        if not self.save_checkpoints:
            return
        
        round_num = round_num or self.current_round
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'client_id': self.client_id,
            'config': self.config,
            'model_config': {
                'vocab_size': self.model.config.vocab_size,
                'hidden_size': self.model.config.hidden_size,
                'num_hidden_layers': self.model.config.num_hidden_layers,
                'num_attention_heads': self.model.config.num_attention_heads,
                'intermediate_size': self.model.config.intermediate_size,
                'mask_prob': self.model.mask_prob,
                'mask_length': self.model.mask_length
            }
        }
        
        # Save latest checkpoint
        latest_path = self.client_checkpoint_dir / f"latest_round_{round_num}.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved latest checkpoint for client {self.client_id} at round {round_num}")
        
        # Save best checkpoint if this is the best loss so far
        if is_best:
            best_path = self.client_checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint for client {self.client_id} with loss {loss:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load model checkpoint and restore training state."""
        if checkpoint_path is None:
            # Try to load the latest checkpoint
            checkpoints = list(self.client_checkpoint_dir.glob("latest_round_*.pt"))
            if not checkpoints:
                logger.info(f"No checkpoints found for client {self.client_id}")
                return False
            
            # Get the latest checkpoint by round number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_path = str(latest_checkpoint)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore training state
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.current_round = checkpoint.get('round', 0)
            
            logger.info(f"Loaded checkpoint for client {self.client_id} from round {self.current_round}")
            logger.info(f"Best loss: {self.best_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.client_checkpoint_dir.glob("latest_round_*.pt"))
        
        if len(checkpoints) > self.keep_last_n_checkpoints:
            # Sort by round number and keep only the latest N
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            checkpoints_to_remove = checkpoints[:-self.keep_last_n_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    # Get model parameters as NDArrays for federated learning.
    def get_parameters(self, config: Config) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    # Set model parameters from the received NDArrays.
    def set_parameters(self, parameters: NDArrays) -> None:
        if not parameters:
            return
            
        current_state = self.model.state_dict()
        param_keys = list(current_state.keys())
        
        state_dict = OrderedDict()
        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            if i >= len(parameters):
                break
                
            param_tensor = torch.tensor(param_array)
            current_shape = current_state[key].shape
            
            if param_tensor.shape == current_shape:
                state_dict[key] = param_tensor
            else:
                state_dict[key] = current_state[key]
        
        self.model.load_state_dict(state_dict, strict=False)
    
    # Fit the model on the local dataset and return updated parameters, number of samples, and metrics.
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, float]]:
        self.set_parameters(parameters)
        
        # Update current round number
        self.current_round = config.get('server_round', self.current_round + 1)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.config['pretraining'].get('weight_decay', 0.01)),
            eps=1e-8,  # Better numerical stability
            betas=(0.9, 0.999)
        )
        
        # Enable mixed precision training if available
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        
        # Use a linear learning rate scheduler for better convergence.
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=int(self.local_epochs * len(self.train_loader))
        )
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.90)  # Leave more memory for other processes
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch in tqdm(self.train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}"):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                optimizer.zero_grad()
                
                # Use mixed precision if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_values=input_values,
                            attention_mask=attention_mask
                        )
                        loss = outputs['loss']
                    
                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        
                        batch_size = input_values.size(0)
                        epoch_loss += loss.item() * batch_size
                        epoch_samples += batch_size
                else:
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    loss = outputs['loss']
                    
                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        batch_size = input_values.size(0)
                        epoch_loss += loss.item() * batch_size
                        epoch_samples += batch_size
                
            if epoch_samples > 0:
                total_loss += epoch_loss
                num_samples += epoch_samples
                
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Ensure we always return a positive number of samples
        if num_samples == 0:
            logger.warning(f"Client {self.client_id}: No valid samples processed during training")
            # Return dummy values to avoid division by zero
            return self.get_parameters(config={}), 1, {"pretrain_loss": 0.0}
        
        avg_loss = total_loss / num_samples
        
        # Save checkpoint if enabled
        if self.save_checkpoints and self.current_round % self.checkpoint_freq == 0:
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            self.save_checkpoint(
                loss=avg_loss,
                is_best=is_best,
                round_num=self.current_round
            )
        
        return self.get_parameters(config={}), num_samples, {"pretrain_loss": avg_loss}

# Main client function to initialize the federated learning client.
def client_fn(context: Context) -> Client:
    optimized_config_path = "configs/pretraining_config_optimized.yaml"
    original_config_path = "configs/pretraining_config.yaml"
    
    # Prefer optimized config
    config_path = optimized_config_path if Path(optimized_config_path).exists() else original_config_path
    logger.info(f"Client using config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine the client ID based on the node ID and number of supernodes for assigning data partitions
    node_id = context.node_id
    num_clients = int(config['simulation']['num_supernodes'])
    client_id = hash(str(node_id)) % num_clients
    
    data_root_config = config['data']['partitioned_data_root']
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)
    
    client_data_path = data_root / f"client_{client_id}"
    
    if not client_data_path.exists():
        raise FileNotFoundError(f"Client data directory not found: {client_data_path}")
    
    manifest_path = client_data_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    client = FederatedHuBERTPretrainingClient(
        client_id=client_id,
        data_path=str(client_data_path),
        config=config
    )
    
    # Try to load existing checkpoint if resume is enabled
    if config['pretraining'].get('resume_from_checkpoint', False):
        client.load_checkpoint()
    
    return client

# Function to aggregate metrics from multiple clients, rather then using simple averaging.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        weighted_sum = sum(num_examples * m[key] for num_examples, m in metrics)
        weighted_metrics[key] = weighted_sum / total_examples
    
    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:
    optimized_config_path = "configs/pretraining_config_optimized.yaml"
    original_config_path = "configs/pretraining_config.yaml"
    
    config_path = optimized_config_path if Path(optimized_config_path).exists() else original_config_path
    
# Server-side checkpoint management
class ServerCheckpointManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = Path(config['pretraining'].get('checkpoint_dir', 'checkpoints'))
        self.server_checkpoint_dir = self.checkpoint_dir / "server"
        self.server_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_checkpoints = config['pretraining'].get('save_checkpoints', True)
        self.checkpoint_freq = int(config['pretraining'].get('checkpoint_freq', 1))
        self.keep_last_n_checkpoints = int(config['pretraining'].get('keep_last_n_checkpoints', 3))
        
        self.best_loss = float('inf')
        self.current_round = 0
        
    def save_global_model(self, parameters: NDArrays, metrics: Dict[str, float], round_num: int):
        """Save the global model parameters and metrics."""
        if not self.save_checkpoints or round_num % self.checkpoint_freq != 0:
            return
        
        self.current_round = round_num
        current_loss = metrics.get('pretrain_loss', float('inf'))
        
        # Create a dummy model to save the architecture
        dummy_model = HuBERTPretrainingModel(
            vocab_size=int(self.config['pretraining'].get('vocab_size', 504)),
            hidden_size=int(self.config['pretraining'].get('hidden_size', 768)),
            num_hidden_layers=int(self.config['pretraining'].get('num_hidden_layers', 12)),
            num_attention_heads=int(self.config['pretraining'].get('num_attention_heads', 12)),
            intermediate_size=int(self.config['pretraining'].get('intermediate_size', 3072)),
            mask_prob=float(self.config['pretraining'].get('mask_prob', 0.05)),
            mask_length=int(self.config['pretraining'].get('mask_length', 10))
        )
        
        # Convert parameters back to state dict
        state_dict = OrderedDict()
        param_keys = list(dummy_model.state_dict().keys())
        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            if i < len(parameters):
                state_dict[key] = torch.tensor(param_array)
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': state_dict,
            'global_metrics': metrics,
            'best_loss': self.best_loss,
            'config': self.config,
            'model_config': {
                'vocab_size': dummy_model.config.vocab_size,
                'hidden_size': dummy_model.config.hidden_size,
                'num_hidden_layers': dummy_model.config.num_hidden_layers,
                'num_attention_heads': dummy_model.config.num_attention_heads,
                'intermediate_size': dummy_model.config.intermediate_size,
                'mask_prob': dummy_model.mask_prob,
                'mask_length': dummy_model.mask_length
            }
        }
        
        # Save latest checkpoint
        latest_path = self.server_checkpoint_dir / f"global_model_round_{round_num}.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved global model checkpoint at round {round_num}")
        
        # Save best checkpoint if this is the best loss so far
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss
            best_path = self.server_checkpoint_dir / "best_global_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best global model checkpoint with loss {current_loss:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def load_global_model(self, checkpoint_path: str = None):
        """Load global model checkpoint."""
        if checkpoint_path is None:
            # Try to load the latest checkpoint
            checkpoints = list(self.server_checkpoint_dir.glob("global_model_round_*.pt"))
            if not checkpoints:
                logger.info("No global model checkpoints found")
                return None
            
            # Get the latest checkpoint by round number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_path = str(latest_checkpoint)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract parameters as NDArrays and convert to Parameters object
            parameters_ndarrays = [param.numpy() for param in checkpoint['model_state_dict'].values()]
            parameters = ndarrays_to_parameters(parameters_ndarrays)
            
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.current_round = checkpoint.get('round', 0)
            
            logger.info(f"Loaded global model checkpoint from round {self.current_round}")
            logger.info(f"Best global loss: {self.best_loss:.4f}")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to load global model checkpoint {checkpoint_path}: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old global model checkpoints to save disk space."""
        checkpoints = list(self.server_checkpoint_dir.glob("global_model_round_*.pt"))
        
        if len(checkpoints) > self.keep_last_n_checkpoints:
            # Sort by round number and keep only the latest N
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            checkpoints_to_remove = checkpoints[:-self.keep_last_n_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old global checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove global checkpoint {checkpoint}: {e}")

# Custom FedAvg strategy with checkpoint management
class FedAvgWithCheckpoints(FedAvg):
    
    def __init__(self, checkpoint_manager: ServerCheckpointManager, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_manager = checkpoint_manager
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | Tuple[ClientProxy, Exception]]):
        """Aggregate fit results and save checkpoints."""
        # Call parent aggregate_fit method
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            aggregated_parameters, aggregated_metrics = aggregated_result
            
            # Convert parameters to NDArrays for checkpoint saving
            parameters_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Save global model checkpoint
            self.checkpoint_manager.save_global_model(
                parameters=parameters_ndarrays,
                metrics=aggregated_metrics,
                round_num=server_round
            )
        
        return aggregated_result

# Custom FedAdam strategy with checkpoint management
class FedAdamWithCheckpoints(FedAdam):
    
    def __init__(self, checkpoint_manager: ServerCheckpointManager, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_manager = checkpoint_manager
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | Tuple[ClientProxy, Exception]]):
        """Aggregate fit results and save checkpoints."""
        # Call parent aggregate_fit method
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            aggregated_parameters, aggregated_metrics = aggregated_result
            
            # Convert parameters to NDArrays for checkpoint saving
            parameters_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Save global model checkpoint
            self.checkpoint_manager.save_global_model(
                parameters=parameters_ndarrays,
                metrics=aggregated_metrics,
                round_num=server_round
            )
        
        return aggregated_result

# Main server function to initialize the federated learning strategy and server configuration
def server_fn(context: Context) -> ServerAppComponents:
    optimized_config_path = "configs/pretraining_config_optimized.yaml"
    original_config_path = "configs/pretraining_config.yaml"
    
    # Prefer optimized config
    config_path = optimized_config_path if Path(optimized_config_path).exists() else original_config_path
    logger.info(f"Server using config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategy_config = config['strategy']
    pretraining_config = config['pretraining']
    
    # Initialize checkpoint manager
    checkpoint_manager = ServerCheckpointManager(config)
    
    # Try to load existing checkpoint if resume is enabled
    initial_parameters = None
    if config['pretraining'].get('resume_from_checkpoint', False):
        initial_parameters = checkpoint_manager.load_global_model()
    
    # Build strategy arguments
    strategy_args = {
        'checkpoint_manager': checkpoint_manager,
        'fraction_fit': float(strategy_config['fraction_fit']),
        'fraction_evaluate': float(strategy_config['fraction_evaluate']),
        'min_fit_clients': int(strategy_config['min_fit_clients']),
        'min_evaluate_clients': int(strategy_config['min_evaluate_clients']),
        'min_available_clients': int(strategy_config['min_available_clients']),
        'evaluate_metrics_aggregation_fn': weighted_average,
        'fit_metrics_aggregation_fn': weighted_average,
    }
    
    # FedAdam requires initial_parameters - use checkpoint if available, otherwise create random initialization
    if initial_parameters is None:
        # Create a model with random initialization for first round
        dummy_model = HuBERTPretrainingModel(
            vocab_size=int(pretraining_config.get('vocab_size', 504)),
            hidden_size=int(pretraining_config.get('hidden_size', 768)),
            num_hidden_layers=int(pretraining_config.get('num_hidden_layers', 12)),
            num_attention_heads=int(pretraining_config.get('num_attention_heads', 12)),
            intermediate_size=int(pretraining_config.get('intermediate_size', 3072)),
            mask_prob=float(pretraining_config.get('mask_prob', 0.05)),
            mask_length=int(pretraining_config.get('mask_length', 10))
        )
        initial_parameters_ndarrays = [param.cpu().numpy() for param in dummy_model.state_dict().values()]
        initial_parameters = ndarrays_to_parameters(initial_parameters_ndarrays)
    
    strategy_args['initial_parameters'] = initial_parameters
    
    strategy = FedAdamWithCheckpoints(**strategy_args)
    
    server_config = ServerConfig(
        num_rounds=int(pretraining_config['num_rounds'])
    )
    
    return ServerAppComponents(strategy=strategy, config=server_config)

def main():
    parser = argparse.ArgumentParser(description="Federated HuBERT Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretraining_config.yaml")
    parser.add_argument("--simulation", action="store_true", help="Run simulation")
    parser.add_argument("--num-clients", type=int, help="Override number of clients")
    parser.add_argument("--num-rounds", type=int, help="Override number of rounds")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.num_clients:
        config['simulation']['num_supernodes'] = int(args.num_clients)
    if args.num_rounds:
        config['pretraining']['num_rounds'] = int(args.num_rounds)
    
    if args.simulation:
        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(server_fn=server_fn)
        
        backend_config = config['simulation']['backend']['config']
        
        # Main simulation loop with specific configurations from federated config file
        history = run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=int(config['simulation']['num_supernodes']),
            backend_config=backend_config,
        )
        
        logger.info("Pretraining simulation completed!")
    else:
        logger.info("Non-simulation mode not implemented. Use --simulation flag.")

if __name__ == "__main__":
    main()