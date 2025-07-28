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
from flwr.common.typing import Config, Scalar

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


def generate_kmeans_targets(features: np.ndarray, n_clusters: int = 504, random_state: int = 42) -> np.ndarray:
    """
    Generate k-means cluster assignments for HuBERT pretraining targets.

    Args:
        features: Feature array of shape (num_samples, seq_len, feature_dim)
        n_clusters: Number of clusters (vocab_size)
        random_state: Random seed for reproducibility

    Returns:
        Cluster assignments of shape (num_samples, seq_len)
    """
    from sklearn.cluster import KMeans

    # Reshape features to (num_samples * seq_len, feature_dim)
    original_shape = features.shape
    features_flat = features.reshape(-1, features.shape[-1])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    cluster_assignments = kmeans.fit_predict(features_flat)

    # Reshape back to (num_samples, seq_len)
    cluster_assignments = cluster_assignments.reshape(original_shape[:2])

    return cluster_assignments


def align_targets_to_features(targets: np.ndarray, feature_seq_len: int) -> np.ndarray:
    """
    Align k-means targets to the feature encoder output sequence length.

    Args:
        targets: Original targets of shape (num_samples, original_seq_len)
        feature_seq_len: Target sequence length after feature extraction

    Returns:
        Aligned targets of shape (num_samples, feature_seq_len)
    """
    num_samples, original_seq_len = targets.shape

    # Calculate the sequence reduction factor
    seq_reduction_factor = original_seq_len // feature_seq_len

    # Downsample targets to match feature sequence length
    aligned_targets = np.zeros(
        (num_samples, feature_seq_len), dtype=targets.dtype)

    for i in range(num_samples):
        for j in range(feature_seq_len):
            # Take the most common target in the corresponding window
            start_idx = j * seq_reduction_factor
            end_idx = min((j + 1) * seq_reduction_factor, original_seq_len)
            window_targets = targets[i, start_idx:end_idx]

            if len(window_targets) > 0:
                # Use mode (most common value) for the window
                from scipy.stats import mode
                aligned_targets[i, j] = mode(window_targets, keepdims=False)[0]

    return aligned_targets

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
        mask_length: int = 10,
        kmeans_targets_file: str = None,  # NEW: path to k-means cluster assignments
        feature_seq_len: int = None,  # NEW: target sequence length after feature extraction
        auto_generate_kmeans: bool = False,  # NEW: auto-generate k-means if not provided
        vocab_size: int = 504  # NEW: number of clusters for k-means
    ):
        import time
        import os

        # Robust manifest file reading with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if file exists and has content
                if not os.path.exists(manifest_file):
                    raise FileNotFoundError(
                        f"Manifest file not found: {manifest_file}")

                if os.path.getsize(manifest_file) == 0:
                    raise ValueError(
                        f"Manifest file is empty: {manifest_file}")

                # Try to read the CSV file
                self.manifest_df = pd.read_csv(manifest_file)

                # Validate the DataFrame
                if self.manifest_df.empty:
                    raise ValueError(
                        f"Manifest file has no data: {manifest_file}")

                # Check for required columns
                required_cols = ['audio_path', 'duration']
                missing_cols = [
                    col for col in required_cols if col not in self.manifest_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns in manifest: {missing_cols}")

                break  # Success, exit retry loop

            except (pd.errors.EmptyDataError, ValueError, FileNotFoundError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise RuntimeError(
                        f"Failed to read manifest file after {max_retries} attempts: {str(e)}")
                else:
                    print(
                        f"Attempt {attempt + 1} failed to read manifest file: {str(e)}. Retrying...")
                    time.sleep(0.5)  # Wait before retry

        self.audio_root = Path(audio_root)
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.feature_seq_len = feature_seq_len
        self.auto_generate_kmeans = auto_generate_kmeans
        self.vocab_size = vocab_size

        self.resampler = T.Resample(
            orig_freq=16000, new_freq=sample_rate) if sample_rate != 16000 else None

        # Load or generate k-means targets
        self.kmeans_targets = None
        if kmeans_targets_file is not None:
            import numpy as np
            logger.info(f"Loading k-means targets from {kmeans_targets_file}")
            self.kmeans_targets = np.load(kmeans_targets_file)

            # Align targets to feature sequence length if needed
            if self.feature_seq_len is not None and len(self.kmeans_targets.shape) == 2:
                original_seq_len = self.kmeans_targets.shape[1]
                if original_seq_len != self.feature_seq_len:
                    logger.info(
                        f"Aligning targets from {original_seq_len} to {self.feature_seq_len} sequence length")
                    self.kmeans_targets = align_targets_to_features(
                        self.kmeans_targets, self.feature_seq_len)

            logger.info(
                f"Loaded k-means targets with shape: {self.kmeans_targets.shape}")

        elif auto_generate_kmeans:
            logger.info("Auto-generating k-means targets...")
            self.kmeans_targets = self._generate_kmeans_targets()
            logger.info(
                f"Generated k-means targets with shape: {self.kmeans_targets.shape}")

    def _generate_kmeans_targets(self):
        """Auto-generate k-means targets from the dataset."""
        import numpy as np
        from sklearn.cluster import KMeans

        logger.info("Extracting features for k-means clustering...")

        # Create a temporary feature encoder to extract features
        temp_config = HubertConfig(
            hidden_size=768,
            feat_extract_dropout=0.0
        )
        temp_feature_encoder = HuBERTFeatureEncoder(temp_config)

        # Extract features from a subset of the dataset for clustering
        max_samples_for_clustering = min(
            1000, len(self.manifest_df))  # Use max 1000 samples
        features_list = []

        for i in tqdm(range(max_samples_for_clustering), desc="Extracting features for clustering"):
            try:
                row = self.manifest_df.iloc[i]
                audio_path = self.audio_root / row['audio_path']
                audio, _ = sf.read(str(audio_path))
                audio = torch.tensor(audio, dtype=torch.float32)

                if self.resampler is not None:
                    audio = self.resampler(audio)

                if len(audio) > self.max_length:
                    start = torch.randint(
                        0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    padding = self.max_length - len(audio)
                    audio = torch.nn.functional.pad(audio, (0, padding))

                # Extract features
                with torch.no_grad():
                    features = temp_feature_encoder(
                        audio.unsqueeze(0))  # Add batch dimension
                    features_list.append(features.squeeze(0).numpy())

            except Exception as e:
                logger.warning(
                    f"Failed to extract features from sample {i}: {e}")
                continue

        if not features_list:
            raise RuntimeError(
                "Failed to extract any features for k-means clustering")

        # Stack features
        # Shape: (num_samples, seq_len, 768)
        features_array = np.stack(features_list)
        logger.info(f"Extracted features with shape: {features_array.shape}")

        # Generate k-means targets
        targets = generate_kmeans_targets(
            features_array, n_clusters=self.vocab_size)

        # Align to feature sequence length if needed
        if self.feature_seq_len is not None:
            original_seq_len = targets.shape[1]
            if original_seq_len != self.feature_seq_len:
                logger.info(
                    f"Aligning generated targets from {original_seq_len} to {self.feature_seq_len}")
                targets = align_targets_to_features(
                    targets, self.feature_seq_len)

        return targets

    def __len__(self) -> int:
        return len(self.manifest_df)

    def _span_mask(self, seq_len: int) -> torch.Tensor:
        # Official HuBERT: non-overlapping span masking
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
        failed_indices = []

        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.manifest_df)
                row = self.manifest_df.iloc[current_idx]

                audio_path = self.audio_root / row['audio_path']

                # Add detailed error checking
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    failed_indices.append(str(audio_path))
                    continue

                if not audio_path.is_file():
                    logger.warning(f"Audio path is not a file: {audio_path}")
                    failed_indices.append(str(audio_path))
                    continue

                # Check file size
                if audio_path.stat().st_size == 0:
                    logger.warning(f"Audio file is empty: {audio_path}")
                    failed_indices.append(str(audio_path))
                    continue

                audio, _ = sf.read(str(audio_path))
                audio = torch.tensor(audio, dtype=torch.float32)

                if self.resampler is not None:
                    audio = self.resampler(audio)

                if len(audio) > self.max_length:
                    start = torch.randint(
                        0, len(audio) - self.max_length + 1, (1,)).item()
                    audio = audio[start:start + self.max_length]
                else:
                    padding = self.max_length - len(audio)
                    audio = torch.nn.functional.pad(audio, (0, padding))

                # Only pad/truncate, do not normalize
                inputs = self.feature_extractor(
                    audio.numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length
                )

                input_values = inputs['input_values'].squeeze(0)
                attention_mask = inputs.get(
                    'attention_mask', torch.ones_like(input_values)).squeeze(0)

                # K-means cluster target
                target = None
                if self.kmeans_targets is not None:
                    if len(self.kmeans_targets.shape) == 2:
                        # (num_samples, seq_len) - use current sample
                        target = torch.tensor(
                            self.kmeans_targets[current_idx], dtype=torch.long)
                    else:
                        # (num_samples,) - use current sample
                        target = torch.tensor(
                            self.kmeans_targets[current_idx], dtype=torch.long)

                return {
                    'input_values': input_values,
                    'attention_mask': attention_mask,
                    'audio_path': str(audio_path),
                    'target': target
                }

            except Exception as e:
                logger.warning(f"Error processing sample {current_idx}: {e}")
                failed_indices.append(
                    str(audio_path) if 'audio_path' in locals() else f"index_{current_idx}")
                continue

        # If all retries failed, return a dummy sample instead of None
        logger.warning(
            f"All retries failed for index {idx}. Failed paths: {failed_indices[:5]}...")

        # Return a dummy sample to prevent None values
        dummy_audio = torch.zeros(self.max_length, dtype=torch.float32)
        inputs = self.feature_extractor(
            dummy_audio.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length
        )

        input_values = inputs['input_values'].squeeze(0)
        attention_mask = inputs.get(
            'attention_mask', torch.ones_like(input_values)).squeeze(0)

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'audio_path': f"dummy_{idx}",
            'target': torch.zeros(self.max_length, dtype=torch.long) if self.kmeans_targets is not None else None
        }

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
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([
            HuBERTTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len = hidden_states.shape[:2]

        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device)
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
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
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
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # Forward pass for the attention mechanism.
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(batch_size, seq_len, self.num_attention_heads,
                           self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads,
                       self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads,
                           self.attention_head_size).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            mask_value = -65504.0 if attention_scores.dtype == torch.float16 else -1e4
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, mask_value)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.all_head_size)

        # Apply output projection
        output = self.output(context)

        return output

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
        self.use_kmeans_targets = False  # NEW: set True if using k-means targets

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

    def _span_mask(self, seq_len, batch_size, attention_mask=None):
        # Official HuBERT: non-overlapping span masking
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.zeros((batch_size, seq_len),
                           dtype=torch.bool, device=device)

        for i in range(batch_size):
            valid_length = seq_len if attention_mask is None else int(
                attention_mask[i].sum().item())
            if valid_length <= self.mask_length:
                continue

            # Calculate number of spans to mask
            num_spans = max(
                1, int(valid_length * self.mask_prob) // self.mask_length)

            # Generate non-overlapping span positions
            span_starts = []
            attempts = 0
            max_attempts = valid_length * 2  # Prevent infinite loops

            while len(span_starts) < num_spans and attempts < max_attempts:
                start = torch.randint(
                    0, valid_length - self.mask_length + 1, (1,)).item()

                # Check if this span overlaps with existing spans
                overlap = False
                for existing_start in span_starts:
                    if (start < existing_start + self.mask_length and
                            start + self.mask_length > existing_start):
                        overlap = True
                        break

                if not overlap:
                    span_starts.append(start)
                    mask[i, start:start + self.mask_length] = True

                attempts += 1

        return mask

    # Forward pass through the model, including feature extraction, masking, and prediction.
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None  # NEW: pass k-means targets
    ) -> Dict[str, torch.Tensor]:

        # Feature extraction
        hidden_states = self.feature_extractor(input_values)

        # Calculate sequence reduction factor: 7 conv layers with total stride = 5 * 2^6 = 320
        seq_reduction_factor = 320

        # Adjust attention mask for feature encoder output
        if attention_mask is not None:
            # Calculate the new sequence length after feature extraction
            original_length = attention_mask.size(1)
            new_length = (original_length + seq_reduction_factor -
                          1) // seq_reduction_factor

            # Create new attention mask for the reduced sequence
            new_attention_mask = torch.zeros(attention_mask.size(0), new_length,
                                             dtype=attention_mask.dtype, device=attention_mask.device)

            for i in range(attention_mask.size(0)):
                # Count valid positions in the original sequence
                valid_positions = attention_mask[i].sum().item()
                # Calculate valid positions in the new sequence
                new_valid_positions = (
                    valid_positions + seq_reduction_factor - 1) // seq_reduction_factor
                new_attention_mask[i, :new_valid_positions] = 1

            attention_mask = new_attention_mask

        batch_size, seq_len, _ = hidden_states.shape

        # Apply span masking
        mask = self._span_mask(seq_len, batch_size, attention_mask)

        # Apply mask token to masked positions
        masked_hidden_states = hidden_states.clone()
        if mask.any():
            mask_token_expanded = self.mask_token.to(
                hidden_states.device).unsqueeze(0).expand_as(hidden_states)
            masked_hidden_states = torch.where(
                mask.unsqueeze(-1), mask_token_expanded, hidden_states)

        # Transformer encoding
        encoder_outputs = self.encoder(masked_hidden_states, attention_mask)
        predictions = self.prediction_head(encoder_outputs)

        # Calculate loss only on masked positions
        loss = None
        if mask.any() and target is not None:
            # Flatten predictions and targets
            flat_predictions = predictions.view(-1, self.config.vocab_size)
            flat_targets = target.view(-1)
            flat_mask = mask.view(-1)

            # Only compute loss on masked positions
            masked_predictions = flat_predictions[flat_mask]
            masked_targets = flat_targets[flat_mask]

            if masked_predictions.size(0) > 0:
                loss = F.cross_entropy(masked_predictions, masked_targets)

        return {
            'loss': loss,
            'predictions': predictions,
            'hidden_states': encoder_outputs,
            'targets': target,
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
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
        self.checkpoint_dir = Path(
            config['pretraining'].get('checkpoint_dir', 'checkpoints'))
        self.save_checkpoints = config['pretraining'].get(
            'save_checkpoints', True)
        self.checkpoint_freq = int(
            config['pretraining'].get('checkpoint_freq', 1))
        self.keep_last_n_checkpoints = int(
            config['pretraining'].get('keep_last_n_checkpoints', 3))

        # Create checkpoint directory
        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.client_checkpoint_dir = self.checkpoint_dir / \
                f"client_{self.client_id}"
            self.client_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float('inf')
        self.current_round = 0

        self._setup_feature_extractor()
        self._setup_data_loader()
        self._setup_model()

    # Setup the feature extractor for audio preprocessing. (Not for feature extraction)
    def _setup_feature_extractor(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base")

    def _setup_data_loader(self):
        train_manifest = self.data_path / "manifest.csv"
        if not train_manifest.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {train_manifest}")

        manifest_df = pd.read_csv(train_manifest)
        train_df = manifest_df[manifest_df['split'] == 'train']

        # Limit dataset size for faster training
        if len(train_df) > 2000:
            train_df = train_df.sample(n=2000, random_state=42)
            logger.info(
                f"Limited training data to 2000 samples for client {self.client_id}")

        train_manifest_path = self.data_path / "pretrain_manifest.csv"
        train_df.to_csv(train_manifest_path, index=False)

        # Calculate feature sequence length after feature extraction
        # 7 conv layers with total stride = 5 * 2^6 = 320
        seq_reduction_factor = 320
        feature_seq_len = self.max_audio_length // seq_reduction_factor

        # NEW: look for k-means cluster file
        kmeans_targets_file = self.config['pretraining'].get(
            'kmeans_targets_file', None)
        if kmeans_targets_file is not None and not os.path.isabs(kmeans_targets_file):
            kmeans_targets_file = str(self.data_path / kmeans_targets_file)

        # Check if k-means targets file exists
        use_auto_generation = False
        if kmeans_targets_file is None or not Path(kmeans_targets_file).exists():
            use_auto_generation = self.config['pretraining'].get(
                'auto_generate_kmeans', True)
            if use_auto_generation:
                logger.info(
                    f"K-means targets file not found. Auto-generating targets for client {self.client_id}")
            else:
                logger.warning(
                    f"No k-means targets provided and auto-generation disabled for client {self.client_id}")

        train_dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(train_manifest_path),
            audio_root=str(self.data_path),
            feature_extractor=self.feature_extractor,
            max_length=self.max_audio_length,
            mask_prob=float(self.config['pretraining']['mask_prob']),
            mask_length=int(self.config['pretraining']['mask_length']),
            kmeans_targets_file=kmeans_targets_file,
            feature_seq_len=feature_seq_len,
            auto_generate_kmeans=use_auto_generation,
            vocab_size=int(self.config['pretraining'].get('vocab_size', 504))
        )

        # Optimized number of workers based on available CPUs and system constraints
        max_workers = min(4, os.cpu_count() // 2)

        # Further reduce workers in federated simulation environment to prevent resource exhaustion
        if 'RAY_HEAD_NODE' in os.environ or 'RAY_NODE_IP_ADDRESS' in os.environ:
            # Use max 2 workers in Ray environment
            max_workers = min(2, max_workers)
            logger.info(
                f"Ray environment detected, reducing workers to {max_workers}")

        # Try different worker configurations if one fails
        for num_workers in [max_workers, max_workers//2, 1, 0]:
            try:
                logger.info(
                    f"Attempting DataLoader with {num_workers} workers")

                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if self.device.type == "cuda" and num_workers > 0 else False,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=2 if num_workers > 0 else None,
                    drop_last=True,  # Ensure consistent batch sizes
                    timeout=30 if num_workers > 0 else 0,  # Add timeout for worker processes
                    # Use spawn for better isolation
                    multiprocessing_context='spawn' if num_workers > 0 else None
                )

                # Test the DataLoader with a small batch to ensure it works
                test_iter = iter(self.train_loader)
                test_batch = next(test_iter)
                logger.info(
                    f"DataLoader test successful with {num_workers} workers")
                break

            except Exception as e:
                logger.warning(
                    f"DataLoader failed with {num_workers} workers: {e}")
                if num_workers == 0:
                    # If even single-threaded fails, there's a deeper issue
                    raise RuntimeError(
                        f"Failed to create DataLoader even with 0 workers: {e}")
                continue

    def _setup_model(self):
        model_config = self.config['pretraining']

        self.model = HuBERTPretrainingModel(
            vocab_size=int(model_config.get('vocab_size', 504)),
            hidden_size=int(model_config.get('hidden_size', 768)),
            num_hidden_layers=int(model_config.get('num_hidden_layers', 12)),
            num_attention_heads=int(
                model_config.get('num_attention_heads', 12)),
            intermediate_size=int(model_config.get('intermediate_size', 3072)),
            mask_prob=float(model_config.get('mask_prob', 0.05)),
            mask_length=int(model_config.get('mask_length', 10))
        ).to(self.device)
        # Set use_kmeans_targets if kmeans_targets_file is provided
        if model_config.get('kmeans_targets_file', None) is not None:
            self.model.use_kmeans_targets = True

    def save_checkpoint(self, loss: float, is_best: bool = False, round_num: int = None):
        """Save model checkpoint with training state."""
        if not self.save_checkpoints:
            return

        round_num = round_num or self.current_round

        # Save only state_dict for s3prl compatibility
        checkpoint = self.model.state_dict()
        latest_path = self.client_checkpoint_dir / \
            f"latest_round_{round_num}_state_dict.pt"
        torch.save(checkpoint, latest_path)
        logger.info(
            f"Saved s3prl-compatible checkpoint for client {self.client_id} at round {round_num}")

        # Save best checkpoint if this is the best loss so far
        if is_best:
            best_path = self.client_checkpoint_dir / "best_model_state_dict.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"Saved best s3prl-compatible checkpoint for client {self.client_id} with loss {loss:.4f}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, checkpoint_path: str = None):
        """Load model checkpoint and restore training state."""
        if checkpoint_path is None:
            # Try to load the latest checkpoint
            checkpoints = list(
                self.client_checkpoint_dir.glob("latest_round_*.pt"))
            if not checkpoints:
                logger.info(
                    f"No checkpoints found for client {self.client_id}")
                return False

            # Get the latest checkpoint by round number
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_path = str(latest_checkpoint)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint)

            # Restore training state
            self.best_loss = float('inf')  # Reset best_loss on load
            self.current_round = 0  # Reset current_round on load

            logger.info(
                f"Loaded s3prl-compatible checkpoint for client {self.client_id} from round {self.current_round}")
            logger.info(f"Best loss: {self.best_loss:.4f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(
            self.client_checkpoint_dir.glob("latest_round_*.pt"))

        if len(checkpoints) > self.keep_last_n_checkpoints:
            # Sort by round number and keep only the latest N
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            checkpoints_to_remove = checkpoints[:-self.keep_last_n_checkpoints]

            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove checkpoint {checkpoint}: {e}")

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
            weight_decay=float(
                self.config['pretraining'].get('weight_decay', 0.01)),
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
            torch.cuda.set_per_process_memory_fraction(
                0.90)  # Leave more memory for other processes

        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch in tqdm(self.train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}"):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target = batch.get('target', None)
                if target is not None:
                    target = target.to(self.device)

                optimizer.zero_grad()

                # Use mixed precision if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_values=input_values,
                            attention_mask=attention_mask,
                            target=target
                        )
                        loss = outputs['loss']

                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        batch_size = input_values.size(0)
                        epoch_loss += loss.item() * batch_size
                        epoch_samples += batch_size
                else:
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        target=target
                    )
                    loss = outputs['loss']

                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
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
            logger.warning(
                f"Client {self.client_id}: No valid samples processed during training")
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
    config_path = optimized_config_path if Path(
        optimized_config_path).exists() else original_config_path
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
        raise FileNotFoundError(
            f"Client data directory not found: {client_data_path}")

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
        weighted_sum = sum(num_examples * m[key]
                           for num_examples, m in metrics)
        weighted_metrics[key] = weighted_sum / total_examples

    return weighted_metrics


# Server-side checkpoint management


class ServerCheckpointManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = Path(
            config['pretraining'].get('checkpoint_dir', 'checkpoints'))
        self.server_checkpoint_dir = self.checkpoint_dir / "server"
        self.server_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_checkpoints = config['pretraining'].get(
            'save_checkpoints', True)
        self.checkpoint_freq = int(
            config['pretraining'].get('checkpoint_freq', 1))
        self.keep_last_n_checkpoints = int(
            config['pretraining'].get('keep_last_n_checkpoints', 3))

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
            hidden_size=int(
                self.config['pretraining'].get('hidden_size', 768)),
            num_hidden_layers=int(
                self.config['pretraining'].get('num_hidden_layers', 12)),
            num_attention_heads=int(
                self.config['pretraining'].get('num_attention_heads', 12)),
            intermediate_size=int(
                self.config['pretraining'].get('intermediate_size', 3072)),
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
        latest_path = self.server_checkpoint_dir / \
            f"global_model_round_{round_num}.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved global model checkpoint at round {round_num}")

        # Save best checkpoint if this is the best loss so far
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss
            best_path = self.server_checkpoint_dir / "best_global_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"Saved best global model checkpoint with loss {current_loss:.4f}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def load_global_model(self, checkpoint_path: str = None):
        """Load global model checkpoint."""
        if checkpoint_path is None:
            # Try to load the latest checkpoint
            checkpoints = list(self.server_checkpoint_dir.glob(
                "global_model_round_*.pt"))
            if not checkpoints:
                logger.info("No global model checkpoints found")
                return None

            # Get the latest checkpoint by round number
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_path = str(latest_checkpoint)

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract parameters as NDArrays and convert to Parameters object
            parameters_ndarrays = [
                param.numpy() for param in checkpoint['model_state_dict'].values()]
            parameters = ndarrays_to_parameters(parameters_ndarrays)

            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.current_round = checkpoint.get('round', 0)

            logger.info(
                f"Loaded global model checkpoint from round {self.current_round}")
            logger.info(f"Best global loss: {self.best_loss:.4f}")

            return parameters

        except Exception as e:
            logger.error(
                f"Failed to load global model checkpoint {checkpoint_path}: {e}")
            return None

    def _cleanup_old_checkpoints(self):
        """Remove old global model checkpoints to save disk space."""
        checkpoints = list(self.server_checkpoint_dir.glob(
            "global_model_round_*.pt"))

        if len(checkpoints) > self.keep_last_n_checkpoints:
            # Sort by round number and keep only the latest N
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            checkpoints_to_remove = checkpoints[:-self.keep_last_n_checkpoints]

            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    logger.debug(
                        f"Removed old global checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove global checkpoint {checkpoint}: {e}")

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
    config_path = optimized_config_path if Path(
        optimized_config_path).exists() else original_config_path
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
            num_hidden_layers=int(
                pretraining_config.get('num_hidden_layers', 12)),
            num_attention_heads=int(
                pretraining_config.get('num_attention_heads', 12)),
            intermediate_size=int(
                pretraining_config.get('intermediate_size', 3072)),
            mask_prob=float(pretraining_config.get('mask_prob', 0.05)),
            mask_length=int(pretraining_config.get('mask_length', 10))
        )
        initial_parameters_ndarrays = [
            param.cpu().numpy() for param in dummy_model.state_dict().values()]
        initial_parameters = ndarrays_to_parameters(
            initial_parameters_ndarrays)

    strategy_args['initial_parameters'] = initial_parameters

    strategy = FedAdamWithCheckpoints(**strategy_args)

    server_config = ServerConfig(
        num_rounds=int(pretraining_config['num_rounds'])
    )

    return ServerAppComponents(strategy=strategy, config=server_config)


def evaluate_fn(
    server_round: int,
    parameters: Parameters,
    config: Config,
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the global model on a small subset of data."""
    try:
        # Load the global model
        model = HuBERTPretrainingModel(
            vocab_size=504,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        # Set parameters
        params_dict = zip(model.parameters(),
                          parameters_to_ndarrays(parameters))
        for param, param_array in params_dict:
            param.data = torch.from_numpy(param_array)

        model.eval()

        # Use a small subset of data for evaluation
        eval_dataset = LibriSpeechPretrainingDataset(
            manifest_file="/lustre07/scratch/saadan/federated_librispeech/src/data/client_0/manifest.csv",
            feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-base-ls960"),
            max_length=80000,  # 5 seconds
            sample_rate=16000,
            kmeans_targets=None  # No k-means for evaluation
        )

        # Create a small evaluation dataloader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                if num_batches >= 10:  # Limit evaluation to 10 batches
                    break

                input_values = batch['input_values']
                attention_mask = batch.get('attention_mask', None)

                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )

                loss = outputs['loss']
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Server evaluation - Round {server_round}: Loss = {avg_loss:.4f}")
            return avg_loss, {"eval_pretrain_loss": avg_loss}
        else:
            logger.warning(
                f"Server evaluation - Round {server_round}: No valid batches for evaluation")
            return None

    except Exception as e:
        logger.error(f"Server evaluation failed for round {server_round}: {e}")
        return None


def main():
    """Main function to run federated HuBERT pretraining."""
    parser = argparse.ArgumentParser(
        description="Federated HuBERT Pretraining")
    parser.add_argument("--config", type=str, required=True,
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

    logger.info("Starting Federated HuBERT Pretraining")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Number of clients: {args.num_clients}")
    logger.info(f"Number of rounds: {args.num_rounds}")

    try:

        # Run simulation with better error handling
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=args.num_clients,
            backend_name="ray",
            backend_config={
                "client_resources": config['simulation']['backend']['config']['client_resources'],
                "init_args": config['simulation']['backend']['config']['init_args'],
            },
        )

        logger.info("Federated pretraining completed successfully!")

    except Exception as e:
        logger.error(f"Federated pretraining failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
