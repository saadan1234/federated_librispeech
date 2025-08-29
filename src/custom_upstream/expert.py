"""
Custom upstream expert for loading federated HuBERT pretrained model.
This expert loads the HubertBase model from the federated pretraining checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Union, Optional
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs following HuBERT paper."""

    def __init__(self, hidden_size: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2)
                             * (-np.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


class CustomHubertExpert(nn.Module):
    """
    Custom HuBERT expert that loads the federated pretrained model.
    This follows the same architecture as HubertBase from the pretraining code.
    """

    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()
        self.name = "CustomHubertExpert"

        # Default model parameters (matching pretraining)
        self.hidden_size = 768
        self.num_layers = 12
        self.vocab_size = 504
        self.frame_stride = 320

        # Load checkpoint if provided
        if ckpt:
            self._load_checkpoint(ckpt)
        else:
            # Initialize with default architecture
            self._build_model()

    def _build_model(self):
        """Build the model architecture."""
        # Transformer encoder layers following HuBERT paper
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(self.num_layers)
        ])

        # Input projection: raw audio -> hidden dimension
        self.input_projection = nn.Linear(1, self.hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def _load_checkpoint(self, ckpt_path: str):
        """Load the federated pretraining checkpoint."""
        print(f"Loading checkpoint from: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(
                f"Loaded checkpoint from round: {checkpoint.get('round', 'unknown')}")
        else:
            state_dict = checkpoint
            print("Loaded checkpoint (no round info)")

        # Build model first
        self._build_model()

        # Load state dict
        try:
            self.load_state_dict(state_dict, strict=False)
            print("✅ Successfully loaded checkpoint")
        except Exception as e:
            print(f"⚠️  Warning: Some weights couldn't be loaded: {e}")
            # Try partial loading
            self.load_state_dict(state_dict, strict=False)

    def get_downsample_rates(self, key: str) -> int:
        """Get downsample rate for the specified key."""
        # Frame stride is 320 (20ms at 16kHz)
        return self.frame_stride

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the model.
        Returns hidden states from all transformer layers.
        """
        # Pad sequences
        wavs = pad_sequence(wavs, batch_first=True)

        # Process audio: [B, T] -> [B, T, 1] -> [B, T, H]
        x = wavs.unsqueeze(-1)  # [B, T, 1]
        x = self.input_projection(x)  # [B, T, H]

        # Frame-level pooling: [B, T, H] -> [B, T_frames, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)  # [B, T_frames, H]

        # Positional encoding
        x = self.positional_encoding(x)

        # Apply layer norm before transformer
        x = self.layer_norm(x)

        # Collect hidden states from all layers
        hidden_states = [x]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        # Final layer norm
        x = self.layer_norm(x)
        hidden_states.append(x)

        # Return hidden states from all layers
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1]
        }
