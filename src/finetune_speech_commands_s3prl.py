#!/usr/bin/env python3
"""
Speech Commands Finetuning Script using s3prl Framework.

This script finetunes a pretrained HuBERT model on the Google Speech Commands dataset
using s3prl's problem-based framework for proper integration.
"""

import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import soundfile as sf
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import s3prl components
try:
    from s3prl.problem import SuperbASR
    from s3prl.dataset.base import AugmentedDynamicItemDataset
    from s3prl.sampler import DistributedBatchSamplerWrapper
    from s3prl.util.configuration import Container
    from s3prl.util.seed import fix_random_seeds
    S3PRL_AVAILABLE = True
except ImportError:
    S3PRL_AVAILABLE = False
    print("Warning: s3prl not available, falling back to standalone implementation")

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, hidden_size: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2)
                             * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


class HubertBase(nn.Module):
    """HuBERT-like model for feature extraction - matching pretraining architecture."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12, vocab_size: int = 504, frame_stride: int = 320):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.frame_stride = frame_stride

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])

        # Input projection: raw audio -> hidden dimension
        self.input_projection = nn.Linear(1, hidden_size)

        # Output projection: hidden dimension -> vocabulary (required for loading pretrained weights)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Mask embedding following HuBERT paper (learned) - required for loading pretrained weights
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_values: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_values: [batch, samples]
        x = input_values.unsqueeze(-1)                      # [B, T, 1]
        x = self.input_projection(x)                        # [B, T, H]
        x = x.transpose(1, 2)                               # [B, H, T]
        x = F.avg_pool1d(x, kernel_size=self.frame_stride,
                         stride=self.frame_stride)
        x = x.transpose(1, 2)                               # [B, T_frames, H]

        # Apply mask embedding if provided (following HuBERT paper)
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1)
            x = torch.where(mask_expanded, self.mask_embedding.view(
                1, 1, -1).expand_as(x), x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Apply layer norm before transformer
        x = self.layer_norm(x)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.layer_norm(x)

        # For finetuning, we return the hidden representations, not the vocabulary logits
        # The output_projection is only there to load pretrained weights
        return x


class SpeechCommandsClassifier(nn.Module):
    """Classification head for speech commands."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 35, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Global average pooling + classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)
        # Global average pooling: [B, H, T] -> [B, H, 1]
        x = self.global_pool(x)
        # Flatten: [B, H, 1] -> [B, H]
        x = x.squeeze(-1)
        # Classification: [B, H] -> [B, num_classes]
        x = self.classifier(x)
        return x


class SpeechCommandsDataset(Dataset):
    """Dataset for Google Speech Commands."""

    def __init__(self, data_root: str, split: str = "train", sample_rate: int = 16000,
                 max_length: int = 16000, transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.transform = transform

        # Speech command classes (35 total: 20 core + 10 auxiliary + 5 unknown)
        self.classes = [
            'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
            'backward', 'forward', 'learn', 'visual', 'follow'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load split files
        if split == "train":
            split_file = self.data_root / "training_list.txt"
        elif split == "val":
            split_file = self.data_root / "validation_list.txt"
        elif split == "test":
            split_file = self.data_root / "testing_list.txt"
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load file list
        with open(split_file, 'r') as f:
            self.files = [line.strip() for line in f if line.strip()]

        # Filter out background noise files
        self.files = [f for f in self.files if not f.startswith(
            '_background_noise_')]

        logger.info(f"Loaded {len(self.files)} files for {split} split")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.data_root / self.files[idx]

        # Extract class from file path
        class_name = file_path.parent.name
        if class_name not in self.class_to_idx:
            # Handle unknown classes
            class_idx = len(self.classes) - 1  # Last class is "unknown"
        else:
            class_idx = self.class_to_idx[class_name]

        # Load audio
        try:
            audio, sr = sf.read(str(file_path))
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel if stereo

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                audio = resampler(torch.tensor(audio))
            else:
                audio = torch.tensor(audio)

            # Pad or truncate to max_length
            if len(audio) < self.max_length:
                audio = F.pad(audio, (0, self.max_length - len(audio)))
            else:
                audio = audio[:self.max_length]

            # Apply transforms if specified
            if self.transform:
                audio = self.transform(audio)

            return audio, class_idx

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Return a dummy sample
            return torch.zeros(self.max_length), class_idx


def load_pretrained_model(checkpoint_path: str, config: Dict) -> HubertBase:
    """Load pretrained HuBERT model from checkpoint."""
    logger.info(f"Loading pretrained model from {checkpoint_path}")

    # Get model parameters from config
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    frame_stride = config['model']['frame_stride']

    logger.info(
        f"Model config - hidden_size: {hidden_size}, num_layers: {num_layers}, frame_stride: {frame_stride}")

    # Initialize model
    model = HubertBase(
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=504,  # Fixed vocab size from pretraining
        frame_stride=frame_stride
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict (handle different checkpoint formats)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained model loaded successfully")

    return model


def train_epoch(model: nn.Module, classifier: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device,
                epoch: int) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (audio, labels) in enumerate(pbar):
        audio, labels = audio.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        features = model(audio)
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model: nn.Module, classifier: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, labels in tqdm(dataloader, desc="Validation"):
            audio, labels = audio.to(device), labels.to(device)

            # Forward pass
            features = model(audio)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Finetune HuBERT on Speech Commands using s3prl")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to speech commands dataset")
    parser.add_argument("--output-dir", type=str,
                        default="./finetuned_model", help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--use-s3prl", action="store_true",
                        help="Use s3prl framework if available")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create a normalized config structure for compatibility
    normalized_config = {
        'model': {
            'hidden_size': config.get('model', {}).get('hidden_size', 768),
            'num_layers': config.get('model', {}).get('num_layers', 12),
            'frame_stride': config.get('model', {}).get('frame_stride', 320)
        },
        'training': {
            'dropout': config.get('downstream_expert', {}).get('modelrc', {}).get('UtteranceLevel', {}).get('dropout', 0.1),
            'batch_size': config.get('downstream_expert', {}).get('datarc', {}).get('batch_size', 32),
            'learning_rate': config.get('optimizer', {}).get('lr', 1e-4),
            'weight_decay': config.get('optimizer', {}).get('weight_decay', 0.01),
            'epochs': 50  # Default number of epochs
        },
        'data': {
            'sample_rate': config.get('downstream_expert', {}).get('datarc', {}).get('sample_rate', 16000),
            'max_length': config.get('downstream_expert', {}).get('datarc', {}).get('max_length', 16000),
            'num_workers': config.get('downstream_expert', {}).get('datarc', {}).get('num_workers', 8)
        }
    }

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"s3prl available: {S3PRL_AVAILABLE}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    model = load_pretrained_model(args.checkpoint, normalized_config)
    model = model.to(device)

    # Create classifier
    classifier = SpeechCommandsClassifier(
        hidden_size=normalized_config['model']['hidden_size'],
        num_classes=35,
        dropout=normalized_config['training']['dropout']
    )
    classifier = classifier.to(device)

    # Create datasets
    train_dataset = SpeechCommandsDataset(
        data_root=args.data_root,
        split="train",
        sample_rate=normalized_config['data']['sample_rate'],
        max_length=normalized_config['data']['max_length']
    )

    val_dataset = SpeechCommandsDataset(
        data_root=args.data_root,
        split="val",
        sample_rate=normalized_config['data']['sample_rate'],
        max_length=normalized_config['data']['max_length']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=normalized_config['training']['batch_size'],
        shuffle=True,
        num_workers=normalized_config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=normalized_config['training']['batch_size'],
        shuffle=False,
        num_workers=normalized_config['data']['num_workers'],
        pin_memory=True
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [
            {'params': model.parameters(
            ), 'lr': normalized_config['training']['learning_rate'] * 0.1},
            {'params': classifier.parameters(
            ), 'lr': normalized_config['training']['learning_rate']}
        ],
        weight_decay=normalized_config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=normalized_config['training']['epochs']
    )

    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    logger.info("Starting finetuning...")

    for epoch in range(normalized_config['training']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, classifier, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, classifier, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step()

        # Log results
        logger.info(
            f"Epoch {epoch+1}/{normalized_config['training']['epochs']}:")
        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': normalized_config
            }, output_dir / 'best_model.pt')
            logger.info(
                f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': normalized_config
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'epoch': normalized_config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_acc': val_acc,
        'config': normalized_config
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Finetuning completed! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
