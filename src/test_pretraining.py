#!/usr/bin/env python3
"""
Test script for the simplified federated HuBERT pretraining
"""

from federated_hubert_pretraining import HubertBase, LibriSpeechPretrainingDataset
import os
import sys
import logging
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model():
    """Test the HubertBase model"""
    logger.info("Testing HubertBase model...")

    # Create model
    model = HubertBase()
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 40000  # 2.5 seconds at 16kHz
    input_values = torch.randn(batch_size, seq_len)

    with torch.no_grad():
        outputs = model(input_values)
        predictions = outputs['predictions']

    logger.info(f"Input shape: {input_values.shape}")
    logger.info(f"Output shape: {predictions.shape}")
    logger.info(f"Expected output shape: ({batch_size}, {seq_len}, 504)")

    assert predictions.shape == (
        batch_size, seq_len, 504), f"Expected shape ({batch_size}, {seq_len}, 504), got {predictions.shape}"
    logger.info("✅ Model forward pass test passed!")


def test_dataset():
    """Test the dataset (if data is available)"""
    logger.info("Testing dataset...")

    # Check if test data exists
    test_data_path = Path("federated_librispeech/data/client_0")
    manifest_path = test_data_path / "manifest.csv"

    if not manifest_path.exists():
        logger.warning("Test data not found, skipping dataset test")
        logger.info(
            "To test dataset, ensure you have data at: federated_librispeech/data/client_0/manifest.csv")
        return

    try:
        # Create dataset
        dataset = LibriSpeechPretrainingDataset(
            manifest_file=str(manifest_path),
            audio_root=str(test_data_path),
            split="train",
            max_length=40000,
            sample_rate=16000,
            mask_prob=0.08,
            mask_length=10,
            vocab_size=504
        )

        logger.info(f"Dataset created with {len(dataset)} samples")

        # Test getting a sample
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Input values shape: {sample['input_values'].shape}")
        logger.info(f"Targets shape: {sample['targets'].shape}")
        logger.info(f"Mask shape: {sample['mask'].shape}")

        logger.info("✅ Dataset test passed!")

    except Exception as e:
        logger.error(f"Dataset test failed: {e}")


def test_training_step():
    """Test a single training step"""
    logger.info("Testing training step...")

    # Create model
    model = HubertBase()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Create dummy data
    batch_size = 2
    seq_len = 40000
    input_values = torch.randn(batch_size, seq_len)
    # Approximate feature sequence length
    targets = torch.randint(0, 504, (batch_size, seq_len // 320))
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

    # Training step
    model.train()
    optimizer.zero_grad()

    outputs = model(input_values)
    predictions = outputs['predictions']

    # Apply mask and compute loss
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    if masked_predictions.size(0) > 0:
        loss = torch.nn.functional.cross_entropy(
            masked_predictions.view(-1, 504),
            masked_targets.view(-1)
        )

        loss.backward()
        optimizer.step()

        logger.info(f"Training step completed, loss: {loss.item():.4f}")
        logger.info("✅ Training step test passed!")
    else:
        logger.warning("No masked predictions, skipping loss computation")


def main():
    """Run all tests"""
    logger.info("Starting tests for simplified federated HuBERT pretraining...")

    try:
        test_model()
        test_dataset()
        test_training_step()

        logger.info("🎉 All tests passed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
