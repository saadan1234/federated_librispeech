#!/usr/bin/env python3
"""
Test script to verify real data loading for distillation
"""

from federated_hubert_distillation import LibriSpeechDistillationDataset, FederatedClient
import os
import sys
import logging
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test that data loading works correctly"""

    # Load config
    config_path = "configs/distillation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get data path
    data_root_config = config['data']['partitioned_data_root']
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)

    logger.info(f"Testing data loading from: {data_root}")

    # Test each client
    for client_id in range(10):
        client_data_path = data_root / f"client_{client_id}"
        manifest_path = client_data_path / "manifest.csv"

        logger.info(f"\n--- Testing Client {client_id} ---")
        logger.info(f"Client data path: {client_data_path}")
        logger.info(f"Manifest path: {manifest_path}")

        if not client_data_path.exists():
            logger.warning(f"Client {client_id} data directory not found")
            continue

        if not manifest_path.exists():
            logger.warning(f"Client {client_id} manifest not found")
            continue

        try:
            # Test train dataset creation
            train_dataset = LibriSpeechDistillationDataset(
                manifest_file=str(manifest_path),
                audio_root=str(client_data_path),
                split="train",
                max_length=80000,
                sample_rate=16000,
                mask_prob=0.15,
                mask_length=10,
                vocab_size=504
            )

            # Test validation dataset creation
            val_dataset = LibriSpeechDistillationDataset(
                manifest_file=str(manifest_path),
                audio_root=str(client_data_path),
                split="validation",
                max_length=80000,
                sample_rate=16000,
                mask_prob=0.15,
                mask_length=10,
                vocab_size=504
            )

            logger.info(f"Client {client_id}: Train dataset created with {len(train_dataset)} samples")
            logger.info(f"Client {client_id}: Val dataset created with {len(val_dataset)} samples")

            # Test loading a few samples from train
            for i in range(min(3, len(train_dataset))):
                try:
                    sample = train_dataset[i]
                    logger.info(f"Client {client_id} Train Sample {i}: {sample['input_values'].shape}, targets: {sample['targets'].shape}")
                except Exception as e:
                    logger.error(f"Client {client_id} Train Sample {i} failed: {e}")

            # Test loading a few samples from validation
            for i in range(min(3, len(val_dataset))):
                try:
                    sample = val_dataset[i]
                    logger.info(f"Client {client_id} Val Sample {i}: {sample['input_values'].shape}, targets: {sample['targets'].shape}")
                except Exception as e:
                    logger.error(f"Client {client_id} Val Sample {i} failed: {e}")

            # Test client initialization
            try:
                client = FederatedClient(
                    client_id=client_id, data_path=str(client_data_path))
                logger.info(f"Client {client_id}: Initialized successfully")
                logger.info(
                    f"Client {client_id}: Train samples: {len(client.train_dataset)}")
                logger.info(
                    f"Client {client_id}: Val samples: {len(client.val_dataset)}")
            except Exception as e:
                logger.error(f"Client {client_id} initialization failed: {e}")

        except Exception as e:
            logger.error(f"Client {client_id} dataset creation failed: {e}")


def test_single_client():
    """Test a single client in detail"""
    config_path = "configs/distillation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_root_config = config['data']['partitioned_data_root']
    if not os.path.isabs(data_root_config):
        data_root = Path.cwd() / data_root_config
    else:
        data_root = Path(data_root_config)

    client_id = 0
    client_data_path = data_root / f"client_{client_id}"

    logger.info(f"Testing client {client_id} in detail")
    logger.info(f"Data path: {client_data_path}")

    if not client_data_path.exists():
        logger.error(f"Client data path does not exist: {client_data_path}")
        return

    # List contents
    logger.info("Client directory contents:")
    for item in client_data_path.iterdir():
        logger.info(f"  {item.name}")

    manifest_path = client_data_path / "manifest.csv"
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        logger.info(f"Manifest has {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            logger.info(f"First row: {dict(df.iloc[0])}")
    else:
        logger.error("Manifest file not found")


if __name__ == "__main__":
    logger.info("Starting data loading tests...")

    # Test single client first
    test_single_client()

    # Test all clients
    test_data_loading()

    logger.info("Data loading tests completed")
