#!/usr/bin/env python3
"""
Test script to verify checkpointing functionality
"""

from federated_hubert_distillation import CheckpointManager, HubertStudent
import os
import sys
import logging
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_checkpoint_manager():
    """Test the checkpoint manager functionality"""

    # Create checkpoint manager
    checkpoint_dir = "test_checkpoints"
    manager = CheckpointManager(save_dir=checkpoint_dir)

    # Create a dummy model
    model = HubertStudent()
    state_dict = model.state_dict()

    # Convert to numpy arrays (simulating FL parameters)
    parameters = [val.cpu().numpy() for val in state_dict.values()]

    # Test saving global model
    logger.info("Testing global model saving...")
    checkpoint_path = manager.save_global_model(
        parameters=parameters,
        round_num=1,
        metrics={"loss": 0.5, "accuracy": 0.8}
    )
    logger.info(f"Global model saved to: {checkpoint_path}")

    # Test saving client model
    logger.info("Testing client model saving...")
    client_checkpoint_path = manager.save_client_model(
        client_id=0,
        parameters=parameters,
        round_num=1,
        metrics={"loss": 0.6, "client_id": 0}
    )
    logger.info(f"Client model saved to: {client_checkpoint_path}")

    # Test loading global model
    logger.info("Testing global model loading...")
    loaded_checkpoint = manager.load_latest_global_model()
    if loaded_checkpoint:
        logger.info(
            f"Loaded global model from round {loaded_checkpoint['round']}")
        logger.info(f"Metrics: {loaded_checkpoint['metrics']}")

    # Test loading client model
    logger.info("Testing client model loading...")
    loaded_client_checkpoint = manager.load_client_model(client_id=0)
    if loaded_client_checkpoint:
        logger.info(
            f"Loaded client model from round {loaded_client_checkpoint['round']}")
        logger.info(f"Client ID: {loaded_client_checkpoint['client_id']}")

    # Test saving training history
    logger.info("Testing training history saving...")
    history = {
        "loss": [0.5, 0.4, 0.3],
        "accuracy": [0.7, 0.8, 0.9],
        "rounds": [1, 2, 3]
    }
    history_path = manager.save_training_history(history, round_num=3)
    logger.info(f"Training history saved to: {history_path}")

    # Clean up
    import shutil
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Cleaned up test directory: {checkpoint_dir}")


def test_model_saving():
    """Test saving and loading model state"""

    # Create a student model
    model = HubertStudent()

    # Save model state
    checkpoint_dir = Path("test_model_checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'hidden_size': 384,
            'num_layers': 3,
            'vocab_size': 504
        }
    }

    checkpoint_path = checkpoint_dir / "test_model.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Model saved to: {checkpoint_path}")

    # Load model state
    loaded_checkpoint = torch.load(checkpoint_path)
    logger.info(f"Model loaded from: {checkpoint_path}")
    logger.info(f"Checkpoint keys: {list(loaded_checkpoint.keys())}")

    # Clean up
    import shutil
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Cleaned up test directory: {checkpoint_dir}")


if __name__ == "__main__":
    logger.info("Starting checkpointing tests...")

    # Test checkpoint manager
    test_checkpoint_manager()

    # Test model saving
    test_model_saving()

    logger.info("Checkpointing tests completed successfully!")
