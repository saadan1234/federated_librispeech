#!/usr/bin/env python3
"""
Test script to verify enhanced logging functionality for federated distillation
"""

from federated_hubert_distillation import (
    FederatedHuBERTDistillationClient,
    weighted_average,
    evaluate_fn
)
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')


def test_enhanced_logging():
    """Test the enhanced logging functionality"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Testing enhanced logging functionality...")

    # Test weighted average function
    test_metrics = [
        (100, {"student_loss": 0.5, "accuracy": 0.8, "client_id": 1}),
        (150, {"student_loss": 0.4, "accuracy": 0.85, "client_id": 2}),
        (200, {"student_loss": 0.3, "accuracy": 0.9, "client_id": 3})
    ]

    logger.info("Testing weighted average calculation...")
    aggregated_metrics = weighted_average(test_metrics)

    logger.info(f"Aggregated metrics: {aggregated_metrics}")

    # Test client creation and logging
    try:
        # Create a test client
        config = {
            'distillation': {
                'vocab_size': 504,
                'mask_prob': 0.08,
                'mask_length': 10,
                'temperature': 4.0,
                'alpha': 0.7
            },
            'client': {
                'local_epochs': 1,
                'batch_size': 4,
                'learning_rate': 1e-4
            }
        }

        # Test data path
        test_data_path = Path("src/data/client_0")
        if test_data_path.exists():
            logger.info(f"Test data path exists: {test_data_path}")

            # Try to create client (this will test the logging)
            client = FederatedHuBERTDistillationClient(
                client_id=0,
                data_path=str(test_data_path),
                config=config
            )

            logger.info("Client created successfully")

        else:
            logger.warning(f"Test data path does not exist: {test_data_path}")

    except Exception as e:
        logger.error(f"Error testing client creation: {e}")

    logger.info("Enhanced logging test completed!")


if __name__ == "__main__":
    test_enhanced_logging()
