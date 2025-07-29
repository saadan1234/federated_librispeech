#!/usr/bin/env python3
"""
Test script for custom collate function to verify None handling and identify problematic files.
"""

import torch
import logging
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import traceback

# Add src to path to import the custom collate function
sys.path.append('/lustre07/scratch/saadan/federated_librispeech/src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('collate_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the custom collate function
try:
    from federated_hubert_distillation import custom_collate_fn
    logger.info("Successfully imported custom_collate_fn")
except ImportError as e:
    logger.error(f"Failed to import custom_collate_fn: {e}")
    sys.exit(1)


class TestDataset(Dataset):
    """Test dataset that can return None values to test collate function"""
    
    def __init__(self, size=100, none_probability=0.2):
        self.size = size
        self.none_probability = none_probability
        self.data = []
        
        # Generate test data with some None values
        for i in range(size):
            if torch.rand(1).item() < none_probability:
                self.data.append(None)
            else:
                # Create a valid sample mimicking the expected structure
                sample = {
                    'input_values': torch.randn(16000),  # 1 second at 16kHz
                    'attention_mask': torch.ones(16000),
                    'audio_path': f'test_audio_{i}.wav',
                    'target': torch.randint(0, 100, (1,)) if torch.rand(1).item() > 0.5 else None
                }
                self.data.append(sample)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


class RealDatasetWrapper(Dataset):
    """Wrapper around real dataset to track None returns"""
    
    def __init__(self, real_dataset):
        self.real_dataset = real_dataset
        self.none_indices = []
        self.error_indices = []
        self.total_accessed = 0
    
    def __len__(self):
        return len(self.real_dataset)
    
    def __getitem__(self, idx):
        self.total_accessed += 1
        try:
            item = self.real_dataset[idx]
            if item is None:
                self.none_indices.append(idx)
                logger.warning(f"Dataset returned None for index {idx}")
            return item
        except Exception as e:
            self.error_indices.append(idx)
            logger.error(f"Error accessing index {idx}: {e}")
            return None
    
    def get_stats(self):
        return {
            'total_accessed': self.total_accessed,
            'none_count': len(self.none_indices),
            'error_count': len(self.error_indices),
            'none_indices': self.none_indices[:10],  # First 10 for logging
            'error_indices': self.error_indices[:10]  # First 10 for logging
        }


def test_collate_scenarios():
    """Test different batch scenarios"""
    logger.info("=" * 60)
    logger.info("TESTING COLLATE FUNCTION SCENARIOS")
    logger.info("=" * 60)
    
    # Test 1: All valid samples
    logger.info("\n--- Test 1: All valid samples ---")
    valid_batch = [
        {'input_values': torch.randn(100), 'attention_mask': torch.ones(100), 'audio_path': 'test1.wav'},
        {'input_values': torch.randn(100), 'attention_mask': torch.ones(100), 'audio_path': 'test2.wav'},
        {'input_values': torch.randn(100), 'attention_mask': torch.ones(100), 'audio_path': 'test3.wav'}
    ]
    result = custom_collate_fn(valid_batch)
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Valid samples: {result.get('valid_samples', 'N/A')}")
    logger.info(f"Total samples: {result.get('total_samples', 'N/A')}")
    logger.info(f"Empty batch: {result.get('empty_batch', False)}")
    
    # Test 2: Mixed batch (some None)
    logger.info("\n--- Test 2: Mixed batch (some None) ---")
    mixed_batch = [
        {'input_values': torch.randn(100), 'attention_mask': torch.ones(100), 'audio_path': 'test1.wav'},
        None,
        {'input_values': torch.randn(100), 'attention_mask': torch.ones(100), 'audio_path': 'test3.wav'},
        None
    ]
    result = custom_collate_fn(mixed_batch)
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Valid samples: {result.get('valid_samples', 'N/A')}")
    logger.info(f"Total samples: {result.get('total_samples', 'N/A')}")
    logger.info(f"Empty batch: {result.get('empty_batch', False)}")
    
    # Test 3: All None
    logger.info("\n--- Test 3: All None samples ---")
    none_batch = [None, None, None]
    result = custom_collate_fn(none_batch)
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Valid samples: {result.get('valid_samples', 'N/A')}")
    logger.info(f"Total samples: {result.get('total_samples', 'N/A')}")
    logger.info(f"Empty batch: {result.get('empty_batch', False)}")
    
    # Test 4: Empty batch
    logger.info("\n--- Test 4: Empty batch ---")
    try:
        empty_result = custom_collate_fn([])
        logger.info(f"Empty batch result: {empty_result}")
    except Exception as e:
        logger.error(f"Error with empty batch: {e}")


def test_dataloader_with_test_dataset():
    """Test DataLoader with synthetic test dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DATALOADER WITH SYNTHETIC DATASET")
    logger.info("=" * 60)
    
    # Create test dataset with 30% None probability
    test_dataset = TestDataset(size=50, none_probability=0.3)
    
    # Create DataLoader with custom collate function
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0  # Single process for easier debugging
    )
    
    batch_stats = []
    total_valid_samples = 0
    total_samples = 0
    empty_batches = 0
    
    logger.info(f"Dataset size: {len(test_dataset)}")
    logger.info(f"Expected batches: {len(test_loader)}")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing test batches")):
        if isinstance(batch, dict) and batch.get('empty_batch', False):
            empty_batches += 1
            logger.warning(f"Batch {batch_idx}: Empty batch (all None)")
            continue
        
        valid_samples = batch.get('valid_samples', 0)
        batch_total = batch.get('total_samples', 0)
        total_valid_samples += valid_samples
        total_samples += batch_total
        
        batch_stats.append({
            'batch_idx': batch_idx,
            'valid_samples': valid_samples,
            'total_samples': batch_total,
            'filtered': batch_total - valid_samples
        })
        
        logger.info(f"Batch {batch_idx}: {valid_samples}/{batch_total} valid samples")
    
    logger.info(f"\nSUMMARY:")
    logger.info(f"Total batches processed: {len(batch_stats)}")
    logger.info(f"Empty batches skipped: {empty_batches}")
    logger.info(f"Total valid samples: {total_valid_samples}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Overall filtering rate: {(total_samples - total_valid_samples) / total_samples * 100:.1f}%")


def test_real_dataset():
    """Test with real dataset to identify None-returning files"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH REAL DATASET")
    logger.info("=" * 60)
    
    try:
        # Try to import and create real dataset
        from federated_hubert_pretraining import LibriSpeechPretrainingDataset
        
        # Look for config file
        config_paths = [
            '/home/saadan/scratch/federated_librispeech/src/configs/distillation_config.yaml',
            '/home/saadan/scratch/federated_librispeech/src/configs/distillation_config_optimized.yaml'
        ]
        
        config_path = None
        for path in config_paths:
            if Path(path).exists():
                config_path = path
                break
        
        if not config_path:
            logger.warning("No config file found, skipping real dataset test")
            return
        
        logger.info(f"Using config: {config_path}")
        
        # Create a small test dataset
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Use the partitioned data from client_0 as an example
        partitioned_data_root = config['data']['partitioned_data_root']
        manifest_path = f"{partitioned_data_root}/client_0/distill_manifest.csv"
        
        if not Path(manifest_path).exists():
            logger.warning(f"Manifest not found at {manifest_path}, skipping real dataset test")
            return
        
        # Import feature extractor
        from transformers import Wav2Vec2FeatureExtractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Check for K-means targets file
        kmeans_targets_file = f"{partitioned_data_root}/client_0/kmeans_targets.pkl"
        if Path(kmeans_targets_file).exists():
            logger.info(f"Using K-means targets from {kmeans_targets_file}")
        else:
            logger.warning(f"K-means targets not found at {kmeans_targets_file}")
            kmeans_targets_file = None
        
        # Create dataset with limited samples for testing
        dataset = LibriSpeechPretrainingDataset(
            manifest_file=manifest_path,
            audio_root=partitioned_data_root + "/client_0",  # Audio files are relative to this root
            feature_extractor=feature_extractor,
            max_length=config['data'].get('max_length', 160000),
            sample_rate=config['data'].get('sample_rate', 16000),
            mask_prob=0.05,
            mask_length=10,
            kmeans_targets_file=kmeans_targets_file,
            auto_generate_kmeans=True if kmeans_targets_file is None else False,
            vocab_size=504
        )
        
        # Wrap with tracker
        wrapped_dataset = RealDatasetWrapper(dataset)
        
        # Create DataLoader
        real_loader = DataLoader(
            wrapped_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        logger.info(f"Real dataset size: {len(wrapped_dataset)}")
        
        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(real_loader, desc="Testing real dataset")):
            batch_count += 1
            
            if isinstance(batch, dict) and batch.get('empty_batch', False):
                logger.warning(f"Real dataset batch {batch_idx}: Empty batch")
                continue
            
            valid_samples = batch.get('valid_samples', 0)
            total_samples = batch.get('total_samples', 0)
            
            if valid_samples < total_samples:
                logger.info(f"Real dataset batch {batch_idx}: {valid_samples}/{total_samples} valid samples")
            
            # Only test first few batches to avoid long runtime
            if batch_count >= 5:
                break
        
        # Print statistics
        stats = wrapped_dataset.get_stats()
        logger.info(f"\nREAL DATASET STATISTICS:")
        logger.info(f"Total items accessed: {stats['total_accessed']}")
        logger.info(f"None returns: {stats['none_count']}")
        logger.info(f"Error returns: {stats['error_count']}")
        
        if stats['none_indices']:
            logger.info(f"First None indices: {stats['none_indices']}")
        if stats['error_indices']:
            logger.info(f"First error indices: {stats['error_indices']}")
    
    except Exception as e:
        logger.error(f"Error testing real dataset: {e}")
        logger.error(traceback.format_exc())


def main():
    """Main test function"""
    logger.info("Starting collate function tests...")
    
    # Test different scenarios
    test_collate_scenarios()
    
    # Test with synthetic dataset
    test_dataloader_with_test_dataset()
    
    # Test with real dataset
    test_real_dataset()
    
    logger.info("\nAll tests completed! Check collate_test.log for detailed logs.")


if __name__ == "__main__":
    main()