#!/usr/bin/env python3
"""
Test script to load actual federated LibriSpeech partition data and identify problematic files.
"""

import torch
import pandas as pd
import logging
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import traceback
import librosa
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('partition_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Define the custom collate function (from federated_hubert_distillation.py)
def custom_collate_fn(batch):
    """Custom collate function to handle None values in the batch."""
    import logging
    
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    # Log filtering statistics
    total_samples = len(batch)
    valid_samples = len(valid_batch)
    filtered_samples = total_samples - valid_samples
    
    if filtered_samples > 0:
        logging.info(f"Batch filtering: {valid_samples}/{total_samples} valid samples, filtered out {filtered_samples} None items")

    if not valid_batch:
        # If all items are None, return an empty batch structure that won't cause errors
        logging.warning(f"All {total_samples} items in batch were None, returning empty batch structure")
        # Return a minimal batch structure that the training loop can detect and skip
        return {'empty_batch': True, 'valid_samples': 0, 'total_samples': total_samples}

    # Use the default collate function for valid items
    from torch.utils.data._utils.collate import default_collate
    collated_batch = default_collate(valid_batch)
    
    # Add metadata about batch composition
    collated_batch['valid_samples'] = valid_samples
    collated_batch['total_samples'] = total_samples
    
    return collated_batch


class PartitionDataset(Dataset):
    """Dataset to load actual partition data and track problematic files"""
    
    def __init__(self, manifest_path, data_root, max_length=16000, sample_rate=16000):
        self.manifest_path = manifest_path
        self.data_root = Path(data_root)
        self.max_length = max_length
        self.sample_rate = sample_rate
        
        # Load manifest
        self.manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(self.manifest_df)} entries from {manifest_path}")
        
        # Track problematic files
        self.none_indices = []
        self.error_indices = []
        self.error_details = {}
        self.access_count = 0
        
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        self.access_count += 1
        
        try:
            row = self.manifest_df.iloc[idx]
            audio_path = self.data_root / row['audio_path']
            
            # Check if file exists
            if not audio_path.exists():
                error_msg = f"Audio file does not exist: {audio_path}"
                logger.warning(f"Index {idx}: {error_msg}")
                self.error_indices.append(idx)
                self.error_details[idx] = error_msg
                return None
            
            # Check file size
            if audio_path.stat().st_size == 0:
                error_msg = f"Audio file is empty: {audio_path}"
                logger.warning(f"Index {idx}: {error_msg}")
                self.error_indices.append(idx)
                self.error_details[idx] = error_msg
                return None
            
            # Try to load audio
            try:
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                
                # Check if audio is valid
                if len(audio) == 0:
                    error_msg = f"Loaded audio is empty: {audio_path}"
                    logger.warning(f"Index {idx}: {error_msg}")
                    self.error_indices.append(idx)
                    self.error_details[idx] = error_msg
                    return None
                
                # Pad or truncate to max_length
                if len(audio) > self.max_length:
                    audio = audio[:self.max_length]
                else:
                    audio = np.pad(audio, (0, self.max_length - len(audio)))
                
                # Create attention mask
                attention_mask = np.ones(len(audio))
                
                return {
                    'input_values': torch.FloatTensor(audio),
                    'attention_mask': torch.FloatTensor(attention_mask),
                    'audio_path': str(audio_path),
                    'speaker_id': row['speaker_id'],
                    'duration': row['duration'],
                    'index': idx
                }
                
            except Exception as e:
                error_msg = f"Error loading audio {audio_path}: {str(e)}"
                logger.error(f"Index {idx}: {error_msg}")
                self.error_indices.append(idx)
                self.error_details[idx] = error_msg
                return None
                
        except Exception as e:
            error_msg = f"General error processing index {idx}: {str(e)}"
            logger.error(error_msg)
            self.error_indices.append(idx)
            self.error_details[idx] = error_msg
            return None
    
    def get_stats(self):
        return {
            'total_accessed': self.access_count,
            'none_count': len(self.none_indices),
            'error_count': len(self.error_indices),
            'none_indices': self.none_indices,
            'error_indices': self.error_indices,
            'error_details': self.error_details
        }


def test_partition(client_id, max_samples=50):
    """Test a specific client partition"""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING CLIENT {client_id} PARTITION")
    logger.info(f"{'='*60}")
    
    manifest_path = f"/lustre07/scratch/saadan/federated_librispeech/src/federated_librispeech/data/client_{client_id}/distill_manifest.csv"
    data_root = f"/lustre07/scratch/saadan/federated_librispeech/src/federated_librispeech/data/client_{client_id}"
    
    if not Path(manifest_path).exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    # Create dataset
    dataset = PartitionDataset(manifest_path, data_root, max_length=16000)
    
    # Limit to max_samples for testing
    if len(dataset) > max_samples:
        logger.info(f"Limiting test to first {max_samples} samples (dataset has {len(dataset)} total)")
        test_indices = list(range(max_samples))
    else:
        test_indices = list(range(len(dataset)))
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # Test loading
    batch_count = 0
    total_valid_samples = 0
    total_samples = 0
    empty_batches = 0
    
    logger.info(f"Starting to test {len(test_indices)} samples...")
    
    # Test individual samples first to identify None returns
    none_samples = []
    for i in tqdm(test_indices[:20], desc=f"Testing individual samples for client {client_id}"):
        try:
            sample = dataset[i]
            if sample is None:
                none_samples.append(i)
                logger.warning(f"Sample {i} returned None")
        except Exception as e:
            logger.error(f"Error accessing sample {i}: {e}")
            none_samples.append(i)
    
    logger.info(f"Found {len(none_samples)} None samples in first 20: {none_samples}")
    
    # Test with DataLoader
    try:
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing batches for client {client_id}")):
            batch_count += 1
            
            if isinstance(batch, dict) and batch.get('empty_batch', False):
                empty_batches += 1
                logger.warning(f"Client {client_id} Batch {batch_idx}: Empty batch")
                continue
            
            valid_samples = batch.get('valid_samples', 0)
            batch_total = batch.get('total_samples', 0)
            total_valid_samples += valid_samples
            total_samples += batch_total
            
            if batch_count <= 5:  # Log details for first few batches
                logger.info(f"Client {client_id} Batch {batch_idx}: {valid_samples}/{batch_total} valid samples")
                if 'input_values' in batch:
                    logger.info(f"  Input values shape: {batch['input_values'].shape}")
            
            # Stop after processing a reasonable number of batches
            if batch_count >= 10:
                break
                
    except Exception as e:
        logger.error(f"Error during DataLoader processing: {e}")
        logger.error(traceback.format_exc())
    
    # Get statistics
    stats = dataset.get_stats()
    
    logger.info(f"\nCLIENT {client_id} RESULTS:")
    logger.info(f"Batches processed: {batch_count}")
    logger.info(f"Empty batches: {empty_batches}")
    logger.info(f"Total valid samples: {total_valid_samples}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Dataset access count: {stats['total_accessed']}")
    logger.info(f"None returns: {stats['none_count']}")
    logger.info(f"Error returns: {stats['error_count']}")
    
    if stats['error_indices']:
        logger.info(f"First 5 error indices: {stats['error_indices'][:5]}")
        logger.info("Error details for first few indices:")
        for idx in stats['error_indices'][:3]:
            logger.info(f"  Index {idx}: {stats['error_details'].get(idx, 'No details')}")
    
    return stats


def main():
    """Main test function"""
    logger.info("Starting partition data testing...")
    
    # Test each client partition
    all_stats = {}
    for client_id in range(4):  # Assuming clients 0-3
        try:
            stats = test_partition(client_id, max_samples=50)
            all_stats[client_id] = stats
        except Exception as e:
            logger.error(f"Failed to test client {client_id}: {e}")
            logger.error(traceback.format_exc())
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*60}")
    
    total_errors = 0
    total_accessed = 0
    
    for client_id, stats in all_stats.items():
        if stats:
            total_errors += stats['error_count']
            total_accessed += stats['total_accessed']
            logger.info(f"Client {client_id}: {stats['error_count']}/{stats['total_accessed']} errors")
    
    if total_accessed > 0:
        error_rate = (total_errors / total_accessed) * 100
        logger.info(f"Overall error rate: {error_rate:.2f}% ({total_errors}/{total_accessed})")
    
    logger.info("Testing completed! Check partition_test.log for detailed logs.")


if __name__ == "__main__":
    main()