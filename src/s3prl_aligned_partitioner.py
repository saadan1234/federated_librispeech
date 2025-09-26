#!/usr/bin/env python3
"""
S3PRL-Aligned LibriSpeech Partitioner for Federated Learning
Uses S3PRL components for data processing consistency
"""

import sys
import os
from pathlib import Path

# Add S3PRL to path
sys.path.append(str(Path(__file__).parent.parent / 's3prl'))

# S3PRL imports with fallback
try:
    from s3prl.dataio.corpus.librispeech import LibriSpeech, read_text, _parse_spk_to_gender
    from s3prl.pretrain.bucket_dataset import FeatDataset, WaveDataset
    S3PRL_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"S3PRL not available: {e}. Using fallback implementations.")
    S3PRL_AVAILABLE = False

import numpy as np
import random
from typing import Dict, List, Any, Optional
import yaml
import json
import logging
from collections import defaultdict
import time

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor
import shutil

logger = logging.getLogger(__name__)

# Fallback implementations when S3PRL is not available
class FallbackLibriSpeech:
    """Fallback LibriSpeech processor when S3PRL is not available"""
    
    def __init__(self, dataset_root, n_jobs=4, train_split=None, valid_split=None, test_split=None):
        self.dataset_root = Path(dataset_root)
        self.train_split = train_split or []
        self.valid_split = valid_split or []
        self.test_split = test_split or []
        self.train = {}
        
        # Process training splits
        for split in self.train_split:
            split_data = self._process_split(split)
            self.train.update(split_data)
            
    def _process_split(self, split_name):
        """Process a single split directory"""
        split_dir = self.dataset_root / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return {}
            
        # Find all audio files
        audio_files = list(split_dir.rglob("*.flac"))
        split_data = {}
        
        for audio_file in audio_files:
            # Extract metadata from filename (speaker-chapter-utterance.flac)
            stem = audio_file.stem
            parts = stem.split('-')
            
            if len(parts) >= 2:
                speaker_id = int(parts[0])
                chapter_id = parts[1]
                
                # Load transcript
                transcript = self._load_transcript(audio_file)
                
                split_data[stem] = {
                    'wav_path': audio_file,
                    'transcription': transcript,
                    'speaker': speaker_id,
                    'chapter': chapter_id,
                    'corpus_split': split_name
                }
                
        return split_data
        
    def _load_transcript(self, wav_path):
        """Load transcript for an audio file"""
        transcript_file = wav_path.parent / f"{wav_path.parent.name}.trans.txt"
        utterance_id = wav_path.stem
        
        try:
            with open(transcript_file, 'r') as f:
                for line in f:
                    if line.startswith(utterance_id):
                        return line.strip().split(' ', 1)[1]
        except FileNotFoundError:
            pass
        return ""

def create_csv_manifest(data, output_file):
    """Create CSV manifest with or without pandas"""
    if PANDAS_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
    else:
        # Fallback CSV writing
        if not data:
            return
            
        with open(output_file, 'w') as f:
            # Write header
            keys = data[0].keys()
            f.write(','.join(keys) + '\n')
            
            # Write data
            for row in data:
                values = [str(row.get(key, '')) for key in keys]
                f.write(','.join(values) + '\n')

class S3PRLAlignedPartitioner:
    """
    Federated LibriSpeech partitioner using S3PRL components for consistency
    """
    
    def __init__(self, config_path: str):
        """Initialize with S3PRL-compatible configuration"""
        self.config = self._load_config(config_path)
        self.data_config = self.config['data']
        self.simulation_config = self.config['simulation']
        
        # Core parameters
        self.num_clients = self.simulation_config['num_supernodes']
        self.librispeech_root = Path(self.data_config['librispeech_source'])
        self.output_root = Path(self.data_config['partitioned_data_root'])
        self.dataset_subsets = self.data_config['subset']
        
        # Ensure subsets is a list
        if isinstance(self.dataset_subsets, str):
            self.dataset_subsets = [self.dataset_subsets]
            
        # S3PRL-compatible parameters
        self.validation_split = self.data_config.get('validation_split', 0.1)
        self.test_split = self.data_config.get('test_split', 0.1)
        self.global_validation_path = self.data_config.get('global_validation_path', 'dev-clean')
        self.global_test_path = self.data_config.get('global_test_path', 'test-clean')
        
        # Seed handling - check multiple possible locations
        self.seed = (self.config.get('reproducibility', {}).get('seed') or 
                    self.config.get('seed') or 
                    42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize S3PRL LibriSpeech corpus
        self.corpus = None
        
        # Setup directories
        self._setup_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
            
    def _setup_directories(self):
        """Create all necessary output directories"""
        # Ensure output_root is absolute
        self.output_root = self.output_root.resolve()

        directories = [
            self.output_root,
            self.output_root / "global" / "validation",
            self.output_root / "global" / "test",
        ]

        # Create client directories
        for i in range(self.num_clients):
            client_dir = self.output_root / f"client_{i}"
            directories.extend([
                client_dir,
                client_dir / "len_for_bucket"  # S3PRL bucket dataset directory
            ])

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure for {self.num_clients} clients")
            
    def initialize_s3prl_corpus(self):
        """Initialize S3PRL LibriSpeech corpus for consistent data handling"""
        logger.info("Initializing LibriSpeech corpus...")
        
        if S3PRL_AVAILABLE:
            # Use S3PRL's LibriSpeech class for consistent data processing
            self.corpus = LibriSpeech(
                dataset_root=str(self.librispeech_root),
                n_jobs=4,
                train_split=self.dataset_subsets,  # Use training subsets for partitioning
                valid_split=[self.global_validation_path],
                test_split=[self.global_test_path]
            )
            logger.info(f"S3PRL corpus initialized with {len(self.corpus.train)} training samples")
            return self.corpus.train
        else:
            # Use fallback implementation
            logger.info("Using fallback LibriSpeech processor")
            self.corpus = FallbackLibriSpeech(
                dataset_root=str(self.librispeech_root),
                n_jobs=4,
                train_split=self.dataset_subsets,
                valid_split=[self.global_validation_path],
                test_split=[self.global_test_path]
            )
            logger.info(f"Fallback corpus initialized with {len(self.corpus.train)} training samples")
            return self.corpus.train
        
    def extract_speaker_partitions_s3prl_style(self, train_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create speaker partitions using S3PRL data structure"""
        
        # Group samples by speaker using S3PRL's speaker extraction
        speaker_to_samples = defaultdict(list)
        speaker_stats = defaultdict(lambda: {'duration': 0.0, 'files': 0})
        
        for sample_id, sample_data in train_data.items():
            speaker_id = str(sample_data['speaker'])  # S3PRL already extracts speaker
            speaker_to_samples[speaker_id].append(sample_id)
            
            # Calculate duration for load balancing (you may need to implement this)
            # For now, use file count as proxy
            speaker_stats[speaker_id]['files'] += 1
            
        logger.info(f"Found {len(speaker_to_samples)} unique speakers")
        
        # Create balanced partitions using S3PRL-consistent approach
        return self._create_balanced_speaker_partitions(speaker_to_samples, speaker_stats)
        
    def _create_balanced_speaker_partitions(self, speaker_to_samples: Dict[str, List[str]], 
                                          speaker_stats: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Create balanced speaker partitions for federated clients"""
        
        # Sort speakers by number of files for balanced distribution
        sorted_speakers = sorted(speaker_stats.keys(), 
                               key=lambda s: speaker_stats[s]['files'], 
                               reverse=True)
        
        # Initialize client partitions
        client_partitions = {f"client_{i}": [] for i in range(self.num_clients)}
        client_loads = {f"client_{i}": 0 for i in range(self.num_clients)}
        
        # Greedy assignment for load balancing
        for speaker in sorted_speakers:
            # Find client with minimum load
            min_client = min(client_loads.keys(), key=lambda x: client_loads[x])
            
            # Assign speaker to this client
            client_partitions[min_client].append(speaker)
            client_loads[min_client] += speaker_stats[speaker]['files']
            
        # Create train/validation splits within each client
        partitions = {}
        for client_id, client_speakers in client_partitions.items():
            random.shuffle(client_speakers)
            val_count = max(1, int(len(client_speakers) * 0.1))  # 10% for validation
            
            partitions[client_id] = {
                'train_speakers': client_speakers[val_count:],
                'val_speakers': client_speakers[:val_count]
            }
            
        return partitions
        
    def generate_s3prl_compatible_manifests(self, partitions: Dict[str, Dict], 
                                          train_data: Dict[str, Any]):
        """Generate CSV manifests compatible with S3PRL bucket dataset format"""
        
        # Group samples by speaker for easy lookup
        speaker_to_samples = defaultdict(list)
        for sample_id, sample_data in train_data.items():
            speaker_id = str(sample_data['speaker'])
            speaker_to_samples[speaker_id].append((sample_id, sample_data))
            
        # Create manifests for each client
        for client_id, partition_data in partitions.items():
            if client_id == 'global':
                continue
                
            client_dir = self.output_root / client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if client already has complete manifests (resume capability)
            if self._client_has_complete_manifests(client_id):
                logger.info(f"✅ {client_id}: Skipping (already has complete manifests)")
                continue
            
            logger.info(f"🔄 Processing {client_id}...")
            
            # Generate train manifest
            train_manifest = self._create_manifest_for_speakers(
                partition_data['train_speakers'], 
                speaker_to_samples, 
                'train'
            )
            
            # Generate validation manifest  
            val_manifest = self._create_manifest_for_speakers(
                partition_data['val_speakers'], 
                speaker_to_samples, 
                'validation'
            )
            
            # Save in S3PRL bucket dataset format
            create_csv_manifest(train_manifest, client_dir / 'train.csv')
            create_csv_manifest(val_manifest, client_dir / 'validation.csv')
            
            logger.info(f"✅ {client_id}: {len(train_manifest)} train, {len(val_manifest)} val samples")
            
    def _create_manifest_for_speakers(self, speakers: List[str], 
                                    speaker_to_samples: Dict[str, List], 
                                    split: str) -> List[Dict]:
        """Create S3PRL-compatible manifest entries for given speakers"""
        
        manifest_entries = []
        
        for speaker in speakers:
            if speaker not in speaker_to_samples:
                continue
                
            for sample_id, sample_data in speaker_to_samples[speaker]:
                # Create S3PRL bucket dataset compatible entry
                wav_path = sample_data['wav_path']
                
                # Calculate relative path from LibriSpeech root
                relative_path = Path(wav_path).relative_to(self.librispeech_root)
                
                # Calculate audio length
                audio_length = self._calculate_audio_length(wav_path)
                
                manifest_entries.append({
                    'file_path': str(relative_path),  # S3PRL bucket dataset expects this column
                    'length': audio_length,  # Actual audio length in frames
                    'utterance_id': sample_id,
                    'speaker_id': speaker,
                    'transcription': sample_data['transcription'],
                    'split': split
                })
                
        return manifest_entries
        
    def _calculate_audio_length(self, wav_path: Path) -> int:
        """Calculate audio length in frames"""
        try:
            if SOUNDFILE_AVAILABLE:
                import soundfile as sf
                with sf.SoundFile(wav_path) as f:
                    return len(f)
            else:
                # Fallback: estimate based on file size (rough approximation)
                file_size = wav_path.stat().st_size
                # Rough estimate: 16kHz * 2 bytes per sample = 32000 bytes per second
                estimated_seconds = file_size / 32000
                return int(estimated_seconds * 16000)  # frames at 16kHz
        except Exception as e:
            logger.warning(f"Could not calculate length for {wav_path}: {e}")
            return 160000  # Default fallback (10 seconds at 16kHz)
    
    def _client_has_complete_manifests(self, client_id: str) -> bool:
        """Check if a client already has complete train and validation manifests"""
        client_dir = self.output_root / client_id
        
        required_files = ['train.csv', 'validation.csv']
        
        for filename in required_files:
            file_path = client_dir / filename
            if not file_path.exists():
                return False
            
            # Check if file is not empty and has more than just header
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) <= 1:  # Only header or empty
                        return False
            except Exception:
                return False
        
        return True
    
    def _client_has_complete_bucket_files(self, client_id: str) -> bool:
        """Check if a client already has complete bucket files"""
        client_dir = self.output_root / client_id
        bucket_dir = client_dir / "len_for_bucket"
        
        if not bucket_dir.exists():
            return False
        
        required_files = ['train.csv', 'validation.csv']
        
        for filename in required_files:
            bucket_file = bucket_dir / filename
            source_file = client_dir / filename
            
            if not bucket_file.exists():
                return False
            
            # Check if bucket file is up to date with source
            if source_file.exists():
                try:
                    if bucket_file.stat().st_mtime < source_file.stat().st_mtime:
                        return False
                except Exception:
                    return False
        
        return True
        
    def create_s3prl_bucket_files(self):
        """Create S3PRL-style bucket files for length-based bucketing"""
        logger.info("Generating S3PRL-compatible bucket files...")
        
        # This would call S3PRL's generate_len_for_bucket.py equivalent
        # For each partition, create len_for_bucket directory structure
        
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            client_dir = self.output_root / client_id
            bucket_dir = client_dir / "len_for_bucket"
            
            # Check if client already has complete bucket files (resume capability)
            if self._client_has_complete_bucket_files(client_id):
                logger.info(f"✅ {client_id}: Skipping bucket files (already complete)")
                continue
                
            logger.info(f"🔄 Creating bucket files for {client_id}...")
            bucket_dir.mkdir(exist_ok=True)
            
            # Copy CSV files to bucket directory (lengths already calculated)
            for split in ['train', 'validation']:
                csv_file = client_dir / f"{split}.csv"
                bucket_csv = bucket_dir / f"{split}.csv"
                if csv_file.exists():
                    shutil.copy2(csv_file, bucket_csv)
                    logger.debug(f"Copied {csv_file} to {bucket_csv}")
            
            logger.info(f"✅ {client_id}: Bucket files created")
    
    def _check_existing_progress(self) -> List[str]:
        """Check which clients already have complete data processing"""
        completed_clients = []
        
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            
            # Check if client has both complete manifests and bucket files
            if (self._client_has_complete_manifests(client_id) and 
                self._client_has_complete_bucket_files(client_id)):
                completed_clients.append(client_id)
        
        return completed_clients
                    
    def run_s3prl_aligned_partitioning(self) -> bool:
        """Main method using S3PRL components with resume capability"""
        
        try:
            logger.info("Starting S3PRL-aligned federated partitioning...")
            
            # Step 0: Validate configuration
            if not self.validate_configuration():
                return False
            
            # Check for existing progress
            completed_clients = self._check_existing_progress()
            if completed_clients:
                logger.info(f"📋 Found existing progress: {len(completed_clients)}/{self.num_clients} clients completed")
                logger.info(f"✅ Completed clients: {sorted(completed_clients)}")
                remaining_clients = [f"client_{i}" for i in range(self.num_clients) if f"client_{i}" not in completed_clients]
                if remaining_clients:
                    logger.info(f"🔄 Remaining clients: {sorted(remaining_clients)}")
                else:
                    logger.info("🎉 All clients already completed! Nothing to do.")
                    return True
            
            # Step 1: Initialize S3PRL corpus
            logger.info("Step 1: Initializing corpus...")
            train_data = self.initialize_s3prl_corpus()
            
            # Step 2: Create speaker partitions using S3PRL data structure
            logger.info("Step 2: Creating speaker partitions...")
            partitions = self.extract_speaker_partitions_s3prl_style(train_data)
            
            # Step 3: Generate S3PRL-compatible manifests (with resume capability)
            logger.info("Step 3: Generating S3PRL-compatible manifests...")
            self.generate_s3prl_compatible_manifests(partitions, train_data)
            
            # Step 4: Create bucket files for S3PRL dataset compatibility (with resume capability)
            logger.info("Step 4: Creating S3PRL bucket files...")
            self.create_s3prl_bucket_files()
            
            # Step 5: Save metadata
            logger.info("Step 5: Saving partition metadata...")
            self._save_partition_metadata(partitions)
            
            # Final progress check
            final_completed = self._check_existing_progress()
            logger.info(f"� Final status: {len(final_completed)}/{self.num_clients} clients completed")
            
            if len(final_completed) == self.num_clients:
                logger.info("�🎉 S3PRL-aligned partitioning completed successfully!")
                logger.info(f"📁 Partitioned data available at: {self.output_root}")
                logger.info(f"📊 Created {self.num_clients} federated clients")
                return True
            else:
                incomplete = [f"client_{i}" for i in range(self.num_clients) if f"client_{i}" not in final_completed]
                logger.warning(f"⚠️ Some clients may be incomplete: {sorted(incomplete)}")
                return False
            
        except Exception as e:
            logger.error(f"S3PRL-aligned partitioning failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _save_partition_metadata(self, partitions: Dict[str, Any]):
        """Save partition metadata"""
        metadata = {
            'partitioning_method': 's3prl_aligned_speaker_based',
            'num_clients': self.num_clients,
            'dataset_subsets': self.dataset_subsets,
            'speaker_partitions': partitions,
            'seed': self.seed,
            's3prl_compatible': True
        }
        
        metadata_file = self.output_root / "partition_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Partition metadata saved to {metadata_file}")
        
    def validate_configuration(self) -> bool:
        """Validate that all required paths and configurations are correct"""
        logger.info("Validating S3PRL-aligned partitioner configuration...")
        
        issues = []
        
        # Check LibriSpeech source directory
        if not self.librispeech_root.exists():
            issues.append(f"LibriSpeech source directory not found: {self.librispeech_root}")
        
        # Check for required subsets
        for subset in self.dataset_subsets:
            subset_path = self.librispeech_root / subset
            if not subset_path.exists():
                issues.append(f"Dataset subset not found: {subset_path}")
                
        # Check output directory is writable
        try:
            test_file = self.output_root / "test_write.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"Output directory not writable: {self.output_root} ({e})")
            
        # Report validation results
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ Configuration validation passed!")
            return True


def main():
    """Main entry point for S3PRL-aligned partitioner"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="S3PRL-aligned LibriSpeech partitioner for federated learning"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/pretraining_config.yaml",  # Use your actual config file
        help="Path to federated learning configuration file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running partitioning"
    )
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force overwrite existing output directory (restart from beginning)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume capability - always restart from beginning"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create partitioner
    partitioner = S3PRLAlignedPartitioner(args.config)
    
    # Check if validation only
    if args.validate_only:
        logger.info("Running configuration validation only...")
        success = partitioner.validate_configuration()
        if success:
            logger.info("✅ Configuration is valid!")
        else:
            logger.error("❌ Configuration validation failed!")
        exit(0 if success else 1)
    
    # Check if output already exists and handle resume logic
    if partitioner.output_root.exists():
        existing_files = list(partitioner.output_root.iterdir())
        if existing_files:
            if args.force:
                logger.warning(f"🗑️ Force flag specified - removing existing data: {partitioner.output_root}")
                import shutil
                shutil.rmtree(partitioner.output_root)
                logger.info("✅ Existing data removed - starting fresh")
            elif args.no_resume:
                logger.error(f"Output directory already exists with {len(existing_files)} items: {partitioner.output_root}")
                logger.error("Use --force to overwrite or remove --no-resume to enable resume capability")
                exit(1)
            else:
                # Resume mode - check for existing progress
                completed_clients = partitioner._check_existing_progress()
                if completed_clients:
                    logger.info(f"📋 Resume mode: Found {len(completed_clients)}/{partitioner.num_clients} completed clients")
                    logger.info("🔄 Will continue from where it left off...")
                else:
                    logger.info("📋 Resume mode: No completed clients found - starting fresh")
    
    # Disable resume capability if requested
    if args.no_resume:
        logger.info("🚫 Resume capability disabled - will process all clients")
        # Override the resume check methods to always return False/empty
        partitioner._client_has_complete_manifests = lambda client_id: False
        partitioner._client_has_complete_bucket_files = lambda client_id: False
    
    # Run partitioning
    success = partitioner.run_s3prl_aligned_partitioning()
    
    if success:
        logger.info("🎉 S3PRL-aligned partitioning completed!")
        logger.info("🚀 Ready for federated training with S3PRL compatibility!")
    else:
        logger.error("❌ Partitioning failed!")
        exit(1)


if __name__ == "__main__":
    main()
