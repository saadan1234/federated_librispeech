#!/usr/bin/env python3
"""
LibriSpeech Dataset Partitioner for Flower Federated Learning Simulation
Optimized for HuBERT training with speaker-based non-IID partitioning
"""

import os
import json
import shutil
import tarfile
import random
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import yaml
import soundfile as sf
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
from functools import partial
import time

# Setup logging - create directory first
log_dir = Path('/home/saadan/scratch/federated_librispeech/src/logs/partition_logs')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'partitioning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_optimal_worker_count() -> int:
    """Get optimal number of workers based on CPU cores and memory"""
    # Get CPU count
    cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
    logical_cores = psutil.cpu_count(logical=True)  # Logical cores
    
    # Get available memory in GB
    memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative approach: use 80% of physical cores but limit by memory
    # Assume each worker needs ~2GB for audio processing
    memory_limited_workers = max(1, int(memory_gb / 2))
    cpu_limited_workers = max(1, int(cpu_cores * 0.8))
    
    optimal_workers = min(memory_limited_workers, cpu_limited_workers, logical_cores)
    
    logger.info(f"System resources: {cpu_cores} physical cores, {logical_cores} logical cores, {memory_gb:.1f}GB memory")
    logger.info(f"Optimal worker count: {optimal_workers}")
    
    return optimal_workers

def process_audio_file(audio_file: Path) -> Optional[Dict[str, Any]]:
    """Process a single audio file and return its metadata"""
    try:
        with sf.SoundFile(audio_file) as f:
            duration = len(f) / f.samplerate
        
        return {
            'path': str(audio_file),
            'utterance_id': audio_file.stem,
            'duration': duration
        }
    except Exception as e:
        logger.warning(f"Could not process {audio_file}: {e}")
        return None

def process_chapter_directory(chapter_args: Tuple[Path, str, str]) -> Optional[Dict[str, Any]]:
    """Process a single chapter directory and return chapter info"""
    chapter_dir, speaker_id, chapter_id = chapter_args
    
    try:
        chapter_info = {
            'chapter_id': chapter_id,
            'audio_files': [],
            'transcript_file': None,
            'total_duration': 0.0
        }
        
        # Find transcript file
        trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
        if trans_file.exists():
            chapter_info['transcript_file'] = str(trans_file)
        
        # Process all audio files in this chapter
        audio_files = list(chapter_dir.glob("*.flac"))
        
        # Use ThreadPoolExecutor for I/O bound audio file processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            audio_results = list(executor.map(process_audio_file, audio_files))
        
        # Filter out None results and add to chapter info
        for audio_info in audio_results:
            if audio_info is not None:
                chapter_info['audio_files'].append(audio_info)
                chapter_info['total_duration'] += audio_info['duration']
        
        return chapter_info if chapter_info['audio_files'] else None
        
    except Exception as e:
        logger.warning(f"Error processing chapter {speaker_id}-{chapter_id}: {e}")
        return None

def process_speaker_directory(speaker_args: Tuple[Path, str]) -> Tuple[str, Dict[str, Any]]:
    """Process a single speaker directory and return speaker data"""
    speaker_dir, speaker_id = speaker_args
    
    speaker_data = {
        'chapters': [],
        'total_files': 0,
        'total_duration': 0.0,
        'audio_files': []
    }
    
    try:
        # Get all chapter directories for this speaker
        chapter_dirs = [(chapter_dir, speaker_id, chapter_dir.name) 
                       for chapter_dir in speaker_dir.iterdir() 
                       if chapter_dir.is_dir()]
        
        # Process chapters in parallel using threads (I/O bound)
        with ThreadPoolExecutor(max_workers=8) as executor:
            chapter_results = list(executor.map(process_chapter_directory, chapter_dirs))
        
        # Aggregate chapter results
        for chapter_info in chapter_results:
            if chapter_info is not None:
                speaker_data['chapters'].append(chapter_info)
                speaker_data['total_files'] += len(chapter_info['audio_files'])
                speaker_data['total_duration'] += chapter_info['total_duration']
                speaker_data['audio_files'].extend(chapter_info['audio_files'])
        
        return speaker_id, speaker_data
        
    except Exception as e:
        logger.warning(f"Error processing speaker {speaker_id}: {e}")
        return speaker_id, speaker_data

def copy_file_safe(src_dest_pair: Tuple[Path, Path], max_retries: int = 3, verify_copy: bool = True) -> bool:
    """Safely copy a file with comprehensive error handling and retry mechanism"""
    src, dest = src_dest_pair
    
    # Convert to Path objects if they aren't already
    src = Path(src)
    dest = Path(dest)
    
    # Pre-flight checks
    if not src.exists():
        logger.error(f"Source file does not exist: {src}")
        return False
    
    if not src.is_file():
        logger.error(f"Source is not a file: {src}")
        return False
    
    # Check available disk space (basic check)
    try:
        src_size = src.stat().st_size
        dest_free_space = shutil.disk_usage(dest.parent).free if dest.parent.exists() else shutil.disk_usage(dest.parent.parent).free
        
        if src_size > dest_free_space:
            logger.error(f"Insufficient disk space to copy {src} (need {src_size} bytes, have {dest_free_space} bytes)")
            return False
    except Exception as e:
        logger.warning(f"Could not check disk space for {dest}: {e}")
    
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            # Ensure destination directory exists with proper permissions
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            
            # If destination already exists and is identical, skip
            if dest.exists():
                if src.stat().st_size == dest.stat().st_size:
                    # Quick size check first
                    if verify_copy:
                        # For small files, do a full content comparison
                        if src_size < 1024 * 1024:  # 1MB
                            try:
                                import filecmp
                                if filecmp.cmp(src, dest, shallow=False):
                                    return True  # File already exists and is identical
                            except Exception:
                                pass  # Fall through to copy
                    else:
                        return True  # Assume it's correct if size matches
            
            # Perform the copy
            if src_size > 100 * 1024 * 1024:  # 100MB
                # For large files, use copy with buffer
                shutil.copyfile(src, dest)
                shutil.copystat(src, dest)
            else:
                # For smaller files, use copy2 (preserves metadata)
                shutil.copy2(src, dest)
            
            # Verify the copy if requested
            if verify_copy:
                if not dest.exists():
                    raise FileNotFoundError(f"Destination file was not created: {dest}")
                
                dest_size = dest.stat().st_size
                if src_size != dest_size:
                    raise ValueError(f"File size mismatch: src={src_size}, dest={dest_size}")
            
            return True
            
        except PermissionError as e:
            logger.warning(f"Permission error copying {src} to {dest} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
        except OSError as e:
            if e.errno == 28:  # No space left on device
                logger.error(f"Disk full - cannot copy {src} to {dest}: {e}")
                return False
            elif e.errno == 36:  # File name too long
                logger.error(f"Filename too long - cannot copy {src} to {dest}: {e}")
                return False
            else:
                logger.warning(f"OS error copying {src} to {dest} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    
        except Exception as e:
            logger.warning(f"Unexpected error copying {src} to {dest} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
    
    # If we get here, all retries failed
    logger.error(f"Failed to copy {src} to {dest} after {max_retries} attempts")
    return False

class FlowerLibriSpeechPartitioner:
    """
    Advanced LibriSpeech partitioner for Flower federated learning simulations
    Implements speaker-based non-IID partitioning with comprehensive validation
    """
    
    def __init__(self, config_path: str = "configs/federated_config.yaml"):
        """Initialize partitioner with configuration"""
        self.config = self._load_config(config_path)
        self.data_config = self.config['data']
        self.simulation_config = self.config['simulation']
        
        # Extract key parameters
        self.num_clients = self.simulation_config['num_supernodes']
        self.librispeech_source = Path(self.data_config['librispeech_source'])
        self.output_root = Path(self.data_config['partitioned_data_root'])
        self.dataset_subset = self.data_config['subset']
        self.validation_split = self.data_config['validation_split']
        self.test_split = self.data_config['test_split']
        self.seed = self.config['reproducibility']['seed']
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Parallel processing configuration
        self.max_workers = get_optimal_worker_count()
        self.chunk_size = max(1, self.max_workers // 2)  # Adaptive chunk size
        
        # Create output directories
        self._setup_directories()
        
        # Statistics tracking
        self.partition_stats = defaultdict(lambda: defaultdict(int))
        self.speaker_distribution = {}
        
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
            self.output_root / "global" / "validation" / "audio",
            self.output_root / "global" / "validation" / "transcripts",
            self.output_root / "global" / "test" / "audio",
            self.output_root / "global" / "test" / "transcripts",
            Path("logs/partition_logs")
        ]
        
        # Create client directories
        for i in range(self.num_clients):
            client_dir = self.output_root / f"client_{i}"
            directories.extend([
                client_dir / "train" / "audio",
                client_dir / "train" / "transcripts",
                client_dir / "validation" / "audio",
                client_dir / "validation" / "transcripts"
            ])
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure for {self.num_clients} clients")
    
    def extract_dataset(self) -> bool:
        """Extract LibriSpeech dataset if needed"""
        # Handle multiple subsets
        if isinstance(self.dataset_subset, str):
            subsets = [self.dataset_subset]
        else:
            subsets = self.dataset_subset
        
        all_extracted = True
        for subset in subsets:
            dataset_path = self.librispeech_source / subset
            
            if dataset_path.exists():
                logger.info(f"Dataset {subset} already extracted")
                continue
            
            # Try to extract from tar.gz
            tar_file = self.librispeech_source / f"{subset}.tar.gz"
            if not tar_file.exists():
                logger.error(f"Neither {dataset_path} nor {tar_file} found!")
                all_extracted = False
                continue
            
            logger.info(f"Extracting {tar_file}...")
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    # Extract with progress bar
                    members = tar.getmembers()
                    for member in tqdm(members, desc=f"Extracting {tar_file.name}"):
                        tar.extract(member, self.librispeech_source)
                
                logger.info(f"Successfully extracted {tar_file}")
                
            except Exception as e:
                logger.error(f"Failed to extract {tar_file}: {e}")
                all_extracted = False
        
        return all_extracted
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze LibriSpeech dataset structure and collect metadata with parallel processing"""
        # Handle multiple subsets
        if isinstance(self.dataset_subset, str):
            subsets = [self.dataset_subset]
        else:
            subsets = self.dataset_subset
        
        logger.info(f"Analyzing dataset structure for subsets: {subsets} using {self.max_workers} workers")
        
        speakers_data = {}
        
        # Process all subsets
        for subset in subsets:
            dataset_path = self.librispeech_source / subset
            
            if not dataset_path.exists():
                logger.warning(f"Dataset subset not found: {dataset_path}")
                continue
            
            logger.info(f"Processing subset: {subset}")
            
            # Get all speaker directories
            speaker_dirs = [(speaker_dir, speaker_dir.name) 
                          for speaker_dir in sorted(dataset_path.iterdir()) 
                          if speaker_dir.is_dir()]
            
            logger.info(f"Found {len(speaker_dirs)} speakers in {subset}")
            
            # Process speakers in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit jobs with progress tracking
                future_to_speaker = {
                    executor.submit(process_speaker_directory, speaker_args): speaker_args[1]
                    for speaker_args in speaker_dirs
                }
                
                # Collect results with progress bar
                with tqdm(total=len(speaker_dirs), desc=f"Processing speakers in {subset}") as pbar:
                    for future in as_completed(future_to_speaker):
                        speaker_id = future_to_speaker[future]
                        try:
                            speaker_id, speaker_data = future.result()
                            if speaker_data['total_files'] > 0:  # Only add speakers with files
                                if speaker_id in speakers_data:
                                    # Merge speaker data from multiple subsets
                                    existing_data = speakers_data[speaker_id]
                                    existing_data['chapters'].extend(speaker_data['chapters'])
                                    existing_data['total_files'] += speaker_data['total_files']
                                    existing_data['total_duration'] += speaker_data['total_duration']
                                    existing_data['subsets'] = existing_data.get('subsets', []) + [subset]
                                else:
                                    speaker_data['subsets'] = [subset]
                                    speakers_data[speaker_id] = speaker_data
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error processing speaker {speaker_id}: {e}")
                            pbar.update(1)
        
        # Convert to regular dict for JSON serialization
        speakers_data = dict(speakers_data)
        
        # Calculate summary statistics
        total_speakers = len(speakers_data)
        total_files = sum(data['total_files'] for data in speakers_data.values())
        total_hours = sum(data['total_duration'] for data in speakers_data.values()) / 3600
        
        summary = {
            'dataset_subset': self.dataset_subset,
            'total_speakers': total_speakers,
            'total_files': total_files,
            'total_hours': total_hours,
            'avg_files_per_speaker': total_files / total_speakers if total_speakers > 0 else 0,
            'avg_duration_per_speaker': total_hours / total_speakers if total_speakers > 0 else 0,
            'speakers_data': speakers_data
        }
        
        logger.info(f"Dataset analysis complete:")
        logger.info(f"  - {total_speakers} speakers")
        logger.info(f"  - {total_files} audio files")
        logger.info(f"  - {total_hours:.2f} hours total")
        
        return summary
    
    def load_global_dataset_speakers(self, dataset_path: str) -> List[str]:
        """Load speaker IDs from dev-clean or test-clean datasets"""
        global_path = Path(dataset_path)
        
        if not global_path.exists():
            logger.warning(f"Global dataset path not found: {dataset_path}")
            return []
        
        logger.info(f"Loading speakers from {dataset_path}")
        speakers = []
        
        # Scan for speakers in the global dataset
        for speaker_dir in global_path.iterdir():
            if speaker_dir.is_dir() and speaker_dir.name.isdigit():
                speakers.append(speaker_dir.name)
        
        logger.info(f"Found {len(speakers)} speakers in {dataset_path}")
        return sorted(speakers)
    
    def create_speaker_partitions(self, speakers_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Create balanced speaker-based partitions for federated learning
        Implements non-IID distribution based on speaker characteristics
        """
        speakers = list(speakers_data.keys())
        speaker_stats = []
        
        # Collect speaker statistics for balanced partitioning
        for speaker_id in speakers:
            data = speakers_data[speaker_id]
            speaker_stats.append({
                'speaker_id': speaker_id,
                'num_files': data['total_files'],
                'total_duration': data['total_duration'],
                'num_chapters': len(data['chapters'])
            })
        
        # Sort speakers by total content for balanced distribution
        speaker_stats.sort(key=lambda x: x['total_duration'], reverse=True)
        
        # Initialize client bins
        client_partitions = {f"client_{i}": [] for i in range(self.num_clients)}
        client_loads = {f"client_{i}": 0.0 for i in range(self.num_clients)}
        
        # Distribute speakers using a greedy algorithm for load balancing
        for speaker_stat in speaker_stats:
            # Find client with minimum current load
            min_client = min(client_loads.keys(), key=lambda x: client_loads[x])
            
            # Assign speaker to this client
            client_partitions[min_client].append(speaker_stat['speaker_id'])
            client_loads[min_client] += speaker_stat['total_duration']
        
        # Create train/validation split within each speaker
        partitions = {}
        for client_id, client_speakers in client_partitions.items():
            # Separate speakers into train and validation
            random.shuffle(client_speakers)
            
            # Calculate split point
            val_speakers_count = max(1, int(len(client_speakers) * self.validation_split))
            
            partitions[client_id] = {
                'train_speakers': client_speakers[val_speakers_count:],
                'val_speakers': client_speakers[:val_speakers_count]
            }
        
        # Create global validation and test sets from separate datasets
        global_test_speakers = []
        global_val_speakers = []

        # Load speakers from test-clean dataset
        if 'global_test_path' in self.data_config:
            global_test_path = self.data_config['global_test_path']
            if global_test_path.startswith('/'):
                # Absolute path
                global_test_speakers = self.load_global_dataset_speakers(global_test_path)
            else:
                # Relative to librispeech_source
                full_test_path = self.librispeech_source / global_test_path
                global_test_speakers = self.load_global_dataset_speakers(str(full_test_path))

        # Load speakers from dev-clean dataset  
        if 'global_validation_path' in self.data_config:
            global_val_path = self.data_config['global_validation_path']
            if global_val_path.startswith('/'):
                # Absolute path
                global_val_speakers = self.load_global_dataset_speakers(global_val_path)
            else:
                # Relative to librispeech_source
                full_val_path = self.librispeech_source / global_val_path
                global_val_speakers = self.load_global_dataset_speakers(str(full_val_path))

        # Fallback to old behavior if global datasets not found
        if not global_test_speakers or not global_val_speakers:
            logger.warning("Global datasets not found, falling back to train-clean-100 splits")
            all_speakers = list(speakers_data.keys())
            random.shuffle(all_speakers)
            
            global_test_count = max(1, int(len(all_speakers) * self.test_split))
            global_val_count = max(1, int(len(all_speakers) * self.validation_split))
            
            global_test_speakers = all_speakers[:global_test_count]
            global_val_speakers = all_speakers[global_test_count:global_test_count + global_val_count]

        partitions['global'] = {
            'test_speakers': global_test_speakers,
            'validation_speakers': global_val_speakers
        }
        
        logger.info("Speaker partitioning completed:")
        for client_id, data in partitions.items():
            if client_id != 'global':
                train_count = len(data['train_speakers'])
                val_count = len(data['val_speakers'])
                logger.info(f"  {client_id}: {train_count} train, {val_count} val speakers")
        
        logger.info(f"  Global: {len(partitions['global']['test_speakers'])} test, "
                   f"{len(partitions['global']['validation_speakers'])} validation speakers")
        
        return partitions
    
    def copy_partition_data_parallel(self, speaker_id: str, speakers_data: Dict[str, Any], 
                                   dest_audio_dir: Path, dest_transcript_dir: Path,
                                   split_type: str) -> List[Dict[str, Any]]:
        """Copy audio and transcript data for a specific speaker using parallel processing"""
        
        if speaker_id not in speakers_data:
            logger.warning(f"Speaker {speaker_id} not found in speakers_data")
            return []
        
        speaker_data = speakers_data[speaker_id]
        manifest_entries = []
        
        # Ensure destination directories exist
        dest_audio_dir.mkdir(parents=True, exist_ok=True)
        dest_transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all file operations
        copy_operations = []
        transcript_operations = []
        
        for chapter_info in speaker_data['chapters']:
            chapter_id = chapter_info['chapter_id']
            
            # Prepare transcript copy operation
            if chapter_info['transcript_file']:
                transcript_src = Path(chapter_info['transcript_file'])
                transcript_dest = dest_transcript_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                transcript_operations.append((transcript_src, transcript_dest))
            
            # Prepare audio copy operations
            for audio_info in chapter_info['audio_files']:
                audio_src = Path(audio_info['path'])
                audio_filename = f"{speaker_id}-{chapter_id}-{audio_info['utterance_id']}.flac"
                audio_dest = dest_audio_dir / audio_filename
                copy_operations.append((audio_src, audio_dest))
                
                # Create manifest entry
                manifest_entries.append({
                    'audio_path': f"{split_type}/audio/{audio_filename}",
                    'transcript_path': f"{split_type}/transcripts/{speaker_id}-{chapter_id}.trans.txt",
                    'speaker_id': speaker_id,
                    'chapter_id': chapter_id,
                    'utterance_id': audio_info['utterance_id'],
                    'duration': audio_info['duration'],
                    'split': split_type
                })
        
        # Perform parallel file operations with enhanced error handling
        all_operations = transcript_operations + copy_operations
        
        if all_operations:
            logger.debug(f"Processing {len(all_operations)} file operations for speaker {speaker_id}")
            
            # Use enhanced copy function with retries and verification
            with ThreadPoolExecutor(max_workers=min(16, len(all_operations))) as executor:
                # For critical files, enable verification; for performance, use faster copy for audio
                enhanced_operations = []
                for i, (src, dest) in enumerate(all_operations):
                    # Enable verification based on settings and file type
                    verify = getattr(self, 'verify_copies', False) or str(dest).endswith('.txt')
                    max_retries = getattr(self, 'max_retries', 3)
                    enhanced_operations.append((src, dest, max_retries, verify))
                
                # Use enhanced copy function with proper function to avoid lambda issues
                def enhanced_copy_task(args):
                    src, dest, max_retries, verify_copy = args
                    return copy_file_safe((src, dest), max_retries, verify_copy)
                
                copy_results = list(executor.map(enhanced_copy_task, enhanced_operations))
                
            failed_copies = sum(1 for result in copy_results if not result)
            success_rate = (len(copy_results) - failed_copies) / len(copy_results) * 100 if copy_results else 0
            
            if failed_copies > 0:
                logger.error(f"Failed to copy {failed_copies}/{len(all_operations)} files for speaker {speaker_id} "
                           f"(success rate: {success_rate:.1f}%)")
                
                # If too many failures, this might indicate a serious problem
                if success_rate < 80:
                    logger.error(f"Critical: High failure rate ({100-success_rate:.1f}%) for speaker {speaker_id}. "
                               f"Check disk space, permissions, and file integrity.")
            else:
                logger.debug(f"Successfully copied all {len(all_operations)} files for speaker {speaker_id}")
        
        return manifest_entries

    def copy_partition_data(self, speaker_id: str, speakers_data: Dict[str, Any], 
                          dest_audio_dir: Path, dest_transcript_dir: Path,
                          split_type: str) -> List[Dict[str, Any]]:
        """Copy audio and transcript data for a specific speaker"""
        
        if speaker_id not in speakers_data:
            logger.warning(f"Speaker {speaker_id} not found in speakers_data")
            return []
        
        speaker_data = speakers_data[speaker_id]
        manifest_entries = []
        
        for chapter_info in speaker_data['chapters']:
            chapter_id = chapter_info['chapter_id']
            
            # Copy transcript file
            if chapter_info['transcript_file']:
                transcript_src = Path(chapter_info['transcript_file'])
                transcript_dest = dest_transcript_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                
                try:
                    shutil.copy2(transcript_src, transcript_dest)
                except Exception as e:
                    logger.warning(f"Failed to copy transcript {transcript_src}: {e}")
            
            # Copy audio files
            for audio_info in chapter_info['audio_files']:
                audio_src = Path(audio_info['path'])
                
                # Create unique filename to avoid conflicts
                audio_filename = f"{speaker_id}-{chapter_id}-{audio_info['utterance_id']}.flac"
                audio_dest = dest_audio_dir / audio_filename
                
                try:
                    shutil.copy2(audio_src, audio_dest)
                    
                    # Create manifest entry
                    manifest_entries.append({
                        'audio_path': f"{split_type}/audio/{audio_filename}",
                        'transcript_path': f"{split_type}/transcripts/{speaker_id}-{chapter_id}.trans.txt",
                        'speaker_id': speaker_id,
                        'chapter_id': chapter_id,
                        'utterance_id': audio_info['utterance_id'],
                        'duration': audio_info['duration'],
                        'split': split_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to copy audio {audio_src}: {e}")
        
        return manifest_entries
    
    def copy_global_dataset_speaker(self, speaker_id: str, dataset_type: str, 
                                   dest_audio_dir: Path, dest_transcript_dir: Path) -> List[Dict[str, Any]]:
        """Copy data from global datasets (dev-clean/test-clean) for a specific speaker"""
        
        # Determine source path based on dataset type
        if dataset_type == "validation" and 'global_validation_path' in self.data_config:
            source_path = self.data_config['global_validation_path']
        elif dataset_type == "test" and 'global_test_path' in self.data_config:
            source_path = self.data_config['global_test_path']
        else:
            logger.warning(f"No global {dataset_type} path configured")
            return []
        
        # Handle relative vs absolute paths
        if source_path.startswith('/'):
            global_dataset_path = Path(source_path)
        else:
            global_dataset_path = self.librispeech_source / source_path
        
        speaker_dir = global_dataset_path / speaker_id
        
        if not speaker_dir.exists():
            logger.warning(f"Speaker {speaker_id} not found in {global_dataset_path}")
            return []
        
        # Ensure destination directories exist and are absolute paths
        dest_audio_dir = dest_audio_dir.resolve()
        dest_transcript_dir = dest_transcript_dir.resolve()
        dest_audio_dir.mkdir(parents=True, exist_ok=True)
        dest_transcript_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_entries = []
        
        # Process each chapter for this speaker
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
                
            chapter_id = chapter_dir.name
            
            # Copy transcript file
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if trans_file.exists():
                transcript_dest = dest_transcript_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                try:
                    shutil.copy2(trans_file, transcript_dest)
                except Exception as e:
                    logger.warning(f"Failed to copy transcript {trans_file}: {e}")
            
            # Copy audio files
            for audio_file in chapter_dir.glob("*.flac"):
                try:
                    # Get audio duration
                    with sf.SoundFile(audio_file) as f:
                        duration = len(f) / f.samplerate
                    
                    # Create unique filename
                    audio_filename = f"{speaker_id}-{chapter_id}-{audio_file.stem}.flac"
                    audio_dest = dest_audio_dir / audio_filename
                    
                    shutil.copy2(audio_file, audio_dest)
                    
                    # Create manifest entry
                    manifest_entries.append({
                        'audio_path': f"{dataset_type}/audio/{audio_filename}",
                        'transcript_path': f"{dataset_type}/transcripts/{speaker_id}-{chapter_id}.trans.txt",
                        'speaker_id': speaker_id,
                        'chapter_id': chapter_id,
                        'utterance_id': audio_file.stem,
                        'duration': duration,
                        'split': dataset_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to copy audio {audio_file}: {e}")
        
        return manifest_entries
    
    def create_client_data(self, client_id: str, partition_data: Dict[str, List[str]], 
                          speakers_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data for a specific client"""
        
        client_dir = self.output_root / client_id
        client_num = int(client_id.split('_')[1])
        
        logger.info(f"Creating data for {client_id}...")
        
        all_manifest_entries = []
        client_stats = {
            'client_id': client_num,
            'train': {'speakers': 0, 'files': 0, 'hours': 0.0},
            'validation': {'speakers': 0, 'files': 0, 'hours': 0.0}
        }
        
        # Process training speakers in parallel
        train_speakers = partition_data.get('train_speakers', [])
        if train_speakers:
            logger.info(f"Processing {len(train_speakers)} training speakers for {client_id}")
            
            # Prepare arguments for parallel processing
            train_tasks = [
                (speaker_id, speakers_data, client_dir / "train" / "audio", 
                 client_dir / "train" / "transcripts", "train")
                for speaker_id in train_speakers
            ]
            
            # Process in parallel using ThreadPoolExecutor (I/O bound)
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(train_speakers))) as executor:
                # Use a proper function instead of lambda to avoid multiprocessing issues
                def process_speaker_task(args):
                    return self.copy_partition_data_parallel(*args)
                
                train_results = list(tqdm(
                    executor.map(process_speaker_task, train_tasks),
                    total=len(train_tasks),
                    desc=f"{client_id} train data"
                ))
            
            # Aggregate results
            for manifest_entries in train_results:
                all_manifest_entries.extend(manifest_entries)
            
            # Update statistics
            for speaker_id in train_speakers:
                if speaker_id in speakers_data:
                    client_stats['train']['files'] += speakers_data[speaker_id]['total_files']
                    client_stats['train']['hours'] += speakers_data[speaker_id]['total_duration'] / 3600
        
        client_stats['train']['speakers'] = len(train_speakers)
        
        # Process validation speakers in parallel
        val_speakers = partition_data.get('val_speakers', [])
        if val_speakers:
            logger.info(f"Processing {len(val_speakers)} validation speakers for {client_id}")
            
            # Prepare arguments for parallel processing
            val_tasks = [
                (speaker_id, speakers_data, client_dir / "validation" / "audio",
                 client_dir / "validation" / "transcripts", "validation")
                for speaker_id in val_speakers
            ]
            
            # Process in parallel using ThreadPoolExecutor (I/O bound)
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(val_speakers))) as executor:
                # Use a proper function instead of lambda to avoid multiprocessing issues
                def process_speaker_task_val(args):
                    return self.copy_partition_data_parallel(*args)
                
                val_results = list(tqdm(
                    executor.map(process_speaker_task_val, val_tasks),
                    total=len(val_tasks),
                    desc=f"{client_id} validation data"
                ))
            
            # Aggregate results
            for manifest_entries in val_results:
                all_manifest_entries.extend(manifest_entries)
            
            # Update statistics
            for speaker_id in val_speakers:
                if speaker_id in speakers_data:
                    client_stats['validation']['files'] += speakers_data[speaker_id]['total_files']
                    client_stats['validation']['hours'] += speakers_data[speaker_id]['total_duration'] / 3600
        
        client_stats['validation']['speakers'] = len(val_speakers)
        
        # Save manifest
        if all_manifest_entries:
            manifest_df = pd.DataFrame(all_manifest_entries)
            manifest_file = client_dir / "manifest.csv"
            manifest_df.to_csv(manifest_file, index=False)
            
            logger.info(f"Saved {len(all_manifest_entries)} entries to {manifest_file}")
        
        # Save client statistics
        stats_file = client_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(client_stats, f, indent=2)
        
        logger.info(f"{client_id} created: "
                   f"{client_stats['train']['files']} train files, "
                   f"{client_stats['validation']['files']} val files")
        
        return client_stats
    
    def create_global_datasets(self, partitions: Dict[str, Any], speakers_data: Dict[str, Any]):
        """Create global validation and test datasets from separate LibriSpeech datasets"""
        
        global_data = partitions['global']
        global_dir = self.output_root / "global"
        
        # Create global validation set
        logger.info("Creating global validation dataset...")
        val_manifest_entries = []
        
        # Prepare tasks for parallel processing of global validation speakers
        val_speakers = global_data['validation_speakers']
        logger.info(f"Processing {len(val_speakers)} validation speakers for global dataset")
        
        # Separate speakers that exist in current data vs those that need to be loaded from dev-clean
        internal_val_tasks = []
        external_val_tasks = []
        
        for speaker_id in val_speakers:
            if speaker_id in speakers_data:
                internal_val_tasks.append((
                    speaker_id, speakers_data,
                    global_dir / "validation" / "audio",
                    global_dir / "validation" / "transcripts",
                    "validation"
                ))
            else:
                external_val_tasks.append((
                    speaker_id, "validation",
                    global_dir / "validation" / "audio",
                    global_dir / "validation" / "transcripts"
                ))
        
        # Process internal speakers (from training data) in parallel
        if internal_val_tasks:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(internal_val_tasks))) as executor:
                def process_internal_val_task(args):
                    return self.copy_partition_data_parallel(*args)
                
                internal_results = list(tqdm(
                    executor.map(process_internal_val_task, internal_val_tasks),
                    total=len(internal_val_tasks),
                    desc="Global validation (internal)"
                ))
                
                for manifest_entries in internal_results:
                    val_manifest_entries.extend(manifest_entries)
        
        # Process external speakers (from dev-clean) in parallel
        if external_val_tasks:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(external_val_tasks))) as executor:
                def process_external_val_task(args):
                    return self.copy_global_dataset_speaker(*args)
                
                external_results = list(tqdm(
                    executor.map(process_external_val_task, external_val_tasks),
                    total=len(external_val_tasks),
                    desc="Global validation (external)"
                ))
                
                for manifest_entries in external_results:
                    val_manifest_entries.extend(manifest_entries)
        
        if val_manifest_entries:
            val_manifest_df = pd.DataFrame(val_manifest_entries)
            val_manifest_file = global_dir / "validation_manifest.csv"
            val_manifest_df.to_csv(val_manifest_file, index=False)
        
        # Create global test set
        logger.info("Creating global test dataset...")
        test_manifest_entries = []
        
        # Prepare tasks for parallel processing of global test speakers
        test_speakers = global_data['test_speakers']
        logger.info(f"Processing {len(test_speakers)} test speakers for global dataset")
        
        # Separate speakers that exist in current data vs those that need to be loaded from test-clean
        internal_test_tasks = []
        external_test_tasks = []
        
        for speaker_id in test_speakers:
            if speaker_id in speakers_data:
                internal_test_tasks.append((
                    speaker_id, speakers_data,
                    global_dir / "test" / "audio",
                    global_dir / "test" / "transcripts",
                    "test"
                ))
            else:
                external_test_tasks.append((
                    speaker_id, "test",
                    global_dir / "test" / "audio",
                    global_dir / "test" / "transcripts"
                ))
        
        # Process internal speakers (from training data) in parallel
        if internal_test_tasks:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(internal_test_tasks))) as executor:
                def process_internal_test_task(args):
                    return self.copy_partition_data_parallel(*args)
                
                internal_results = list(tqdm(
                    executor.map(process_internal_test_task, internal_test_tasks),
                    total=len(internal_test_tasks),
                    desc="Global test (internal)"
                ))
                
                for manifest_entries in internal_results:
                    test_manifest_entries.extend(manifest_entries)
        
        # Process external speakers (from test-clean) in parallel
        if external_test_tasks:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(external_test_tasks))) as executor:
                def process_external_test_task(args):
                    return self.copy_global_dataset_speaker(*args)
                
                external_results = list(tqdm(
                    executor.map(process_external_test_task, external_test_tasks),
                    total=len(external_test_tasks),
                    desc="Global test (external)"
                ))
                
                for manifest_entries in external_results:
                    test_manifest_entries.extend(manifest_entries)
        
        if test_manifest_entries:
            test_manifest_df = pd.DataFrame(test_manifest_entries)
            test_manifest_file = global_dir / "test_manifest.csv"
            test_manifest_df.to_csv(test_manifest_file, index=False)
        
        logger.info(f"Global datasets created: "
                   f"{len(val_manifest_entries)} validation, "
                   f"{len(test_manifest_entries)} test files")
    
    def save_partition_metadata(self, partitions: Dict[str, Any], 
                               client_stats: List[Dict[str, Any]]):
        """Save comprehensive partition metadata"""
        
        metadata = {
            'partitioning_config': {
                'num_clients': self.num_clients,
                'dataset_subset': self.dataset_subset,
                'validation_split': self.validation_split,
                'test_split': self.test_split,
                'partition_method': 'speaker_based_non_iid',
                'seed': self.seed
            },
            'speaker_partitions': partitions,
            'client_statistics': client_stats,
            'global_statistics': {
                'total_train_files': sum(c['train']['files'] for c in client_stats),
                'total_val_files': sum(c['validation']['files'] for c in client_stats),
                'total_train_hours': sum(c['train']['hours'] for c in client_stats),
                'total_val_hours': sum(c['validation']['hours'] for c in client_stats),
                'avg_files_per_client': np.mean([c['train']['files'] for c in client_stats]),
                'std_files_per_client': np.std([c['train']['files'] for c in client_stats])
            }
        }
        
        # Save metadata
        metadata_file = self.output_root / "global" / "partition_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save client summary CSV
        client_df = pd.DataFrame(client_stats)
        summary_file = self.output_root / "global" / "client_summary.csv"
        client_df.to_csv(summary_file, index=False)
        
        # Create Flower-compatible speaker mapping
        speaker_mapping = {}
        for client_id, partition_data in partitions.items():
            if client_id != 'global':
                client_num = int(client_id.split('_')[1])
                for speaker in partition_data.get('train_speakers', []):
                    speaker_mapping[speaker] = client_num
        
        mapping_file = self.output_root / "global" / "speaker_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(speaker_mapping, f, indent=2)
        
        logger.info(f"Partition metadata saved to {metadata_file}")
        
        return metadata
    
    def validate_partition(self) -> bool:
        """Comprehensive validation of the created partition"""
        
        logger.info("Validating partition...")
        
        validation_passed = True
        issues = []
        
        # Check client directories and files
        for i in range(self.num_clients):
            client_dir = self.output_root / f"client_{i}"
            
            # Check directory structure
            required_dirs = [
                client_dir / "train" / "audio",
                client_dir / "train" / "transcripts",
                client_dir / "validation" / "audio",
                client_dir / "validation" / "transcripts"
            ]
            
            for req_dir in required_dirs:
                if not req_dir.exists():
                    issues.append(f"Missing directory: {req_dir}")
                    validation_passed = False
            
            # Check files exist
            manifest_file = client_dir / "manifest.csv"
            stats_file = client_dir / "stats.json"
            
            if not manifest_file.exists():
                issues.append(f"Missing manifest: {manifest_file}")
                validation_passed = False
            
            if not stats_file.exists():
                issues.append(f"Missing stats: {stats_file}")
                validation_passed = False
        
        # Check for speaker overlaps in training data
        speaker_mapping_file = self.output_root / "global" / "speaker_mapping.json"
        if speaker_mapping_file.exists():
            with open(speaker_mapping_file, 'r') as f:
                speaker_mapping = json.load(f)
            
            # Check for duplicates
            speaker_to_clients = defaultdict(list)
            for speaker, client in speaker_mapping.items():
                speaker_to_clients[speaker].append(client)
            
            overlaps = {speaker: clients for speaker, clients in speaker_to_clients.items() 
                       if len(clients) > 1}
            
            if overlaps:
                issues.append(f"Speaker overlaps found: {overlaps}")
                validation_passed = False
        
        # Log validation results
        if validation_passed:
            logger.info("✅ Partition validation passed!")
        else:
            logger.error("❌ Partition validation failed!")
            for issue in issues:
                logger.error(f"  - {issue}")
        
        return validation_passed
    
    def _validate_system_requirements(self) -> bool:
        """Validate system requirements before starting partitioning"""
        logger.info("Validating system requirements...")
        
        # Check available disk space
        try:
            free_space = shutil.disk_usage(self.output_root.parent if self.output_root.parent.exists() 
                                         else Path('.')).free
            free_space_gb = free_space / (1024**3)
            
            # Estimate required space (rough calculation)
            if isinstance(self.dataset_subset, str):
                subsets = [self.dataset_subset]
            else:
                subsets = self.dataset_subset
                
            # Updated estimates: train-clean-100 (~6GB), train-clean-360 (~23GB), train-other-500 (~30GB)
            subset_sizes = {
                "train-clean-100": 6,
                "train-clean-360": 23, 
                "train-other-500": 30
            }
            estimated_space_gb = sum(subset_sizes.get(subset, 25) for subset in subsets)
            
            if free_space_gb < estimated_space_gb:
                logger.error(f"Insufficient disk space: {free_space_gb:.1f}GB available, "
                           f"~{estimated_space_gb}GB estimated required")
                return False
            else:
                logger.info(f"Disk space check passed: {free_space_gb:.1f}GB available")
                
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Check source dataset exists
        for subset in (self.dataset_subset if isinstance(self.dataset_subset, list) else [self.dataset_subset]):
            dataset_path = self.librispeech_source / subset
            if not dataset_path.exists():
                tar_path = self.librispeech_source / f"{subset}.tar.gz"
                if not tar_path.exists():
                    logger.error(f"Neither dataset directory {dataset_path} nor tar file {tar_path} exists")
                    return False
        
        # Check write permissions
        try:
            test_file = self.output_root / "test_write_permission.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            logger.info("Write permission check passed")
        except Exception as e:
            logger.error(f"No write permission to output directory {self.output_root}: {e}")
            return False
        
        return True

    def run_partitioning(self, global_only: bool = False) -> bool:
        """Main method to run the complete partitioning process"""
        
        if global_only:
            logger.info("Running global dataset creation only...")
            return self.run_global_dataset_creation()
        
        # Validate system requirements first (unless skipped)
        if not getattr(self, 'skip_validation', False):
            if not self._validate_system_requirements():
                logger.error("System requirements validation failed. Aborting partitioning.")
                return False
        else:
            logger.warning("System requirements validation SKIPPED (--skip-validation flag used)")
        
        start_time = time.time()
        logger.info("Starting LibriSpeech partitioning for Flower simulation...")
        logger.info(f"Configuration: {self.num_clients} clients, "
                   f"subset: {self.dataset_subset}, seed: {self.seed}")
        logger.info(f"Parallel processing: {self.max_workers} workers")
        
        try:
            # Step 1: Extract dataset if needed
            step_start = time.time()
            if not self.extract_dataset():
                return False
            logger.info(f"✓ Dataset extraction completed in {time.time() - step_start:.2f}s")
            
            # Step 2: Analyze dataset structure
            step_start = time.time()
            dataset_analysis = self.analyze_dataset_structure()
            logger.info(f"✓ Dataset analysis completed in {time.time() - step_start:.2f}s")
            
            # Step 3: Create speaker partitions
            step_start = time.time()
            partitions = self.create_speaker_partitions(dataset_analysis['speakers_data'])
            logger.info(f"✓ Speaker partitioning completed in {time.time() - step_start:.2f}s")
            
            # Step 4: Create client data
            step_start = time.time()
            client_stats = []
            for i in range(self.num_clients):
                client_id = f"client_{i}"
                partition_data = partitions[client_id]
                stats = self.create_client_data(client_id, partition_data, dataset_analysis['speakers_data'])
                client_stats.append(stats)
            logger.info(f"✓ Client data creation completed in {time.time() - step_start:.2f}s")
            
            # Step 5: Create global datasets
            step_start = time.time()
            self.create_global_datasets(partitions, dataset_analysis['speakers_data'])
            logger.info(f"✓ Global datasets creation completed in {time.time() - step_start:.2f}s")
            
            # Step 6: Save metadata
            step_start = time.time()
            metadata = self.save_partition_metadata(partitions, client_stats)
            logger.info(f"✓ Metadata saving completed in {time.time() - step_start:.2f}s")
            
            # Step 7: Validate partition
            step_start = time.time()
            validation_passed = self.validate_partition()
            logger.info(f"✓ Partition validation completed in {time.time() - step_start:.2f}s")
            
            if validation_passed:
                logger.info("🎉 Partitioning completed successfully!")
                logger.info(f"📊 Summary:")
                logger.info(f"  - {len(dataset_analysis['speakers_data'])} total speakers")
                logger.info(f"  - {metadata['global_statistics']['total_train_files']} total training files")
                logger.info(f"  - {metadata['global_statistics']['total_train_hours']:.2f} total training hours")
                logger.info(f"  - Average {metadata['global_statistics']['avg_files_per_client']:.1f} files per client")
                total_time = time.time() - start_time
                logger.info(f"🕒 Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
                return True
            else:
                logger.error("Partitioning validation failed!")
                return False
                
        except Exception as e:
            logger.error(f"Partitioning failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_global_dataset_creation(self) -> bool:
        """Create only global datasets using existing partition metadata"""
        
        try:
            # Load existing partition metadata
            metadata_file = self.output_root / "global" / "partition_metadata.json"
            if not metadata_file.exists():
                logger.error(f"No existing partition metadata found at {metadata_file}")
                logger.error("Run full partitioning first or provide partition metadata")
                return False
            
            logger.info(f"Loading existing partition metadata from {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            partitions = metadata['speaker_partitions']
            
            # Step 1: Extract global datasets if needed
            if not self.extract_dataset():
                return False
            
            # Step 2: Analyze only the training dataset for speaker data
            dataset_analysis = self.analyze_dataset_structure()
            
            # Step 3: Clear and recreate global dataset directories
            global_test_dir = self.output_root / "global" / "test"
            global_val_dir = self.output_root / "global" / "validation"
            
            # Clear existing global data
            if global_test_dir.exists():
                shutil.rmtree(global_test_dir)
            if global_val_dir.exists():
                shutil.rmtree(global_val_dir)
            
            # Recreate directories
            (global_test_dir / "audio").mkdir(parents=True, exist_ok=True)
            (global_test_dir / "transcripts").mkdir(parents=True, exist_ok=True)
            (global_val_dir / "audio").mkdir(parents=True, exist_ok=True)
            (global_val_dir / "transcripts").mkdir(parents=True, exist_ok=True)
            
            # Step 4: Create global datasets
            logger.info("Creating global datasets with fixed paths...")
            self.create_global_datasets(partitions, dataset_analysis['speakers_data'])
            
            logger.info("✅ Global dataset creation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Global dataset creation failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Partition LibriSpeech dataset for Flower federated learning simulation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/federated_config.yaml",
        help="Path to federated learning configuration file"
    )
    parser.add_argument(
        "--librispeech-path",
        type=str,
        help="Override LibriSpeech source path from config"
    )
    parser.add_argument(
        "--subset",
        type=str,
        nargs='+',
        choices=["train-clean-100", "train-clean-360", "train-other-500"],
        help="Override dataset subset(s) from config (can specify multiple)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        help="Override number of clients from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed from config"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-partitioning even if data exists"
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Only recreate global datasets using existing partition metadata"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override maximum number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--disable-parallel",
        action="store_true",
        help="Disable parallel processing (use single-threaded mode)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip system requirements validation (use with caution)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed file operations (default: 3)"
    )
    parser.add_argument(
        "--verify-copies",
        action="store_true",
        help="Enable file copy verification (slower but more reliable)"
    )
    
    args = parser.parse_args()
    
    # Create partitioner
    partitioner = FlowerLibriSpeechPartitioner(args.config)
    
    # Override parallel processing settings
    if args.disable_parallel:
        partitioner.max_workers = 1
        logger.info("Parallel processing disabled - using single-threaded mode")
    elif args.max_workers:
        partitioner.max_workers = args.max_workers
        logger.info(f"Using {args.max_workers} workers (user override)")
    
    # Override config parameters if provided
    if args.librispeech_path:
        partitioner.librispeech_source = Path(args.librispeech_path)
        partitioner.data_config['librispeech_source'] = args.librispeech_path
    
    if args.subset:
        partitioner.dataset_subset = args.subset
        partitioner.data_config['subset'] = args.subset
    
    if args.num_clients:
        partitioner.num_clients = args.num_clients
        partitioner.simulation_config['num_supernodes'] = args.num_clients
        # Recreate directories for new client count
        partitioner._setup_directories()
    
    if args.seed:
        partitioner.seed = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Configure error handling settings
    partitioner.skip_validation = args.skip_validation
    partitioner.max_retries = args.max_retries
    partitioner.verify_copies = args.verify_copies
    
    # Check if partitioning already exists (skip for global-only mode)
    if not args.force and not args.global_only:
        existing_metadata = partitioner.output_root / "global" / "partition_metadata.json"
        if existing_metadata.exists():
            logger.info(f"Partition already exists at {partitioner.output_root}")
            logger.info("Use --force to re-partition or --global-only to recreate global datasets")
            return
    
    # Run partitioning
    success = partitioner.run_partitioning(global_only=args.global_only)
    
    if success:
        logger.info("✅ Partitioning completed successfully!")
        logger.info(f"📁 Partitioned data available at: {partitioner.output_root}")
        logger.info("🚀 Ready for Flower simulation!")
    else:
        logger.error("❌ Partitioning failed!")
        exit(1)


if __name__ == "__main__":
    main()