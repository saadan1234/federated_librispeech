#!/usr/bin/env python3
"""
Fixed K-means clustering script for your specific directory structure
"""

import torch
import pandas as pd
import logging
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.cluster import MiniBatchKMeans
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import librosa
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kmeans_generation.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from audio for K-means clustering"""
    
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        # Auto-detect device with fallback
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        self.device = device
        logger.info(f"Using device: {device}")
        
        logger.info(f"Loading {model_name} model...")
        try:
            # Load feature extractor and model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            
            # Move to device
            self.model = self.model.to(device)
            self.model.eval()
            
            # Enable DataParallel for multi-GPU only if CUDA available
            if device == "cuda" and torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                self.model = torch.nn.DataParallel(self.model)
            
            logger.info(f"Successfully loaded {model_name} model on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_features(self, audio_path, max_length=80000):
        """Extract features from a single .flac audio file"""
        try:
            # Ensure path exists and is .flac
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return None
            
            # Load audio (librosa handles .flac automatically)
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            if len(audio) == 0:
                logger.warning(f"Empty audio file: {audio_path}")
                return None
            
            # Truncate if too long
            if len(audio) > max_length:
                audio = audio[:max_length]
            
            # Extract features using the model
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Use mixed precision only if CUDA is available
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                # Use the last hidden state as features
                features = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
                
            # Return mean-pooled features for clustering
            return features.mean(dim=0).cpu().numpy()  # [hidden_dim]
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None


def load_audio_files_from_manifest(manifest_path, audio_root):
    """Load audio file paths from manifest, handling your specific structure"""
    if not Path(manifest_path).exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return []
    
    df = pd.read_csv(manifest_path)
    audio_files = []
    
    for idx, row in df.iterrows():
        # The manifest contains relative paths like "train/audio/file.flac"
        audio_path = Path(audio_root) / row['audio_path']
        
        if audio_path.exists() and audio_path.suffix.lower() == '.flac':
            audio_files.append(str(audio_path))
        else:
            logger.warning(f"Audio file not found or not .flac: {audio_path}")
    
    logger.info(f"Found {len(audio_files)} valid .flac files in {manifest_path}")
    return audio_files


def collect_features_from_client(data_root, client_id, max_samples=None):
    """Collect features from a specific client using your directory structure"""
    logger.info(f"Collecting features from client_{client_id}")
    
    client_path = Path(data_root) / f"client_{client_id}"
    
    # Check which manifest to use (prioritize distill_manifest.csv)
    manifest_candidates = [
        "distill_manifest.csv",
        "manifest.csv", 
        "pretrain_manifest.csv"
    ]
    
    manifest_path = None
    for candidate in manifest_candidates:
        candidate_path = client_path / candidate
        if candidate_path.exists():
            manifest_path = candidate_path
            break
    
    if manifest_path is None:
        logger.error(f"No manifest found for client_{client_id}")
        return None, None
    
    logger.info(f"Using manifest: {manifest_path}")
    
    # Load audio files from manifest
    audio_files = load_audio_files_from_manifest(manifest_path, client_path)
    
    if not audio_files:
        logger.error(f"No audio files found for client_{client_id}")
        return None, None
    
    if max_samples and len(audio_files) > max_samples:
        audio_files = audio_files[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Extract features
    logger.info(f"Initializing feature extractor for client_{client_id}...")
    extractor = FeatureExtractor()
    logger.info(f"Feature extractor initialized. Processing {len(audio_files)} audio files...")
    
    features = []
    valid_paths = []
    
    for audio_path in tqdm(audio_files, desc=f"Extracting features for client_{client_id}"):
        feature = extractor.extract_features(audio_path)
        if feature is not None:
            features.append(feature)
            valid_paths.append(audio_path)
    
    if not features:
        logger.error(f"No valid features extracted for client_{client_id}")
        return None, None
    
    features = np.array(features)
    logger.info(f"Extracted {len(features)} valid features with shape {features.shape}")
    
    return features, {'client_id': client_id, 'paths': valid_paths}


def collect_features_from_global(data_root, split_name):
    """Collect features from global test/validation data"""
    logger.info(f"Collecting features from global/{split_name}")
    
    global_path = Path(data_root) / "global"
    
    # Look for the appropriate manifest
    manifest_candidates = [
        f"{split_name}_manifest.csv",
        f"global_{split_name}_manifest.csv"
    ]
    
    manifest_path = None
    for candidate in manifest_candidates:
        candidate_path = global_path / candidate
        if candidate_path.exists():
            manifest_path = candidate_path
            break
    
    if manifest_path is None:
        logger.error(f"No {split_name} manifest found in global/")
        return None, None
    
    logger.info(f"Using global manifest: {manifest_path}")
    
    # Load audio files
    audio_files = load_audio_files_from_manifest(manifest_path, global_path)
    
    if not audio_files:
        logger.error(f"No audio files found for global/{split_name}")
        return None, None
    
    # Extract features
    logger.info(f"Initializing feature extractor for global/{split_name}...")
    extractor = FeatureExtractor()
    logger.info(f"Feature extractor initialized. Processing {len(audio_files)} audio files...")
    
    features = []
    valid_paths = []
    
    for audio_path in tqdm(audio_files, desc=f"Extracting features for global/{split_name}"):
        feature = extractor.extract_features(audio_path)
        if feature is not None:
            features.append(feature)
            valid_paths.append(audio_path)
    
    if not features:
        logger.error(f"No valid features extracted for global/{split_name}")
        return None, None
    
    features = np.array(features)
    logger.info(f"Extracted {len(features)} valid features with shape {features.shape}")
    
    return features, {'split_name': split_name, 'paths': valid_paths}


def train_kmeans_model(features_list, n_clusters=504, batch_size=1000):
    """Train K-means model on collected features"""
    logger.info(f"Training K-means with {n_clusters} clusters")
    
    # Combine all features
    all_features = np.vstack(features_list)
    logger.info(f"Total features for clustering: {all_features.shape}")
    
    # Use MiniBatchKMeans for large datasets
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=42,
        verbose=1,
        max_iter=100
    )
    
    logger.info("Fitting K-means model...")
    kmeans.fit(all_features)
    
    logger.info("K-means training completed")
    return kmeans


def generate_targets_for_client(data_root, client_id, kmeans_model):
    """Generate K-means targets for a specific client"""
    logger.info(f"Generating targets for client_{client_id}")
    
    client_path = Path(data_root) / f"client_{client_id}"
    
    # Find manifest file
    manifest_candidates = ["distill_manifest.csv", "manifest.csv", "pretrain_manifest.csv"]
    manifest_path = None
    for candidate in manifest_candidates:
        candidate_path = client_path / candidate
        if candidate_path.exists():
            manifest_path = candidate_path
            break
    
    if manifest_path is None:
        logger.error(f"No manifest found for client_{client_id}")
        return
    
    # Load audio files
    audio_files = load_audio_files_from_manifest(manifest_path, client_path)
    
    if not audio_files:
        logger.error(f"No audio files found for client_{client_id}")
        return
    
    # Extract features and generate targets
    extractor = FeatureExtractor()
    targets_dict = {}
    
    for audio_path in tqdm(audio_files, desc=f"Generating targets for client_{client_id}"):
        feature = extractor.extract_features(audio_path)
        if feature is not None:
            # Get cluster assignment
            cluster = kmeans_model.predict(feature.reshape(1, -1))[0]
            targets_dict[audio_path] = int(cluster)
    
    # Save targets
    output_path = client_path / "kmeans_targets.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(targets_dict, f)
    
    logger.info(f"Saved {len(targets_dict)} targets to {output_path}")
    return targets_dict


def generate_targets_for_global(data_root, split_name, kmeans_model):
    """Generate K-means targets for global test/validation data"""
    logger.info(f"Generating targets for global/{split_name}")
    
    global_path = Path(data_root) / "global"
    
    # Find manifest file
    manifest_candidates = [f"{split_name}_manifest.csv", f"global_{split_name}_manifest.csv"]
    manifest_path = None
    for candidate in manifest_candidates:
        candidate_path = global_path / candidate
        if candidate_path.exists():
            manifest_path = candidate_path
            break
    
    if manifest_path is None:
        logger.error(f"No {split_name} manifest found in global/")
        return
    
    # Load audio files
    audio_files = load_audio_files_from_manifest(manifest_path, global_path)
    
    if not audio_files:
        logger.error(f"No audio files found for global/{split_name}")
        return
    
    # Extract features and generate targets
    extractor = FeatureExtractor()
    targets_dict = {}
    
    for audio_path in tqdm(audio_files, desc=f"Generating targets for global/{split_name}"):
        feature = extractor.extract_features(audio_path)
        if feature is not None:
            # Get cluster assignment
            cluster = kmeans_model.predict(feature.reshape(1, -1))[0]
            targets_dict[audio_path] = int(cluster)
    
    # Save targets
    output_path = global_path / f"{split_name}_kmeans_targets.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(targets_dict, f)
    
    logger.info(f"Saved {len(targets_dict)} targets to {output_path}")
    return targets_dict


def main():
    """Main function to generate K-means targets for all clients"""
    logger.info("Starting K-means target generation")
    
    # Configuration - Update this to your actual path
    data_root = "/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data"
    n_clusters = 504
    max_samples_per_client = None  # Use all samples
    
    if torch.cuda.is_available():
        logger.info(f"Using {torch.cuda.device_count()} GPUs for processing")
    else:
        logger.info("Using CPU for processing")
    
    # Check if data root exists
    if not Path(data_root).exists():
        logger.error(f"Data root not found: {data_root}")
        return
    
    # Find all available clients (0-9)
    available_clients = []
    for client_id in range(10):
        client_path = Path(data_root) / f"client_{client_id}"
        if client_path.exists():
            available_clients.append(client_id)
    
    logger.info(f"Found clients: {available_clients}")
    
    # Check for global data
    global_splits = []
    global_path = Path(data_root) / "global"
    if global_path.exists():
        for split_name in ["test", "validation"]:
            for candidate in [f"{split_name}_manifest.csv", f"global_{split_name}_manifest.csv"]:
                if (global_path / candidate).exists():
                    global_splits.append(split_name)
                    break
    
    logger.info(f"Found global splits: {global_splits}")
    
    if not available_clients and not global_splits:
        logger.error("No clients or global data found!")
        return
    
    # Step 1: Collect features from all available data
    logger.info("=" * 50)
    logger.info("STEP 1: Collecting features for K-means training")
    logger.info("=" * 50)
    
    all_features = []
    
    # Collect from clients
    for client_id in available_clients:
        features, info = collect_features_from_client(
            data_root, client_id, max_samples=max_samples_per_client
        )
        if features is not None:
            all_features.append(features)
    
    # Collect from global data
    for split_name in global_splits:
        features, info = collect_features_from_global(data_root, split_name)
        if features is not None:
            all_features.append(features)
    
    if not all_features:
        logger.error("No features collected, exiting")
        return
    
    # Step 2: Train K-means model
    logger.info("=" * 50)
    logger.info("STEP 2: Training K-means model")
    logger.info("=" * 50)
    
    kmeans_model = train_kmeans_model(all_features, n_clusters=n_clusters)
    
    # Save the trained model
    model_path = Path(data_root) / "kmeans_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans_model, f)
    logger.info(f"Saved K-means model to {model_path}")
    
    # Step 3: Generate targets for each client and global data
    logger.info("=" * 50)
    logger.info("STEP 3: Generating targets")
    logger.info("=" * 50)
    
    # Generate for clients
    for client_id in available_clients:
        try:
            generate_targets_for_client(data_root, client_id, kmeans_model)
        except Exception as e:
            logger.error(f"Failed to generate targets for client_{client_id}: {e}")
    
    # Generate for global data
    for split_name in global_splits:
        try:
            generate_targets_for_global(data_root, split_name, kmeans_model)
        except Exception as e:
            logger.error(f"Failed to generate targets for global/{split_name}: {e}")
    
    # Step 4: Verification
    logger.info("=" * 50)
    logger.info("STEP 4: Verification")
    logger.info("=" * 50)
    
    # Verify client targets
    for client_id in available_clients:
        targets_path = Path(data_root) / f"client_{client_id}" / "kmeans_targets.pkl"
        if targets_path.exists():
            with open(targets_path, 'rb') as f:
                targets = pickle.load(f)
            logger.info(f"client_{client_id}: {len(targets)} targets generated")
            
            # Show cluster distribution
            cluster_counts = {}
            for target in targets.values():
                cluster_counts[target] = cluster_counts.get(target, 0) + 1
            logger.info(f"client_{client_id}: Using {len(cluster_counts)} unique clusters out of {n_clusters}")
        else:
            logger.warning(f"client_{client_id}: No targets file found")
    
    # Verify global targets
    for split_name in global_splits:
        targets_path = Path(data_root) / "global" / f"{split_name}_kmeans_targets.pkl"
        if targets_path.exists():
            with open(targets_path, 'rb') as f:
                targets = pickle.load(f)
            logger.info(f"global/{split_name}: {len(targets)} targets generated")
            
            cluster_counts = {}
            for target in targets.values():
                cluster_counts[target] = cluster_counts.get(target, 0) + 1
            logger.info(f"global/{split_name}: Using {len(cluster_counts)} unique clusters out of {n_clusters}")
        else:
            logger.warning(f"global/{split_name}: No targets file found")
    
    logger.info("K-means target generation completed!")


if __name__ == "__main__":
    main()