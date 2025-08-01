#!/usr/bin/env python3
"""
Test script to verify distillation data loading and kmeans targets conversion
"""

from federated_hubert_pretraining import LibriSpeechPretrainingDataset
from transformers import Wav2Vec2FeatureExtractor
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append('src')


def test_data_loading():
    """Test data loading and kmeans targets conversion"""

    # Test data path
    data_path = Path(
        "/home/saadan/scratch/federated_librispeech/src/federated_librispeech/data/client_0")

    print(f"Testing data loading from: {data_path}")

    # Check if files exist
    manifest_file = data_path / "manifest.csv"
    kmeans_file = data_path / "kmeans_targets.pkl"

    print(f"Manifest file exists: {manifest_file.exists()}")
    print(f"Kmeans file exists: {kmeans_file.exists()}")

    if not manifest_file.exists():
        print("Manifest file not found!")
        return

    # Load manifest
    manifest_df = pd.read_csv(manifest_file)
    print(f"Manifest shape: {manifest_df.shape}")
    print(f"Manifest columns: {manifest_df.columns.tolist()}")

    # Filter for training data
    train_df = manifest_df[manifest_df['split'] == 'train']
    print(f"Training samples: {len(train_df)}")

    if len(train_df) == 0:
        print("No training data found!")
        return

    # Test kmeans targets conversion
    if kmeans_file.exists():
        print("\nTesting kmeans targets conversion...")

        # Load original kmeans targets
        with open(kmeans_file, 'rb') as f:
            kmeans_dict = pickle.load(f)

        print(f"Original kmeans dict has {len(kmeans_dict)} entries")

        # Show sample entries
        sample_items = list(kmeans_dict.items())[:3]
        print("Sample kmeans entries:")
        for audio_path, cluster_idx in sample_items:
            print(f"  {audio_path} -> {cluster_idx}")

        # Create mapping from audio paths to indices
        audio_to_idx = {}
        for idx, row in train_df.iterrows():
            audio_path = str(data_path / row['audio_path'])
            audio_to_idx[audio_path] = idx

        print(f"Created mapping for {len(audio_to_idx)} audio files")

        # Create numpy array
        num_samples = len(train_df)
        kmeans_array = np.zeros(num_samples, dtype=np.int64)

        matched_count = 0
        for audio_path, cluster_idx in kmeans_dict.items():
            if audio_path in audio_to_idx:
                idx = audio_to_idx[audio_path]
                kmeans_array[idx] = cluster_idx
                matched_count += 1

        print(f"Matched {matched_count}/{len(kmeans_dict)} kmeans entries")
        print(f"Created numpy array with shape: {kmeans_array.shape}")
        print(
            f"Array statistics: min={kmeans_array.min()}, max={kmeans_array.max()}, mean={kmeans_array.mean():.2f}")

        # Save numpy array
        kmeans_numpy_path = data_path / "kmeans_targets.npy"
        np.save(kmeans_numpy_path, kmeans_array)
        print(f"Saved numpy array to: {kmeans_numpy_path}")

        # Test dataset loading
        print("\nTesting dataset loading...")

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base")

        try:
            dataset = LibriSpeechPretrainingDataset(
                manifest_file=str(data_path / "manifest.csv"),
                audio_root=str(data_path),
                feature_extractor=feature_extractor,
                max_length=80000,
                mask_prob=0.15,
                mask_length=10,
                kmeans_targets_file=str(kmeans_numpy_path),
                vocab_size=504
            )

            print(f"Dataset loaded successfully with {len(dataset)} samples")

            # Test getting a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {list(sample.keys())}")

                if 'input_values' in sample:
                    print(
                        f"Input values shape: {sample['input_values'].shape}")

                if 'target' in sample:
                    target = sample['target']
                    print(f"Target shape: {target.shape}")
                    print(f"Target type: {type(target)}")

                    # Handle different target formats
                    if target.dim() == 0:  # Scalar tensor
                        print(f"Target value: {target.item()}")
                    elif target.dim() == 1:  # 1D tensor
                        print(
                            f"Target values (first 10): {target[:10].tolist()}")
                    else:  # Multi-dimensional tensor
                        print(
                            f"Target values (first 10): {target.flatten()[:10].tolist()}")

                print("Data loading test PASSED!")
            else:
                print("Dataset is empty!")

        except Exception as e:
            print(f"Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Kmeans file not found, testing auto-generation...")

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base")

        try:
            dataset = LibriSpeechPretrainingDataset(
                manifest_file=str(data_path / "manifest.csv"),
                audio_root=str(data_path),
                feature_extractor=feature_extractor,
                max_length=80000,
                mask_prob=0.15,
                mask_length=10,
                auto_generate_kmeans=True,
                vocab_size=504
            )

            print(
                f"Dataset with auto-generated kmeans loaded successfully with {len(dataset)} samples")
            print("Auto-generation test PASSED!")

        except Exception as e:
            print(f"Auto-generation failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()
