#!/usr/bin/env python3
"""
Generate per-client KMeans frame-level targets consistent with HuBERT-style pretraining.

Approach:
- For each client, read its `manifest.csv`
- Load audio, resample to 16kHz, truncate to max_length (default 40000),
  compute MFCC with hop=320, win=400, center=False (matching training)
- Train a MiniBatchKMeans on that client's MFCC frames only (privacy-preserving)
- Assign a cluster id per frame for each utterance
- Save targets as a ragged object array (list of 1D int arrays) to `kmeans_targets.npy`
  in the client's folder. Order matches the rows of manifest.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return df


def resolve_audio_path(row: pd.Series, audio_root: Path) -> Path:
    # Support both 'audio_file' and 'audio_path'
    if 'audio_file' in row:
        return audio_root / row['audio_file']
    if 'audio_path' in row:
        return audio_root / row['audio_path']
    raise KeyError("Manifest row must contain 'audio_file' or 'audio_path'")


def load_audio(audio_path: Path, sample_rate: int, max_length: int) -> torch.Tensor:
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(torch.tensor(audio, dtype=torch.float32))
    else:
        audio = torch.tensor(audio, dtype=torch.float32)
    # Truncate
    if audio.numel() > max_length:
        audio = audio[:max_length]
    return audio


def compute_mfcc_frames(audio: torch.Tensor, sample_rate: int, frame_stride: int) -> torch.Tensor:
    mfcc = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,
            "n_mels": 40,
            "hop_length": frame_stride,
            "win_length": 400,
            "center": False,
        },
    )
    feats = mfcc(audio.unsqueeze(0)).squeeze(
        0).transpose(0, 1).contiguous()  # [T, D]
    return feats


def fit_client_kmeans(all_frames: List[torch.Tensor], n_clusters: int, batch_size: int, max_iter: int, seed: int) -> MiniBatchKMeans:
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        max_iter=max_iter,
        n_init='auto',
        verbose=0,
    )
    # Stream frames in chunks
    for frames in all_frames:
        if frames.numel() == 0:
            continue
        kmeans.partial_fit(frames.cpu().numpy())
    return kmeans


def assign_client_targets(all_frames: List[torch.Tensor], kmeans: MiniBatchKMeans) -> List[np.ndarray]:
    targets_list: List[np.ndarray] = []
    for frames in all_frames:
        if frames.numel() == 0:
            targets_list.append(np.array([], dtype=np.int64))
            continue
        labels = kmeans.predict(frames.cpu().numpy()).astype(np.int64)
        targets_list.append(labels)
    return targets_list


def process_client(client_path: Path, sample_rate: int, frame_stride: int, max_length: int,
                   n_clusters: int, batch_size: int, max_iter: int, seed: int) -> None:
    manifest_path = client_path / "manifest.csv"
    df = load_manifest(manifest_path)

    # Load MFCC frames for each file (ordered by manifest rows)
    all_frames: List[torch.Tensor] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Load MFCC client {client_path.name}"):
        audio_path = resolve_audio_path(row, client_path)
        if not audio_path.exists():
            all_frames.append(torch.zeros(0, 13))
            continue
        audio = load_audio(audio_path, sample_rate, max_length)
        frames = compute_mfcc_frames(audio, sample_rate, frame_stride)
        all_frames.append(frames)

    # Fit per-client KMeans on concatenated frames (streamed)
    kmeans = fit_client_kmeans(
        all_frames, n_clusters, batch_size, max_iter, seed)

    # Assign labels per utterance
    targets_list = assign_client_targets(all_frames, kmeans)

    # Save as ragged list to .npy (object array)
    output_path = client_path / "kmeans_targets.npy"
    np.save(output_path, np.array(targets_list, dtype=object), allow_pickle=True)
    logger.info(f"Saved {len(targets_list)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-client KMeans frame targets")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to data root containing client_* folders")
    parser.add_argument("--n-clusters", type=int, default=504)
    parser.add_argument("--max-length", type=int, default=40000)
    parser.add_argument("--frame-stride", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=str, default="",
                        help="Comma-separated client ids to process; empty=auto-detect")
    args = parser.parse_args()

    data_root = Path(args["data_root"]) if isinstance(
        args, dict) else Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Determine clients
    if args.clients:
        client_ids = [int(x) for x in args.clients.split(',')]
    else:
        client_ids = sorted([int(p.name.split('_')[-1])
                            for p in data_root.glob('client_*') if p.is_dir()])
    logger.info(f"Processing clients: {client_ids}")

    for cid in client_ids:
        try:
            process_client(
                client_path=data_root / f"client_{cid}",
                sample_rate=16000,
                frame_stride=args.frame_stride,
                max_length=args.max_length,
                n_clusters=args.n_clusters,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"Client {cid} failed: {e}")


if __name__ == "__main__":
    main()
