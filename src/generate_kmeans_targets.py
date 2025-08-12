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
import os
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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
    # Compute total frames available for this client
    total_frames = int(sum(fr.shape[0] for fr in all_frames if fr.numel() > 0))

    if total_frames == 0:
        logger.warning("No frames available for client; skipping KMeans fit")
        return None  # Safe because all sequences are empty

    effective_k = max(1, min(n_clusters, total_frames))
    if effective_k < n_clusters:
        logger.warning(
            f"Downscaling n_clusters from {n_clusters} to {effective_k} due to limited frames ({total_frames})."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=effective_k,
        batch_size=batch_size,
        random_state=seed,
        max_iter=max_iter,
        n_init='auto',
        verbose=0,
    )

    # Stream frames in chunks, but ensure the FIRST call has at least effective_k samples
    initialized = False
    init_buffer: List[np.ndarray] = []
    init_count = 0

    for frames in all_frames:
        if frames.numel() == 0:
            continue

        batch_np = frames.cpu().numpy()

        if not initialized:
            init_buffer.append(batch_np)
            init_count += batch_np.shape[0]
            if init_count >= effective_k:
                init_np = np.concatenate(init_buffer, axis=0)
                kmeans.partial_fit(init_np)
                initialized = True
            continue

        kmeans.partial_fit(batch_np)

    if not initialized:
        # Should not happen because total_frames >= effective_k, but handle defensively
        init_np = np.concatenate(init_buffer, axis=0) if init_buffer else None
        if init_np is not None and init_np.shape[0] > 0:
            kmeans.partial_fit(init_np)
        else:
            return None
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


def _frames_task(args: Tuple[str, str, int, int, int]) -> np.ndarray:
    """Worker task to load audio and compute MFCC, returns numpy array [T, 13]."""
    audio_rel, client_path_str, sample_rate, frame_stride, max_length = args
    try:
        if audio_rel is None or audio_rel == "":
            return np.zeros((0, 13), dtype=np.float32)
        audio_path = Path(client_path_str) / audio_rel
        if not audio_path.exists():
            return np.zeros((0, 13), dtype=np.float32)
        audio = load_audio(audio_path, sample_rate, max_length)
        frames = compute_mfcc_frames(audio, sample_rate, frame_stride)
        return frames.cpu().numpy().astype(np.float32)
    except Exception:
        # On any failure, return empty frames to keep order
        return np.zeros((0, 13), dtype=np.float32)


def process_client(client_path: Path, sample_rate: int, frame_stride: int, max_length: int,
                   n_clusters: int, batch_size: int, max_iter: int, seed: int,
                   workers: int = 1) -> None:
    # Check if targets already exist
    output_path = client_path / "kmeans_targets.npy"
    if output_path.exists():
        logger.info(
            f"Skipping {client_path.name} - kmeans_targets.npy already exists")
        return

    manifest_path = client_path / "manifest.csv"
    df = load_manifest(manifest_path)

    # Prepare list of relative audio paths from manifest
    rel_paths: List[str] = []
    for _, row in df.iterrows():
        if 'audio_file' in row:
            rel_paths.append(row['audio_file'])
        elif 'audio_path' in row:
            rel_paths.append(row['audio_path'])
        else:
            rel_paths.append(None)

    # Load MFCC frames for each file (ordered by manifest rows), parallel if workers > 1
    all_frames: List[torch.Tensor] = []
    if workers and workers > 1:
        args_list = [(rel, str(client_path), sample_rate, frame_stride, max_length)
                     for rel in rel_paths]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(
                ex.map(_frames_task, args_list),
                total=len(args_list),
                desc=f"Load MFCC client {client_path.name} (x{workers})"
            ))
        for arr in results:
            if arr.size == 0:
                all_frames.append(torch.zeros(0, 13))
            else:
                all_frames.append(torch.from_numpy(arr))
    else:
        for rel in tqdm(rel_paths, total=len(rel_paths), desc=f"Load MFCC client {client_path.name}"):
            arr = _frames_task(
                (rel, str(client_path), sample_rate, frame_stride, max_length))
            if arr.size == 0:
                all_frames.append(torch.zeros(0, 13))
            else:
                all_frames.append(torch.from_numpy(arr))

    # Fit per-client KMeans on concatenated frames (streamed)
    kmeans = fit_client_kmeans(
        all_frames, n_clusters, batch_size, max_iter, seed)

    # Assign labels per utterance
    if kmeans is None:
        # No frames or unable to fit; return empty arrays in order
        targets_list = [np.array([], dtype=np.int64) for _ in all_frames]
    else:
        targets_list = assign_client_targets(all_frames, kmeans)

    # Save as ragged list to .npy (object array)
    output_path = client_path / "kmeans_targets.npy"
    np.save(output_path, np.array(targets_list, dtype=object), allow_pickle=True)
    logger.info(f"Saved {len(targets_list)} sequences to {output_path}")


def _client_task_entry(args: Tuple[str, int, int, int, int, int, int, int, int]) -> Tuple[int, str]:
    """Top-level picklable entrypoint for processing a single client in a process."""
    (data_root_str, client_id, frame_stride, max_length, n_clusters,
     batch_size, max_iter, seed, workers) = args
    try:
        process_client(
            client_path=Path(data_root_str) / f"client_{client_id}",
            sample_rate=16000,
            frame_stride=frame_stride,
            max_length=max_length,
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            seed=seed,
            workers=workers,
        )
        return client_id, "ok"
    except Exception as e:
        return client_id, f"error: {e}"


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
    parser.add_argument("--workers", type=int, default=1,
                        help="Per-client parallel workers for MFCC loading (threads)")
    parser.add_argument("--client-workers", type=int, default=1,
                        help="Parallel clients to process at once (processes)")
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

    # Optionally process multiple clients in parallel (process-based)
    if args.client_workers and args.client_workers > 1 and len(client_ids) > 1:
        with ProcessPoolExecutor(max_workers=args.client_workers) as ex:
            task_args = [
                (str(data_root), cid, args.frame_stride, args.max_length, args.n_clusters,
                 args.batch_size, args.max_iter, args.seed, args.workers)
                for cid in client_ids
            ]
            results = list(tqdm(ex.map(_client_task_entry, task_args),
                           total=len(task_args), desc="Clients"))
        for cid, status in results:
            if status != "ok":
                logger.error(f"Client {cid} failed: {status}")
    else:
        for cid in client_ids:
            cid_, status = _client_task_entry((str(data_root), cid, args.frame_stride, args.max_length,
                                               args.n_clusters, args.batch_size, args.max_iter,
                                               args.seed, args.workers))
            if status != "ok":
                logger.error(f"Client {cid_} failed: {status}")


if __name__ == "__main__":
    main()
