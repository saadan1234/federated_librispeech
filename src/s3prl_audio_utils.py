#!/usr/bin/env python3
"""
S3PRL-Compatible Audio Processing Utilities
Replaces custom audio handling in partition_data.py
"""

import torchaudio
import torch
from pathlib import Path
import soundfile as sf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class S3PRLAudioProcessor:
    """
    Audio processing utilities following S3PRL conventions
    Replaces custom audio handling in original partitioner
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def load_audio_s3prl_style(self, wav_path: Path) -> torch.Tensor:
        """
        Load audio using S3PRL's standard approach
        Replaces the custom _load_audio_s3prl_style in your dataset
        """
        try:
            wav, sr = torchaudio.load(wav_path)
            
            # S3PRL standard: assert sample rate and reshape
            if sr != self.sample_rate:
                logger.warning(f'Sample rate mismatch: {sr} != {self.sample_rate}, resampling...')
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resampler(wav)
                
            return wav.view(-1)  # S3PRL standard: flatten to 1D
            
        except Exception as e:
            logger.error(f"Failed to load audio {wav_path}: {e}")
            raise
            
    def calculate_audio_length_s3prl_style(self, wav_path: Path) -> int:
        """
        Calculate audio length in frames (S3PRL bucket dataset format)
        Replaces custom duration calculation
        """
        try:
            # Use soundfile for quick length calculation without loading full audio
            with sf.SoundFile(wav_path) as f:
                return len(f)  # Return length in frames
        except Exception as e:
            logger.warning(f"Could not calculate length for {wav_path}: {e}")
            return 0
            
    def get_audio_info_s3prl_format(self, wav_path: Path) -> dict:
        """
        Get audio info in S3PRL bucket dataset format
        Returns dictionary compatible with S3PRL's CSV structure
        """
        try:
            with sf.SoundFile(wav_path) as f:
                return {
                    'file_path': str(wav_path.relative_to(wav_path.parents[3])),  # Relative to LibriSpeech root
                    'length': len(f),  # Frames
                    'sample_rate': f.samplerate,
                    'channels': f.channels
                }
        except Exception as e:
            logger.error(f"Failed to get audio info for {wav_path}: {e}")
            return {
                'file_path': str(wav_path),
                'length': 0,
                'sample_rate': self.sample_rate,
                'channels': 1
            }
            

class S3PRLCompatibleDataLoader:
    """
    Data loading utilities that match S3PRL's approach
    Replaces custom manifest and CSV generation
    """
    
    def __init__(self, librispeech_root: Path, audio_processor: S3PRLAudioProcessor):
        self.librispeech_root = Path(librispeech_root)
        self.audio_processor = audio_processor
        
    def discover_audio_files_s3prl_style(self, subset: str) -> list:
        """
        Discover audio files using S3PRL's approach
        Replaces custom recursive directory scanning
        """
        subset_dir = self.librispeech_root / subset
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
            
        # S3PRL style: use rglob for recursive discovery
        wav_files = list(subset_dir.rglob("*.flac"))
        
        logger.info(f"Found {len(wav_files)} audio files in {subset}")
        return sorted(wav_files)  # S3PRL style: sorted for reproducibility
        
    def extract_utterance_info_s3prl_style(self, wav_path: Path) -> dict:
        """
        Extract utterance information using S3PRL's naming convention
        Replaces custom path parsing
        """
        # S3PRL standard: speaker-chapter-utterance.flac
        stem = wav_path.stem
        parts = stem.split('-')
        
        if len(parts) >= 3:
            return {
                'utterance_id': stem,
                'speaker_id': int(parts[0]),      # S3PRL uses int for speaker
                'chapter_id': parts[1],
                'sequence_id': parts[2]
            }
        else:
            raise ValueError(f"Invalid LibriSpeech filename format: {wav_path}")
            
    def load_transcript_s3prl_style(self, wav_path: Path) -> str:
        """
        Load transcript using S3PRL's approach
        Replaces custom transcript file handling
        """
        # S3PRL style transcript loading
        transcript_file = wav_path.parent / f"{wav_path.parent.name}.trans.txt"
        utterance_id = wav_path.stem
        
        try:
            with open(transcript_file, 'r') as f:
                for line in f:
                    if line.startswith(utterance_id):
                        return line.strip().split(' ', 1)[1]  # Remove utterance ID
            
            logger.warning(f"Transcript not found for {utterance_id}")
            return ""
            
        except FileNotFoundError:
            logger.warning(f"Transcript file not found: {transcript_file}")
            return ""
            

def replace_custom_audio_processing():
    """
    Example of how to replace custom audio processing in partition_data.py
    """
    
    # OLD WAY (in your partition_data.py):
    # def process_audio_file(audio_file: Path) -> Optional[Dict[str, Any]]:
    #     try:
    #         with sf.SoundFile(audio_file) as f:
    #             duration = len(f) / f.samplerate
    #         return {
    #             'path': str(audio_file),
    #             'utterance_id': audio_file.stem,
    #             'duration': duration
    #         }
    #     except Exception as e:
    #         logger.warning(f"Could not process {audio_file}: {e}")
    #         return None
    
    # NEW WAY (S3PRL-aligned):
    audio_processor = S3PRLAudioProcessor()
    data_loader = S3PRLCompatibleDataLoader(
        librispeech_root="/path/to/LibriSpeech",
        audio_processor=audio_processor
    )
    
    def process_audio_file_s3prl_style(audio_file: Path) -> Optional[dict]:
        try:
            # Get audio info in S3PRL format
            audio_info = audio_processor.get_audio_info_s3prl_format(audio_file)
            
            # Extract utterance info using S3PRL convention
            utterance_info = data_loader.extract_utterance_info_s3prl_style(audio_file)
            
            # Load transcript using S3PRL method
            transcript = data_loader.load_transcript_s3prl_style(audio_file)
            
            # Combine into S3PRL-compatible format
            return {
                **audio_info,
                **utterance_info,
                'transcription': transcript
            }
            
        except Exception as e:
            logger.warning(f"Could not process {audio_file}: {e}")
            return None
            
    return process_audio_file_s3prl_style


if __name__ == "__main__":
    # Example usage
    processor = S3PRLAudioProcessor()
    
    # Test audio loading
    test_file = Path("/path/to/test.flac")
    if test_file.exists():
        audio = processor.load_audio_s3prl_style(test_file)
        print(f"Loaded audio shape: {audio.shape}")
        
        info = processor.get_audio_info_s3prl_format(test_file)
        print(f"Audio info: {info}")
