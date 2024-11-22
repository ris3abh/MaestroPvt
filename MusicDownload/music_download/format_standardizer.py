#!/usr/bin/env python3
# format_standardizer.py

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Any
from pydub import AudioSegment
import json

class FormatStandardizer:
    def __init__(self,
                target_sr: int = 44100,
                target_channels: int = 2,
                target_format: str = 'wav',
                target_subtype: str = 'PCM_16',
                target_lufs: float = -14.0):
        """
        Initialize format standardizer
        
        Args:
            target_sr: Target sample rate (44.1kHz for CD quality)
            target_channels: Number of channels (2 for stereo)
            target_format: Output format ('wav' or 'mp3')
            target_subtype: Bit depth format
            target_lufs: Target loudness (industry standard is -14 LUFS)
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.target_format = target_format
        self.target_subtype = target_subtype
        self.target_lufs = target_lufs
        
    def standardize_audio(self, 
                         input_path: Path, 
                         output_path: Path) -> Dict[str, Any]:
        """Standardize a single audio file"""
        # Load audio with original sr
        y, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Convert to target sample rate
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        
        # Convert to stereo if mono
        if len(y.shape) == 1:
            y = np.vstack((y, y))
        
        # Normalize audio
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95  # Leave headroom
            
        # Save with standard format
        sf.write(
            output_path,
            y.T,
            self.target_sr,
            format=self.target_format,
            subtype=self.target_subtype
        )
        
        return {
            "original_sr": sr,
            "new_sr": self.target_sr,
            "channels": self.target_channels,
            "format": self.target_format,
            "peak_normalized": float(peak)
        }

    def process_dataset(self, 
                       input_dir: Path, 
                       output_dir: Path) -> Dict[str, Any]:
        """Process entire dataset"""
        stats = {
            "processed_files": 0,
            "failed_files": 0,
            "format_stats": {}
        }
        
        # Process each genre directory
        for genre_dir in input_dir.iterdir():
            if not genre_dir.is_dir():
                continue
            
            # Create output genre directory
            genre_output_dir = output_dir / genre_dir.name
            genre_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each audio file
            for audio_file in genre_dir.glob("*.*"):
                try:
                    output_path = genre_output_dir / f"{audio_file.stem}.{self.target_format}"
                    file_stats = self.standardize_audio(audio_file, output_path)
                    stats["format_stats"][str(audio_file)] = file_stats
                    stats["processed_files"] += 1
                except Exception as e:
                    stats["failed_files"] += 1
                    print(f"Error processing {audio_file}: {str(e)}")
        
        return stats