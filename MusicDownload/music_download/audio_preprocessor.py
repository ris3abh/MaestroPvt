#!/usr/bin/env python3
# audio_processor.py

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any
from scipy import signal
from pydub import AudioSegment

class AudioPreprocessor:
    def __init__(self, 
                target_sr: int = 44100,
                target_db: float = -14.0,
                min_duration: int = 60,
                max_duration: int = 300):
        """
        Initialize audio preprocessor with target parameters
        
        Args:
            target_sr: Target sample rate
            target_db: Target loudness in dB LUFS
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        """
        self.target_sr = target_sr
        self.target_db = target_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to target loudness"""
        # Calculate current loudness
        rms = np.sqrt(np.mean(y**2))
        current_db = 20 * np.log10(rms)
        
        # Calculate required gain
        gain = 10**((self.target_db - current_db) / 20)
        
        # Apply gain
        return y * gain
    
    def apply_bandpass_filter(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter (20Hz - 20kHz)"""
        nyquist = sr // 2
        low = 20 / nyquist
        high = 20000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, y)
    
    def trim_silence(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence from beginning and end"""
        return librosa.effects.trim(y, top_db=30)[0]
    
    def ensure_duration(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Ensure audio is within duration limits"""
        duration = len(y) / sr
        
        if duration < self.min_duration:
            # Repeat audio to meet minimum duration
            repeats = int(np.ceil(self.min_duration / duration))
            y = np.tile(y, repeats)[:int(self.min_duration * sr)]
            
        if duration > self.max_duration:
            # Take center section
            center = len(y) // 2
            half_samples = int(self.max_duration * sr // 2)
            y = y[center-half_samples:center+half_samples]
            
        return y
    
    def process_file(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Process a single audio file"""
        # Load audio
        y, sr = librosa.load(input_path, sr=self.target_sr)
        
        # Store original properties
        original_duration = len(y) / sr
        original_rms = np.sqrt(np.mean(y**2))
        
        # Apply processing steps
        y = self.trim_silence(y, sr)
        y = self.apply_bandpass_filter(y, sr)
        y = self.normalize_audio(y)
        y = self.ensure_duration(y, sr)
        
        # Calculate new properties
        processed_duration = len(y) / sr
        processed_rms = np.sqrt(np.mean(y**2))
        
        # Save processed audio
        sf.write(output_path, y, sr)
        
        return {
            "original_duration": original_duration,
            "processed_duration": processed_duration,
            "original_rms": float(original_rms),
            "processed_rms": float(processed_rms),
            "sample_rate": sr
        }
    
    def process_dataset(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process entire dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processing_stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "processing_results": {}
        }
        
        # Process each genre directory
        for genre_dir in input_dir.iterdir():
            if not genre_dir.is_dir():
                continue
            
            # Create genre directory in output
            genre_output_dir = output_dir / genre_dir.name
            genre_output_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing {genre_dir.name} tracks...")
            
            # Process each audio file
            for audio_file in genre_dir.glob("*.mp3"):
                processing_stats["total_files"] += 1
                
                try:
                    output_file = genre_output_dir / audio_file.name
                    results = self.process_file(audio_file, output_file)
                    
                    processing_stats["processing_results"][str(audio_file)] = results
                    processing_stats["processed_files"] += 1
                    
                    print(f"Processed: {audio_file.name}")
                    
                except Exception as e:
                    processing_stats["failed_files"] += 1
                    print(f"Error processing {audio_file.name}: {str(e)}")
        
        return processing_stats