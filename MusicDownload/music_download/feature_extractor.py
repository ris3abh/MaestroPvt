#!/usr/bin/env python3
# feature_extractor.py

import librosa
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

@dataclass
class AudioFeatures:
    # Temporal features
    tempo: float
    beats: List[float]
    
    # Spectral features
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    
    # Harmonic features
    key_strength: List[float]
    
    # Energy features
    rms_energy_mean: float
    zero_crossing_rate_mean: float

class FeatureExtractor:
    def __init__(self, frame_size: float = 0.05):
        self.frame_size = frame_size
        
    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def extract_features(self, audio_path: Path) -> Dict[str, Any]:
        """Extract all features from an audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract temporal features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Extract harmonic features
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_strengths = np.mean(chromagram, axis=1)
        
        # Extract energy features
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        
        features = {
            "temporal_features": {
                "tempo": float(tempo),
                "beats": beat_times.tolist()
            },
            "spectral_features": {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidths)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff))
            },
            "harmonic_features": {
                "key_strength": key_strengths.tolist()
            },
            "energy_features": {
                "rms_energy_mean": float(np.mean(rms)),
                "zero_crossing_rate_mean": float(np.mean(zcr))
            }
        }
        
        return self.convert_to_serializable(features)
    
    def process_dataset(self, input_dir: Path) -> Dict[str, Any]:
        """Process entire dataset"""
        dataset_features = {
            "features": {},
            "statistics": {
                "total_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "average_features": {}
            }
        }
        
        # Process each genre directory
        for genre_dir in input_dir.iterdir():
            if not genre_dir.is_dir():
                continue
                
            print(f"\nExtracting features from {genre_dir.name} tracks...")
            
            # Process each audio file
            for audio_file in genre_dir.glob("*.mp3"):
                dataset_features["statistics"]["total_files"] += 1
                
                try:
                    features = self.extract_features(audio_file)
                    dataset_features["features"][str(audio_file)] = features
                    dataset_features["statistics"]["processed_files"] += 1
                    print(f"Processed: {audio_file.name}")
                    
                except Exception as e:
                    dataset_features["statistics"]["failed_files"] += 1
                    print(f"Error processing {audio_file.name}: {str(e)}")
        
        return dataset_features

    def save_features(self, features: Dict[str, Any], output_file: Path) -> None:
        """Save features to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=4)