#!/usr/bin/env python3
# metadata_processor.py

import os
import json
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from mutagen.id3 import ID3
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class TrackMetadata:
    file_path: str
    genre: str
    title: str
    duration: float
    sample_rate: int
    tempo: float
    key: Optional[str]
    mean_amplitude: float
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float

class MetadataProcessor:
    def __init__(self, dataset_dir: Path):
        """Initialize metadata processor"""
        self.dataset_dir = Path(dataset_dir)
        self.metadata_dir = self.dataset_dir / "metadata"
        self.metadata_file = self.metadata_dir / "dataset_metadata.json"
        self.downloads_dir = dataset_dir.parent / "downloads"  # Add this line
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def _hash_file(self, file_path: Path) -> str:
        """Calculate file hash"""
        import hashlib
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def process_file(self, audio_path: Path) -> Dict[str, Any]:
        """Process a single audio file and extract metadata"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract basic metadata
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract musical features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Get ID3 tags if available
            tags = {}
            try:
                audio = ID3(audio_path)
                tags = {
                    "title": str(audio.get("TIT2", "")),
                    "artist": str(audio.get("TPE1", "")),
                    "album": str(audio.get("TALB", "")),
                    "year": str(audio.get("TDRC", "")),
                    "genre": str(audio.get("TCON", ""))
                }
            except Exception as e:
                print(f"Warning: Could not read ID3 tags from {audio_path.name}: {e}")
            
            # Calculate file hash
            file_hash = self._hash_file(audio_path)
            
            # Extract audio features
            metadata = {
                "file_info": {
                    "filename": audio_path.name,
                    "path": str(audio_path),
                    "size_bytes": audio_path.stat().st_size,
                    "format": audio_path.suffix[1:],
                    "hash": file_hash
                },
                "audio_info": {
                    "duration": float(duration),
                    "sample_rate": int(sr),
                    "channels": len(y.shape) if len(y.shape) > 1 else 1,
                },
                "musical_info": {
                    "tempo": float(tempo),
                    "beat_frames": beats.tolist() if len(beats) > 0 else [],
                    "estimated_key": self._estimate_key(y, sr)
                },
                "tags": tags,
                "processing_info": {
                    "processed_date": datetime.now().isoformat(),
                    "processor_version": "1.0.0"
                }
            }
            
            # Save individual metadata file
            metadata_file = self.metadata_dir / f"{audio_path.stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            print(f"Processed metadata for: {audio_path.name}")
            return metadata
            
        except Exception as e:
            print(f"Error processing metadata for {audio_path.name}: {e}")
            return {
                "file_info": {
                    "filename": audio_path.name,
                    "path": str(audio_path),
                    "hash": self._hash_file(audio_path)
                },
                "error": str(e)
            }
    
    def _estimate_key(self, y, sr) -> Optional[str]:
        """Estimate musical key of the audio"""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_indices = np.mean(chroma, axis=1)
            key_index = np.argmax(key_indices)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return keys[key_index]
        except:
            return None

    def get_state(self) -> Dict[str, Any]:
        """Get current processing state"""
        return {
            "total_files_processed": len(list(self.metadata_dir.glob("*.json"))),
            "last_update": datetime.now().isoformat()
        }
        
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

    def process_audio_file(self, audio_path: Path, genre: str) -> TrackMetadata:
        """Extract audio features and metadata from a single track"""
        # Load audio file with a standard sample rate
        y, sr = librosa.load(audio_path, sr=22050)  # Fixed sample rate for consistency
        
        # Extract basic metadata
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        # Extract musical features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        key = self.estimate_key(y, sr)
        
        # Extract audio characteristics
        mean_amplitude = float(np.mean(np.abs(y)))
        rms_energy = float(np.sqrt(np.mean(y**2)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        return TrackMetadata(
            file_path=str(audio_path),
            genre=genre,
            title=audio_path.stem,
            duration=duration,
            sample_rate=int(sr),
            tempo=float(tempo),
            key=key,
            mean_amplitude=mean_amplitude,
            rms_energy=rms_energy,
            zero_crossing_rate=zero_crossing_rate,
            spectral_centroid=float(np.mean(spectral_centroids)),
            spectral_bandwidth=float(np.mean(spectral_bandwidth))
        )
    
    def process_all_tracks(self) -> Dict[str, Any]:
        """Process all downloaded tracks and generate metadata"""
        metadata = {
            "tracks": [],
            "statistics": {
                "total_tracks": 0,
                "total_duration": 0,
                "average_tempo": 0,
                "genre_distribution": {}
            }
        }
        
        all_tempos = []
        
        # Process each genre directory
        for genre_dir in self.downloads_dir.iterdir():
            if not genre_dir.is_dir():
                continue
                
            print(f"\nProcessing {genre_dir.name} tracks...")
            genre_count = 0
            
            # Process each audio file in the genre directory
            for audio_file in genre_dir.glob("*.mp3"):
                try:
                    track_metadata = self.process_audio_file(audio_file, genre_dir.name)
                    metadata["tracks"].append(asdict(track_metadata))
                    print(f"Processed: {audio_file.name}")
                    
                    # Update statistics
                    metadata["statistics"]["total_duration"] += track_metadata.duration
                    all_tempos.append(track_metadata.tempo)
                    genre_count += 1
                    
                except Exception as e:
                    print(f"Error processing {audio_file.name}: {str(e)}")
            
            metadata["statistics"]["genre_distribution"][genre_dir.name] = genre_count
        
        # Calculate final statistics
        metadata["statistics"]["total_tracks"] = len(metadata["tracks"])
        if all_tempos:
            metadata["statistics"]["average_tempo"] = float(np.mean(all_tempos))
            
        # Convert all numpy types to Python native types
        metadata = self.convert_to_serializable(metadata)
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete metadata extraction pipeline"""
        print("Starting metadata extraction...")
        metadata = self.process_all_tracks()
        self.save_metadata(metadata)
        print(f"\nMetadata extraction complete. Saved to {self.metadata_file}")
        
        # Print summary
        stats = metadata["statistics"]
        print("\nDataset Summary:")
        print(f"Total tracks: {stats['total_tracks']}")
        print(f"Total duration: {stats['total_duration']/3600:.2f} hours")
        print(f"Average tempo: {stats['average_tempo']:.1f} BPM")
        print("\nGenre distribution:")
        for genre, count in stats['genre_distribution'].items():
            print(f"  {genre}: {count} tracks")
            
        return metadata