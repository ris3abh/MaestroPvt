#!/usr/bin/env python3
# dataset_organizer.py

import h5py
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np

class DatasetOrganizer:
    def __init__(self, root_dir: Path):
        """
        Initialize dataset organizer
        
        Args:
            root_dir: Root directory for organized dataset
        """
        self.root_dir = root_dir
        self.audio_dir = root_dir / "audio"
        self.features_dir = root_dir / "features"
        self.metadata_dir = root_dir / "metadata"
        
        # Create directory structure
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def organize_dataset(self, 
                        source_dir: Path,
                        features: Dict[str, Any],
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Organize the dataset into a standard structure"""
        
        dataset_info = {
            "total_tracks": 0,
            "genres": {},
            "features_file": str(self.features_dir / "features.h5"),
            "metadata_file": str(self.metadata_dir / "metadata.json")
        }
        
        # Create HDF5 file for features
        with h5py.File(self.features_dir / "features.h5", 'w') as f:
            # Create groups for each feature type
            audio_features = f.create_group("audio_features")
            spectral_features = f.create_group("spectral_features")
            temporal_features = f.create_group("temporal_features")
            
            # Process each audio file
            for genre_dir in source_dir.iterdir():
                if not genre_dir.is_dir():
                    continue
                
                genre = genre_dir.name
                dataset_info["genres"][genre] = 0
                
                # Create genre directory in audio dir
                genre_audio_dir = self.audio_dir / genre
                genre_audio_dir.mkdir(exist_ok=True)
                
                # Process each track
                for audio_file in genre_dir.glob("*.wav"):
                    track_id = audio_file.stem
                    
                    # Copy audio file
                    shutil.copy2(audio_file, genre_audio_dir / audio_file.name)
                    
                    # Store features
                    if track_id in features:
                        track_features = features[track_id]
                        
                        # Create track group
                        track_group = audio_features.create_group(track_id)
                        
                        # Store different feature types
                        for feature_name, feature_data in track_features.items():
                            if isinstance(feature_data, (list, np.ndarray)):
                                track_group.create_dataset(
                                    feature_name,
                                    data=np.array(feature_data)
                                )
                            else:
                                track_group.attrs[feature_name] = feature_data
                    
                    dataset_info["genres"][genre] += 1
                    dataset_info["total_tracks"] += 1
        
        # Save metadata
        with open(self.metadata_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Save dataset info
        with open(self.root_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=4)
            
        return dataset_info