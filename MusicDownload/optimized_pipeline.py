#!/usr/bin/env python3
# optimized_pipeline.py

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import json
import sys

from music_download.pipeline_manager import PipelineManager
from music_download.audio_preprocessor import AudioPreprocessor
from music_download.feature_extractor import FeatureExtractor
from music_download.metadata_processor import MetadataProcessor
from music_download.quality_validator import AudioQualityValidator
from music_download.download_pipeline import MusicDownloadPipeline

class OptimizedPipeline:
    def __init__(self, config_path: str, project_root: Path):
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        # Initialize paths
        self.paths = {}
        self.setup_directories()
        
        # Initialize pipeline manager
        self.pipeline_manager = PipelineManager(self.project_root)
            
        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.metadata_processor = MetadataProcessor(self.paths['dataset_dir'])
        self.quality_validator = AudioQualityValidator()
        
        # Statistics
        self.stats = {
            "start_time": time.time(),
            "phases": {},
            "errors": []
        }

    def setup_directories(self):
        """Setup directory structure"""
        # Define default paths if not in config
        default_paths = {
            'downloads_dir': 'downloads',
            'dataset_dir': 'dataset',
            'features_dir': 'features',
            'processed_dir': 'processed',
            'temp_dir': 'temp'
        }
        
        # Use paths from config if available, otherwise use defaults
        config_paths = self.config.get('paths', {})
        for key, default_path in default_paths.items():
            path = self.project_root / config_paths.get(key, default_path)
            path.mkdir(parents=True, exist_ok=True)
            self.paths[key] = path
            
        print("\nInitialized directories:")
        for key, path in self.paths.items():
            print(f"- {key}: {path}")

    def process_downloads(self):
        """Download and validate new tracks"""
        try:
            download_pipeline = MusicDownloadPipeline(
                self.config_path,
                downloads_dir=self.paths['downloads_dir']
            )
            start_time = time.time()
            
            downloaded_files = download_pipeline.run(
                skip_existing=self.config['download_settings']['skip_existing'],
                check_modified=self.config['download_settings']['check_modified']
            )
            
            # If no new files were tracked, scan the downloads directory
            if not downloaded_files:
                print("Scanning downloads directory for unprocessed files...")
                downloaded_files = []
                for genre_dir in self.paths['downloads_dir'].iterdir():
                    if genre_dir.is_dir():
                        for audio_file in genre_dir.rglob("*.mp3"):
                            downloaded_files.append(audio_file)
                print(f"Found {len(downloaded_files)} existing files")
            
            if downloaded_files:
                # Validate new downloads
                print(f"Validating {len(downloaded_files)} files...")
                self.pipeline_manager.process_batch(
                    downloaded_files,
                    self.quality_validator,
                    phase="validation"
                )
            
            self.stats["phases"]["download"] = {
                "duration": time.time() - start_time,
                "files_processed": len(downloaded_files)
            }
            
            return downloaded_files
            
        except Exception as e:
            self.stats["errors"].append(f"Download error: {str(e)}")
            raise

    def extract_features(self, files):
        """Extract features with caching"""
        if not files:
            return []
            
        try:
            start_time = time.time()
            print(f"\nExtracting features from {len(files)} files...")
            
            processed_files = []
            for file_path in files:
                try:
                    # Check cache first
                    cached_features = self.pipeline_manager.get_cached_features(file_path)
                    if cached_features:
                        processed_files.append(cached_features)
                        print(f"Using cached features for {file_path.name}")
                        continue
                    
                    # Extract and cache features
                    print(f"Extracting features from {file_path.name}")
                    features = self.feature_extractor.extract_features(file_path)
                    self.pipeline_manager.cache_features(file_path, features)
                    processed_files.append(features)
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue
            
            # Save all features
            features_file = self.paths['features_dir'] / 'features.json'
            with open(features_file, 'w') as f:
                json.dump(processed_files, f, indent=4)
            
            self.stats["phases"]["feature_extraction"] = {
                "duration": time.time() - start_time,
                "files_processed": len(processed_files)
            }
            
            return processed_files
            
        except Exception as e:
            self.stats["errors"].append(f"Feature extraction error: {str(e)}")
            raise

    def run(self, skip_phases: list = None):
        """Run the optimized pipeline"""
        skip_phases = skip_phases or []
        
        try:
            # 1. Download Phase
            if "download" not in skip_phases:
                downloaded_files = self.process_downloads()
            else:
                # If skipping download, scan for existing files
                downloaded_files = []
                for genre_dir in self.paths['downloads_dir'].iterdir():
                    if genre_dir.is_dir():
                        for audio_file in genre_dir.rglob("*.mp3"):
                            downloaded_files.append(audio_file)
            
            if not downloaded_files:
                print("No files to process")
                return
            
            print(f"\nFound {len(downloaded_files)} files to process")
            
            # 2. Feature Extraction Phase
            if "features" not in skip_phases:
                features = self.extract_features(downloaded_files)
                print(f"Extracted features from {len(features)} files")
            
            # 3. Metadata Extraction
            if "metadata" not in skip_phases:
                start_time = time.time()
                print("\nExtracting metadata...")
                self.pipeline_manager.process_batch(
                    downloaded_files,
                    self.metadata_processor,
                    phase="metadata"
                )
                self.stats["phases"]["metadata"] = {
                    "duration": time.time() - start_time
                }
            
            # Final statistics
            self.stats["total_duration"] = time.time() - self.stats["start_time"]
            self.stats["final_state"] = self.pipeline_manager.get_processing_stats()
            
            # Save pipeline statistics
            with open(self.project_root / "pipeline_stats.json", 'w') as f:
                json.dump(self.stats, f, indent=4)
            
            # Cleanup if needed
            if not self.config['processing']['keep_temp_files']:
                self.pipeline_manager.clean_temp_files()
            
            print("\nPipeline completed successfully!")
            print(f"Total duration: {self.stats['total_duration']:.2f} seconds")
            print(f"Files processed: {self.stats['final_state']['total_files']}")
            
        except Exception as e:
            self.stats["errors"].append(f"Pipeline error: {str(e)}")
            with open(self.project_root / "pipeline_stats.json", 'w') as f:
                json.dump(self.stats, f, indent=4)
            raise

def main():
    parser = argparse.ArgumentParser(description="Optimized Music Processing Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--skip", nargs="+", choices=["download", "features", "metadata"],
                      help="Skip specified phases")
    
    args = parser.parse_args()
    
    try:
        # Convert project directory to absolute path
        project_dir = Path(args.project_dir).resolve()
        config_path = project_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)
        
        print(f"\nInitializing pipeline:")
        print(f"- Project directory: {project_dir}")
        print(f"- Config file: {config_path}")
        
        pipeline = OptimizedPipeline(
            config_path=str(config_path),
            project_root=project_dir
        )
        
        pipeline.run(skip_phases=args.skip or [])
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()