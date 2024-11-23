#!/usr/bin/env python3
# optimized_pipeline.py

import gc
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any
import multiprocessing
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
        self.quality_validator = AudioQualityValidator(chunk_duration=10.0,  # Process 10 seconds at a time
                                                        max_memory_gb=4.0     # Limit memory usage per process
                                                    )
        
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
        """Download and validate new tracks with improved memory management and parallel processing"""
        try:
            download_pipeline = MusicDownloadPipeline(
                self.config_path,
                downloads_dir=self.paths['downloads_dir']
            )
            start_time = time.time()
            
            # Initialize the improved validator with configurable settings
            quality_validator = AudioQualityValidator(
                chunk_duration=self.config['validation_thresholds']['processing'].get('chunk_duration', 10.0),
                max_memory_gb=self.config['validation_thresholds']['processing'].get('max_memory_gb', 4.0),
                min_duration=self.config['validation_thresholds']['audio'].get('min_duration', 60.0),
                min_sample_rate=self.config['validation_thresholds']['audio'].get('min_sample_rate', 44100),
                min_dynamic_range=self.config['validation_thresholds'].get('min_dynamic_range', 20.0),
                max_clipping_ratio=self.config['validation_thresholds'].get('max_clipping_ratio', 0.01)
            )
            
            # Download new tracks
            downloaded_files = download_pipeline.run(
                skip_existing=self.config['download_settings']['skip_existing'],
                check_modified=self.config['download_settings']['check_modified']
            )
            
            # Scan for existing files if no new downloads
            if not downloaded_files:
                logging.info("Scanning downloads directory for unprocessed files...")
                downloaded_files = []
                for genre_dir in self.paths['downloads_dir'].iterdir():
                    if genre_dir.is_dir():
                        downloaded_files.extend(list(genre_dir.rglob("*.mp3")))
                logging.info(f"Found {len(downloaded_files)} existing files")
            
            validation_results = None
            if downloaded_files:
                # Determine optimal number of workers based on CPU cores and config
                num_workers = min(
                    len(downloaded_files),
                    self.config['validation_thresholds']['processing'].get(
                        'max_workers',
                        max(1, multiprocessing.cpu_count() - 1)
                    )
                )
                
                logging.info(f"Validating {len(downloaded_files)} files using {num_workers} workers...")
                
                # Process validation in batches to manage memory
                batch_size = self.config['validation_thresholds']['processing'].get('batch_size', 100)
                
                # Process files in batches
                total_results = {
                    "files": {},
                    "summary": {
                        "total_files": 0,
                        "passed_files": 0,
                        "failed_files": 0,
                        "average_metrics": {}
                    }
                }
                
                for i in range(0, len(downloaded_files), batch_size):
                    batch = downloaded_files[i:i + batch_size]
                    # Pass the batch directly as a list of files
                    batch_results = quality_validator.validate_dataset(
                        batch,  # Now passing list of files directly
                        num_workers=num_workers
                    )
                    
                    # Merge batch results
                    total_results["files"].update(batch_results["files"])
                    total_results["summary"]["total_files"] += batch_results["summary"]["total_files"]
                    total_results["summary"]["passed_files"] += batch_results["summary"]["passed_files"]
                    total_results["summary"]["failed_files"] += batch_results["summary"]["failed_files"]
                    
                    # Merge average metrics
                    if not total_results["summary"]["average_metrics"]:
                        total_results["summary"]["average_metrics"] = batch_results["summary"]["average_metrics"]
                    else:
                        for metric, value in batch_results["summary"]["average_metrics"].items():
                            if metric in total_results["summary"]["average_metrics"]:
                                total_results["summary"]["average_metrics"][metric] = (
                                    total_results["summary"]["average_metrics"][metric] + value
                                ) / 2
                    
                    # Force garbage collection between batches
                    gc.collect()
                
                validation_results = total_results
                
                # Log validation summary
                logging.info(f"Validation complete: "
                            f"{validation_results['summary']['passed_files']} passed, "
                            f"{validation_results['summary']['failed_files']} failed")
            
            # Update statistics
            self.stats["phases"]["download"] = {
                "duration": time.time() - start_time,
                "files_processed": len(downloaded_files),
                "validation_results": validation_results
            }
            
            return downloaded_files
            
        except Exception as e:
            error_msg = f"Download processing error: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.stats["errors"].append(error_msg)
            raise

        finally:
            # Ensure cleanup of any remaining resources
            gc.collect()

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