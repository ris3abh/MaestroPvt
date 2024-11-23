#!/usr/bin/env python3
# pipeline_manager.py

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from tqdm import tqdm
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class FileState:
    hash: str
    last_processed: float
    features_extracted: bool = False
    standardized: bool = False
    validated: bool = False
    validation_extracted: bool = False  # Add this line
    metadata_extracted: bool = False

class PipelineManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dataset_dir = project_root / "dataset"
        self.state_file = project_root / "pipeline_state.json"
        self.temp_dir = project_root / "temp"
        
        # Create directory structure
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state with validation
        state = self._load_state()
        if not self._validate_state(state):
            self.logger.warning("Invalid state detected, starting fresh")
            state = {}
        self.state = state
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self.progress = {
            "total_files": 0,
            "processed_files": 0,
            "current_phase": ""
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_root / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_state(self) -> Dict[str, FileState]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state_dict = json.load(f)
                return {
                    path: FileState(**state)
                    for path, state in state_dict.items()
                }
        return {}

    def _save_state(self):
        """Save pipeline state to disk"""
        state_dict = {
            path: {
                "hash": state.hash,
                "last_processed": state.last_processed,
                "features_extracted": state.features_extracted,
                "standardized": state.standardized,
                "validated": state.validated,
                "metadata_extracted": state.metadata_extracted
            }
            for path, state in self.state.items()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=4)

    def needs_processing(self, file_path: Path, phase: str) -> bool:
        """Check if file needs processing for given phase"""
        if str(file_path) not in self.state:
            return True
            
        state = self.state[str(file_path)]
        current_hash = self.get_file_hash(file_path)
        
        if current_hash != state.hash:
            return True
            
        return not getattr(state, f"{phase}_extracted")

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_file(self, processor: Any, file_path: Path, 
                    output_path: Optional[Path] = None, phase: str = "") -> bool:
        """Process a single file with tracking"""
        try:
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return False
                
            file_key = str(file_path)
            if not self.needs_processing(file_path, phase):
                self.logger.info(f"Skipping {file_path.name} - already processed")
                return False
                
            # Process file
            if output_path:
                result = processor.process_file(file_path, output_path)
            else:
                result = processor.process_file(file_path)
            
            # Update state with FileState dataclass
            self.state[file_key] = FileState(
                hash=self.get_file_hash(file_path),
                last_processed=time.time(),
                features_extracted=phase == "features" or (file_key in self.state and self.state[file_key].features_extracted),
                standardized=phase == "standardized" or (file_key in self.state and self.state[file_key].standardized),
                validated=phase == "validated" or (file_key in self.state and self.state[file_key].validated),
                metadata_extracted=phase == "metadata" or (file_key in self.state and self.state[file_key].metadata_extracted)
            )
            
            self._save_state()
            
            # Update progress
            with self._progress_lock:
                self.progress["processed_files"] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {str(e)}")
            return False

    def process_batch(self, file_paths: list, processor: Any, 
                     output_dir: Optional[Path] = None, 
                     phase: str = "", max_workers: int = 4):
        """Process a batch of files with parallel processing"""
        self.progress["total_files"] = len(file_paths)
        self.progress["processed_files"] = 0
        self.progress["current_phase"] = phase
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file_path in file_paths:
                output_path = None
                if output_dir:
                    output_path = output_dir / file_path.name
                    
                futures.append(
                    executor.submit(
                        self.process_file,
                        processor,
                        file_path,
                        output_path,
                        phase
                    )
                )
            
            # Monitor progress with tqdm
            with tqdm(total=len(file_paths), desc=f"Processing {phase}") as pbar:
                last_processed = 0
                while not all(f.done() for f in futures):
                    current_processed = self.progress["processed_files"]
                    if current_processed > last_processed:
                        pbar.update(current_processed - last_processed)
                        last_processed = current_processed
                    time.sleep(0.1)
                    
        self._save_state()
        return [f.result() for f in futures]

    def clean_temp_files(self):
        """Clean temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()

    def get_cached_features(self, file_path: Path) -> Optional[Dict]:
        """Get cached features if available"""
        cache_path = self.temp_dir / f"{file_path.stem}_features.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def cache_features(self, file_path: Path, features: Dict):
        """Cache extracted features"""
        cache_path = self.temp_dir / f"{file_path.stem}_features.json"
        with open(cache_path, 'w') as f:
            json.dump(features, f)

    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state data"""
        required_fields = {'hash', 'last_processed', 'features_extracted', 
                         'standardized', 'validated', 'metadata_extracted'}
        try:
            for file_state in state.values():
                if not all(hasattr(file_state, field) for field in required_fields):
                    return False
            return True
        except:
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "total_files": len(self.state),
            "features_extracted": sum(1 for s in self.state.values() if s.features_extracted),
            "standardized": sum(1 for s in self.state.values() if s.standardized),
            "validated": sum(1 for s in self.state.values() if s.validated),
            "metadata_extracted": sum(1 for s in self.state.values() if s.metadata_extracted)
        }
        return stats