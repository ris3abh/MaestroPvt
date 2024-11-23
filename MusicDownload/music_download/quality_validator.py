#!/usr/bin/env python3
# quality_validator.py

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
import gc
import contextlib
import multiprocessing as mp
from tqdm import tqdm
import logging

class AudioQualityValidator:
    def __init__(self, 
                 min_duration: float = 60.0,
                 min_sample_rate: int = 44100,
                 min_bit_depth: int = 16,
                 min_dynamic_range: float = 10.0,
                 max_clipping_ratio: float = 0.01,
                 chunk_duration: float = 10.0,
                 max_memory_gb: float = 4.0):
        """
        Initialize the audio quality validator with improved memory management.
        
        Args:
            chunk_duration: Duration of audio chunks to process at once (seconds)
            max_memory_gb: Maximum memory usage allowed per process (GB)
        """
        self.min_duration = min_duration
        self.min_sample_rate = min_sample_rate
        self.min_bit_depth = min_bit_depth
        self.min_dynamic_range = min_dynamic_range
        self.max_clipping_ratio = max_clipping_ratio
        self.chunk_duration = chunk_duration
        self.max_memory_bytes = int(max_memory_gb * 1e9)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Suppress warnings
        warnings.filterwarnings('ignore')

    def _check_memory_usage(self) -> None:
        """Monitor memory usage and force garbage collection if needed"""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            
            if memory_usage > self.max_memory_bytes:
                logging.warning(f"High memory usage detected: {memory_usage / 1e9:.2f} GB")
                gc.collect()
        except ImportError:
            pass

    @contextlib.contextmanager
    def resource_manager(self):
        """Enhanced context manager for handling resources"""
        try:
            yield
        except MemoryError:
            logging.error("Memory error encountered - forcing garbage collection")
            gc.collect()
            raise
        finally:
            # Clean up resources
            gc.collect()
    
    def _process_audio_chunk(self, chunk: np.ndarray, sr: int) -> Dict[str, float]:
        """Process a single chunk of audio data"""
        metrics = {}
        
        with self.resource_manager():
            # Basic measurements
            metrics['rms'] = float(np.sqrt(np.mean(chunk ** 2)))
            metrics['peak'] = float(np.max(np.abs(chunk)))
            metrics['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
            
            # Spectral features (using smaller windows for efficiency)
            n_fft = min(2048, len(chunk))
            spectral = librosa.stft(chunk, n_fft=n_fft)
            
            if len(spectral) > 0:
                metrics['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=np.abs(spectral), sr=sr)))
                metrics['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=np.abs(spectral), sr=sr)))
            
            self._check_memory_usage()
            return metrics

    def check_audio_quality(self, audio_path: Path) -> Tuple[bool, List[str], Dict[str, float]]:
        """Validate audio file quality with improved error handling and memory management"""
        issues = []
        metrics = {}
        
        try:
            with self.resource_manager():
                # Get audio duration first
                duration = librosa.get_duration(path=str(audio_path))
                metrics['duration'] = float(duration)
                
                if duration < self.min_duration:
                    issues.append(f"Duration too short: {duration:.1f}s < {self.min_duration}s")
                    return False, issues, metrics
                
                # Process audio in chunks
                chunk_samples = None
                accumulated_metrics = []
                total_clipping = 0
                total_samples = 0
                
                for chunk_start in np.arange(0, duration, self.chunk_duration):
                    chunk_duration = min(self.chunk_duration, duration - chunk_start)
                    
                    # Load chunk
                    y, sr = librosa.load(
                        audio_path,
                        sr=None,
                        offset=chunk_start,
                        duration=chunk_duration
                    )
                    
                    if chunk_samples is None:
                        chunk_samples = len(y)
                        metrics['sample_rate'] = int(sr)
                        
                        if sr < self.min_sample_rate:
                            issues.append(f"Sample rate too low: {sr} < {self.min_sample_rate}")
                    
                    # Process chunk
                    chunk_metrics = self._process_audio_chunk(y, sr)
                    accumulated_metrics.append(chunk_metrics)
                    
                    # Clipping analysis
                    total_clipping += np.sum(np.abs(y) >= 0.99)
                    total_samples += len(y)
                    
                    # Clear memory
                    del y
                    self._check_memory_usage()
                
                # Aggregate metrics
                metrics.update(self._aggregate_metrics(accumulated_metrics))
                
                # Calculate final metrics
                clipping_ratio = float(total_clipping / total_samples)
                metrics['clipping_ratio'] = clipping_ratio
                
                if clipping_ratio > self.max_clipping_ratio:
                    issues.append(f"Excessive clipping: {clipping_ratio*100:.1f}% of samples")
                
                # Dynamic range
                if 'peak' in metrics and 'rms' in metrics:
                    dynamic_range = float(20 * np.log10(metrics['peak'] / (metrics['rms'] + 1e-6)))
                    metrics['dynamic_range'] = dynamic_range
                    
                    if dynamic_range < self.min_dynamic_range:
                        issues.append(f"Low dynamic range: {dynamic_range:.1f}dB")
        
        except Exception as e:
            error_msg = f"Error analyzing file: {str(e)}"
            logging.error(error_msg, exc_info=True)
            issues.append(error_msg)
            return False, issues, metrics
        
        return len(issues) == 0, issues, metrics

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple chunks"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            try:
                values = [m[key] for m in metrics_list if key in m]
                aggregated[key] = float(np.mean(values))
            except Exception as e:
                logging.warning(f"Could not aggregate metric {key}: {str(e)}")
        
        return aggregated

    def _worker_process(self, audio_path: Path) -> Dict[str, Any]:
        """Worker process for parallel processing"""
        try:
            return self.process_file(audio_path)
        except Exception as e:
            logging.error(f"Worker process error for {audio_path}: {str(e)}")
            return {
                "passed": False,
                "issues": [f"Processing error: {str(e)}"],
                "metrics": {},
                "filename": audio_path.name,
                "validation_extracted": False
            }

    def validate_dataset(self, downloads_dir: Union[Path, List[Path]], num_workers: Optional[int] = None) -> Dict[str, Dict]:
        """
        Validate all audio files in parallel with progress tracking.
        
        Args:
            downloads_dir: Either a Path to directory containing audio files,
                        or a List[Path] of audio files to validate
            num_workers: Number of parallel workers to use
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        validation_results = {
            "files": {},
            "summary": {
                "total_files": 0,
                "passed_files": 0,
                "failed_files": 0,
                "average_metrics": {}
            }
        }
        
        # Handle both directory path and list of files
        if isinstance(downloads_dir, (str, Path)):
            # If it's a directory path, collect all audio files
            downloads_dir = Path(downloads_dir)
            audio_files = []
            for genre_dir in downloads_dir.iterdir():
                if genre_dir.is_dir():
                    audio_files.extend(genre_dir.glob("*.mp3"))
        else:
            # If it's already a list of files, use it directly
            audio_files = downloads_dir
        
        total_files = len(audio_files)
        validation_results["summary"]["total_files"] = total_files
        
        if total_files == 0:
            logging.warning("No audio files found to validate")
            return validation_results
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._worker_process, audio_files),
                total=total_files,
                desc="Processing validation"
            ))
        
        # Aggregate results
        all_metrics = []
        for result in results:
            file_path = result.get("filename", "unknown")
            if result.get("passed", False):
                validation_results["summary"]["passed_files"] += 1
            else:
                validation_results["summary"]["failed_files"] += 1
                validation_results["files"][file_path] = result
            
            if result.get("metrics"):
                all_metrics.append(result["metrics"])
        
        # Calculate average metrics
        if all_metrics:
            validation_results["summary"]["average_metrics"] = self._aggregate_metrics(all_metrics)
        
        return validation_results

    def process_file(self, audio_path: Path) -> Dict[str, Any]:
        """Process and validate a single audio file with improved error handling"""
        with self.resource_manager():
            try:
                passed, issues, metrics = self.check_audio_quality(audio_path)
                
                validation_result = {
                    "passed": passed,
                    "issues": issues,
                    "metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat(),
                    "filename": audio_path.name,
                    "validation_extracted": True
                }
                
                logging.info(f"{'✓' if passed else '✗'} {audio_path.name}")
                if not passed:
                    for issue in issues:
                        logging.warning(f"  - {issue}")
                
                return validation_result
                
            except Exception as e:
                error_msg = f"Error validating {audio_path.name}: {str(e)}"
                logging.error(error_msg, exc_info=True)
                return {
                    "passed": False,
                    "issues": [error_msg],
                    "metrics": {},
                    "timestamp": datetime.utcnow().isoformat(),
                    "filename": audio_path.name,
                    "validation_extracted": False
                }