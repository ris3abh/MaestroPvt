#!/usr/bin/env python3
# download_pipeline.py

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from .downloader import setup_config, generate_playlist

class MusicDownloadPipeline:
    def __init__(self, config_path: str, downloads_dir: Path = None):
        """
        Initialize the music download pipeline
        
        Args:
            config_path (str): Path to JSON config file containing playlist URLs and genres
            downloads_dir (Path): Directory for downloads
        """
        self.config_path = config_path
        self.downloads_dir = Path(downloads_dir) if downloads_dir else Path("downloads")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.downloads_dir / ".download_state.json"
        self.load_config()
        self.load_state()

    def load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
                
            if "download_settings" not in self.config:
                raise ValueError("Missing 'download_settings' in config")
            if "playlists" not in self.config["download_settings"]:
                raise ValueError("Missing 'playlists' in download_settings")
                
            return self.config
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in config file")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def load_state(self):
        """Load download state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {"downloaded_files": {}}

    def save_state(self):
        """Save download state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def needs_download(self, url: str, file_path: Path, check_modified: bool) -> bool:
        """Check if file needs to be downloaded"""
        if not file_path.exists():
            return True
            
        if not check_modified:
            return False
            
        current_hash = self.get_file_hash(file_path)
        return self.state["downloaded_files"].get(url) != current_hash

    def run(self, skip_existing: bool = True, check_modified: bool = True) -> List[Path]:
        """Run the download pipeline"""
        downloaded_files = []
        
        for playlist in self.config["download_settings"]["playlists"]:
            genre = playlist["genre"]
            subgenre = playlist.get("subgenre", "")
            
            # Create genre directory
            genre_dir = self.downloads_dir / genre
            if subgenre:
                genre_dir = genre_dir / subgenre
            genre_dir.mkdir(parents=True, exist_ok=True)
            
            # Change to genre directory for download
            original_dir = os.getcwd()
            os.chdir(str(genre_dir))

            try:
                # Setup download configuration
                download_config = {
                    "url": playlist["url"],
                    "reverse_playlist": False,
                    "use_title": True,
                    "use_uploader": True,
                    "use_playlist_name": True,
                    # Use download_settings instead of audio_settings
                    "audio_format": self.config["download_settings"]["audio_format"],
                    "audio_codec": self.config["download_settings"]["audio_codec"],
                    "audio_quality": self.config["download_settings"]["audio_quality"],
                    "name_format": self.config["download_settings"]["name_format"],
                    "include_metadata": self.config["download_settings"]["include_metadata"]
                }
                
                # Generate playlist
                generate_playlist(
                    setup_config(download_config),
                    ".playlist_config.json",
                    update=False,
                    force_update=False,
                    regenerate_metadata=False,
                    single_playlist=True,
                    current_playlist_name=None,
                    track_num_to_update=None
                )
                
                # Track downloaded files
                audio_codec = self.config["download_settings"]["audio_codec"]
                for file_path in genre_dir.glob(f"*.{audio_codec}"):
                    if skip_existing and not self.needs_download(playlist["url"], file_path, check_modified):
                        print(f"Skipping existing file: {file_path.name}")
                        continue
                        
                    downloaded_files.append(file_path)
                    self.state["downloaded_files"][playlist["url"]] = self.get_file_hash(file_path)

            finally:
                os.chdir(original_dir)
        
        # Save state
        self.save_state()
        
        return downloaded_files

    def cleanup(self):
        """Clean up temporary files"""
        for genre_dir in self.downloads_dir.iterdir():
            if genre_dir.is_dir():
                config_file = genre_dir / ".playlist_config.json"
                if config_file.exists():
                    config_file.unlink()