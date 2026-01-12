#!/usr/bin/env python3
"""
Asset Tracker Module
Tracks all agent assets including files, models, configurations, and data.

Author: Manus AI
Version: 1.0.0
"""

import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class AssetTracker:
    """
    Tracks and catalogs all agent assets
    """
    
    ASSET_TYPES = {
        '.py': 'code',
        '.js': 'code',
        '.ts': 'code',
        '.java': 'code',
        '.go': 'code',
        '.rs': 'code',
        '.json': 'config',
        '.yaml': 'config',
        '.yml': 'config',
        '.toml': 'config',
        '.ini': 'config',
        '.md': 'documentation',
        '.txt': 'documentation',
        '.rst': 'documentation',
        '.csv': 'data',
        '.parquet': 'data',
        '.db': 'data',
        '.sqlite': 'data',
        '.pkl': 'model',
        '.h5': 'model',
        '.pt': 'model',
        '.pth': 'model',
        '.onnx': 'model',
        '.pb': 'model'
    }
    
    DEFAULT_EXCLUDE_PATTERNS = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.git',
        '.gitignore',
        'node_modules',
        'venv',
        'env',
        '.env',
        '.vscode',
        '.idea',
        '*.log',
        '.DS_Store'
    ]
    
    def __init__(self, root_dir: str, config: Optional[Dict] = None):
        """
        Initialize asset tracker
        
        Args:
            root_dir: Root directory to scan
            config: Configuration dictionary
        """
        self.root_dir = Path(root_dir)
        self.config = config or {}
        self.exclude_patterns = self.config.get('exclude_patterns', self.DEFAULT_EXCLUDE_PATTERNS)
        self.assets = []
        self.inventory = {}
    
    def scan_assets(self) -> List[Dict[str, Any]]:
        """
        Scan directory tree and catalog all assets
        
        Returns:
            List of asset dictionaries
        """
        self.assets = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            
            for file in files:
                if self._should_exclude(file):
                    continue
                
                filepath = Path(root) / file
                asset = self._create_asset_entry(filepath)
                self.assets.append(asset)
        
        return self.assets
    
    def _should_exclude(self, name: str) -> bool:
        """Check if file/directory should be excluded"""
        for pattern in self.exclude_patterns:
            if pattern.startswith('*'):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern or name.startswith(pattern):
                return True
        return False
    
    def _create_asset_entry(self, filepath: Path) -> Dict[str, Any]:
        """Create asset entry with metadata"""
        
        # Get file stats
        stats = filepath.stat()
        
        # Determine asset type
        asset_type = self._determine_asset_type(filepath)
        
        # Calculate checksum for code and config files
        checksum = None
        if asset_type in ['code', 'config']:
            checksum = self._calculate_checksum(filepath)
        
        # Get relative path from root
        try:
            relative_path = filepath.relative_to(self.root_dir)
        except ValueError:
            relative_path = filepath
        
        # Extract additional metadata
        metadata = self._extract_metadata(filepath, asset_type)
        
        return {
            'path': str(relative_path),
            'absolute_path': str(filepath),
            'type': asset_type,
            'size_bytes': stats.st_size,
            'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'checksum': checksum,
            'metadata': metadata
        }
    
    def _determine_asset_type(self, filepath: Path) -> str:
        """Determine asset type from file extension"""
        ext = filepath.suffix.lower()
        return self.ASSET_TYPES.get(ext, 'other')
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return f"sha256:{sha256_hash.hexdigest()}"
        except Exception:
            return None
    
    def _extract_metadata(self, filepath: Path, asset_type: str) -> Dict[str, Any]:
        """Extract additional metadata based on asset type"""
        metadata = {}
        
        try:
            if asset_type == 'code':
                # Count lines, functions, classes for code files
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    metadata['lines'] = len(lines)
                    metadata['functions'] = content.count('def ') if filepath.suffix == '.py' else None
                    metadata['classes'] = content.count('class ') if filepath.suffix == '.py' else None
            
            elif asset_type == 'config':
                # Try to parse config files
                if filepath.suffix == '.json':
                    with open(filepath, 'r') as f:
                        config_data = json.load(f)
                        metadata['keys'] = list(config_data.keys()) if isinstance(config_data, dict) else None
            
            elif asset_type == 'model':
                # Model-specific metadata
                metadata['model_format'] = filepath.suffix[1:]  # Remove the dot
        
        except Exception:
            pass
        
        return metadata
    
    def generate_inventory(self) -> Dict[str, Any]:
        """
        Generate comprehensive inventory report
        
        Returns:
            Inventory dictionary with statistics and categorization
        """
        if not self.assets:
            self.scan_assets()
        
        # Aggregate statistics
        stats = {
            'total_assets': len(self.assets),
            'total_size_bytes': sum(a['size_bytes'] for a in self.assets),
            'by_type': defaultdict(int),
            'by_directory': defaultdict(int),
            'largest_files': [],
            'recently_modified': []
        }
        
        # Count by type
        for asset in self.assets:
            stats['by_type'][asset['type']] += 1
            
            # Count by directory
            directory = str(Path(asset['path']).parent)
            stats['by_directory'][directory] += 1
        
        # Find largest files
        sorted_by_size = sorted(self.assets, key=lambda x: x['size_bytes'], reverse=True)
        stats['largest_files'] = [
            {
                'path': a['path'],
                'size_bytes': a['size_bytes'],
                'type': a['type']
            }
            for a in sorted_by_size[:10]
        ]
        
        # Find recently modified files
        sorted_by_time = sorted(self.assets, key=lambda x: x['last_modified'], reverse=True)
        stats['recently_modified'] = [
            {
                'path': a['path'],
                'last_modified': a['last_modified'],
                'type': a['type']
            }
            for a in sorted_by_time[:10]
        ]
        
        # Convert defaultdicts to regular dicts
        stats['by_type'] = dict(stats['by_type'])
        stats['by_directory'] = dict(stats['by_directory'])
        
        self.inventory = {
            'timestamp': datetime.now().isoformat(),
            'root_directory': str(self.root_dir),
            'statistics': stats,
            'assets': self.assets
        }
        
        return self.inventory
    
    def get_assets_by_type(self, asset_type: str) -> List[Dict[str, Any]]:
        """Get all assets of a specific type"""
        return [a for a in self.assets if a['type'] == asset_type]
    
    def get_assets_by_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Get all assets in a specific directory"""
        return [a for a in self.assets if str(Path(a['path']).parent) == directory]
    
    def find_asset(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find an asset by filename"""
        for asset in self.assets:
            if Path(asset['path']).name == filename:
                return asset
        return None
    
    def detect_changes(self, previous_inventory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect changes since previous inventory
        
        Args:
            previous_inventory: Previous inventory dictionary
        
        Returns:
            Dictionary with added, modified, and deleted assets
        """
        current_assets = {a['path']: a for a in self.assets}
        previous_assets = {a['path']: a for a in previous_inventory.get('assets', [])}
        
        added = []
        modified = []
        deleted = []
        
        # Find added and modified
        for path, asset in current_assets.items():
            if path not in previous_assets:
                added.append(asset)
            else:
                prev_asset = previous_assets[path]
                if asset['checksum'] != prev_asset['checksum']:
                    modified.append({
                        'path': path,
                        'old_checksum': prev_asset['checksum'],
                        'new_checksum': asset['checksum'],
                        'old_modified': prev_asset['last_modified'],
                        'new_modified': asset['last_modified']
                    })
        
        # Find deleted
        for path in previous_assets:
            if path not in current_assets:
                deleted.append(previous_assets[path])
        
        return {
            'added': added,
            'modified': modified,
            'deleted': deleted,
            'total_changes': len(added) + len(modified) + len(deleted)
        }
    
    def save_inventory(self, filepath: str):
        """Save inventory to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.inventory, f, indent=2)
    
    def load_inventory(self, filepath: str) -> Dict[str, Any]:
        """Load inventory from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


if __name__ == '__main__':
    # Example usage
    tracker = AssetTracker('.')
    inventory = tracker.generate_inventory()
    print(f"Total assets: {inventory['statistics']['total_assets']}")
    print(f"By type: {inventory['statistics']['by_type']}")
