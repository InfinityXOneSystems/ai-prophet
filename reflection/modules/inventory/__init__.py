#!/usr/bin/env python3
"""
Inventory Module
Comprehensive asset and dependency tracking for AI agents.

Author: Manus AI
Version: 1.0.0
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .asset_tracker import AssetTracker
from .dependency_mapper import DependencyMapper


class InventoryModule:
    """
    Main inventory module that orchestrates asset tracking and dependency mapping
    """
    
    def __init__(self, agent, config: Optional[Dict] = None):
        """
        Initialize inventory module
        
        Args:
            agent: Agent instance implementing AgentInterface
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}
        
        # Get agent repository path
        agent_metadata = agent.get_metadata()
        self.root_dir = self.config.get('root_dir', '.')
        
        # Initialize sub-modules
        self.asset_tracker = AssetTracker(self.root_dir, self.config)
        self.dependency_mapper = DependencyMapper(self.root_dir, self.config)
        
        self.results = None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete inventory analysis
        
        Returns:
            Dictionary with inventory results
        """
        print("   📦 Scanning assets...")
        assets = self.asset_tracker.scan_assets()
        inventory = self.asset_tracker.generate_inventory()
        
        print(f"      Found {len(assets)} assets")
        
        # Analyze dependencies if enabled
        dependencies = None
        if self.config.get('track_dependencies', True):
            print("   🔗 Analyzing dependencies...")
            dependencies = self.dependency_mapper.analyze_dependencies()
            print(f"      Found {len(dependencies['external_dependencies'])} external packages")
        
        # Check for version control if enabled
        version_info = None
        if self.config.get('track_versions', True):
            print("   📌 Checking version control...")
            version_info = self._get_version_info()
        
        # Compile results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'inventory': inventory,
            'dependencies': dependencies,
            'version_info': version_info
        }
        
        return self.results
    
    def _get_version_info(self) -> Dict[str, Any]:
        """Get version control information (Git)"""
        import subprocess
        
        version_info = {
            'has_git': False,
            'current_branch': None,
            'latest_commit': None,
            'uncommitted_changes': False,
            'total_commits': 0
        }
        
        try:
            # Check if git repo exists
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version_info['has_git'] = True
                
                # Get current branch
                result = subprocess.run(
                    ['git', 'branch', '--show-current'],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_info['current_branch'] = result.stdout.strip()
                
                # Get latest commit
                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%H|%an|%ae|%at|%s'],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split('|')
                    if len(parts) == 5:
                        version_info['latest_commit'] = {
                            'hash': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'timestamp': parts[3],
                            'message': parts[4]
                        }
                
                # Check for uncommitted changes
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_info['uncommitted_changes'] = len(result.stdout.strip()) > 0
                
                # Get total commits
                result = subprocess.run(
                    ['git', 'rev-list', '--count', 'HEAD'],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version_info['total_commits'] = int(result.stdout.strip())
        
        except Exception as e:
            print(f"      Warning: Could not get Git info: {e}")
        
        return version_info
    
    def get_asset_summary(self) -> Dict[str, Any]:
        """Get summary of assets"""
        if not self.results:
            return {}
        
        inventory = self.results.get('inventory', {})
        stats = inventory.get('statistics', {})
        
        return {
            'total_assets': stats.get('total_assets', 0),
            'total_size_mb': stats.get('total_size_bytes', 0) / (1024 * 1024),
            'by_type': stats.get('by_type', {}),
            'largest_files': stats.get('largest_files', [])[:5]
        }
    
    def get_dependency_summary(self) -> Dict[str, Any]:
        """Get summary of dependencies"""
        if not self.results:
            return {}
        
        dependencies = self.results.get('dependencies', {})
        if not dependencies:
            return {}
        
        return {
            'external_packages': len(dependencies.get('external_dependencies', [])),
            'internal_modules': dependencies.get('statistics', {}).get('total_internal_files', 0),
            'top_external': dependencies.get('external_dependencies', [])[:10]
        }


__all__ = ['InventoryModule', 'AssetTracker', 'DependencyMapper']
