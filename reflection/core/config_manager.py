#!/usr/bin/env python3
"""
Configuration Manager
Handles loading, validation, and management of reflection system configuration.

Author: Manus AI
Version: 1.0.0
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ReflectionConfig:
    """Configuration for reflection system"""
    agent_name: str
    agent_type: str
    repository: str
    schedule: str = "daily"
    enabled_modules: list = None
    save_directory: str = "data/reflection"
    auto_optimize: bool = False
    require_approval: bool = True
    
    def __post_init__(self):
        if self.enabled_modules is None:
            self.enabled_modules = ['inventory', 'documentation', 'reporting']
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConfigManager:
    """
    Manages configuration for the Universal Self-Reflection System
    """
    
    DEFAULT_CONFIG = {
        'agent': {
            'name': 'Unknown Agent',
            'type': 'generic',
            'repository': None
        },
        'reflection': {
            'schedule': 'daily',
            'modules': ['inventory', 'documentation', 'reporting']
        },
        'inventory': {
            'track_files': True,
            'track_dependencies': True,
            'track_versions': True,
            'exclude_patterns': ['__pycache__', '*.pyc', '.git', 'node_modules']
        },
        'taxonomy': {
            'naming_convention': 'snake_case',
            'enforce_standards': True,
            'auto_rename': False,
            'system_suffix': '-system'
        },
        'documentation': {
            'auto_generate_readme': True,
            'auto_generate_changelog': True,
            'generate_diagrams': True,
            'update_on_change': True
        },
        'indexing': {
            'create_search_index': True,
            'extract_metadata': True,
            'build_knowledge_graph': True
        },
        'evolution': {
            'auto_optimize': False,
            'learning_rate': 0.1,
            'performance_threshold': 0.7,
            'track_learning': True
        },
        'reporting': {
            'formats': ['json', 'markdown'],
            'include_visualizations': True,
            'save_to': 'data/reflection/'
        },
        'validation': {
            'triple_check': True,
            'require_external_validation': False,
            'validation_threshold': 0.95
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            config_dict: Configuration dictionary
        """
        self.config = self._load_config(config_path, config_dict)
        self._validate_config()
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict:
        """Load configuration from file or dictionary"""
        
        # Start with default config
        config = self.DEFAULT_CONFIG.copy()
        
        # Load from dictionary if provided
        if config_dict:
            config = self._merge_configs(config, config_dict)
            return config
        
        # Load from file if provided
        if config_path:
            path = Path(config_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
            
            config = self._merge_configs(config, file_config)
        
        return config
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """Validate configuration for required fields and correct types"""
        
        # Validate agent section
        if 'agent' not in self.config:
            raise ValueError("Configuration must include 'agent' section")
        
        required_agent_fields = ['name', 'type']
        for field in required_agent_fields:
            if field not in self.config['agent']:
                raise ValueError(f"Agent configuration missing required field: {field}")
        
        # Validate reflection section
        if 'reflection' not in self.config:
            raise ValueError("Configuration must include 'reflection' section")
        
        if 'modules' not in self.config['reflection']:
            raise ValueError("Reflection configuration must include 'modules' list")
        
        # Validate module configurations
        valid_modules = ['inventory', 'taxonomy', 'documentation', 'indexing', 'evolution', 'reporting']
        for module in self.config['reflection']['modules']:
            if module not in valid_modules:
                raise ValueError(f"Invalid module: {module}. Must be one of {valid_modules}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filepath: str):
        """Save configuration to file"""
        path = Path(filepath)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def to_dict(self) -> Dict:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def get_module_config(self, module_name: str) -> Dict:
        """Get configuration for a specific module"""
        return self.config.get(module_name, {})
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled"""
        return module_name in self.config['reflection']['modules']
    
    def enable_module(self, module_name: str):
        """Enable a reflection module"""
        if module_name not in self.config['reflection']['modules']:
            self.config['reflection']['modules'].append(module_name)
    
    def disable_module(self, module_name: str):
        """Disable a reflection module"""
        if module_name in self.config['reflection']['modules']:
            self.config['reflection']['modules'].remove(module_name)
    
    def get_agent_name(self) -> str:
        """Get agent name from configuration"""
        return self.config['agent']['name']
    
    def get_agent_type(self) -> str:
        """Get agent type from configuration"""
        return self.config['agent']['type']
    
    def get_repository(self) -> Optional[str]:
        """Get repository URL from configuration"""
        return self.config['agent'].get('repository')
    
    def get_schedule(self) -> str:
        """Get reflection schedule"""
        return self.config['reflection']['schedule']
    
    def get_save_directory(self) -> str:
        """Get directory for saving reflection results"""
        return self.config['reporting']['save_to']


if __name__ == '__main__':
    # Example usage
    config = ConfigManager()
    print("Default Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
