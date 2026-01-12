#!/usr/bin/env python3
"""
Agent Interface
Abstract interface that all agents must implement to use the reflection system.

Author: Manus AI
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentMetadata:
    """Standard metadata structure for agents"""
    name: str
    version: str
    type: str  # e.g., 'trading_agent', 'scraper_agent', 'analysis_agent'
    repository: str
    capabilities: List[str]
    description: str
    author: str = "Manus AI"
    created_at: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Standard performance metrics structure"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_pct: Optional[float] = None
    custom_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AgentAsset:
    """Represents a single agent asset (file, model, config, etc.)"""
    path: str
    type: str  # 'code', 'config', 'data', 'model', 'documentation'
    size_bytes: int
    last_modified: str
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentInterface(ABC):
    """
    Abstract interface for AI agents to integrate with Universal Self-Reflection System
    
    Any agent that wants to use the reflection system must implement this interface.
    """
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return agent metadata
        
        Returns:
            Dictionary containing:
            - name: Agent name
            - version: Current version
            - type: Agent type/category
            - repository: GitHub repository URL
            - capabilities: List of capabilities
            - description: Brief description
            - created_at: Creation timestamp
            - last_updated: Last update timestamp
        
        Example:
            {
                "name": "AI Prophet",
                "version": "1.0.0",
                "type": "trading_agent",
                "repository": "InfinityXOneSystems/ai-prophet",
                "capabilities": ["prediction", "trading", "learning"],
                "description": "Financial prediction and trading agent",
                "created_at": "2026-01-11T00:00:00Z",
                "last_updated": "2026-01-12T00:00:00Z"
            }
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Return current performance metrics
        
        Returns:
            Dictionary containing relevant performance metrics:
            - accuracy: Prediction/task accuracy (0.0-1.0)
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - latency_ms: Average latency in milliseconds
            - throughput: Operations per second
            - error_rate: Error rate (0.0-1.0)
            - uptime_pct: Uptime percentage
            - custom_metrics: Dict of agent-specific metrics
        
        Example:
            {
                "accuracy": 0.7143,
                "win_rate": 0.65,
                "total_predictions": 128,
                "avg_confidence": 0.7143,
                "latency_ms": 250,
                "error_rate": 0.02,
                "custom_metrics": {
                    "portfolio_value": 1000000.0,
                    "total_pnl": 15234.50
                }
            }
        """
        pass
    
    @abstractmethod
    def get_assets(self) -> List[Dict[str, Any]]:
        """
        Return list of all agent assets
        
        Returns:
            List of asset dictionaries, each containing:
            - path: File path or identifier
            - type: Asset type ('code', 'config', 'data', 'model', 'documentation')
            - size_bytes: Size in bytes
            - last_modified: Last modification timestamp
            - checksum: Optional checksum for integrity
            - metadata: Optional additional metadata
        
        Example:
            [
                {
                    "path": "src/core/prophet_core.py",
                    "type": "code",
                    "size_bytes": 15234,
                    "last_modified": "2026-01-12T10:30:00Z",
                    "checksum": "sha256:abc123...",
                    "metadata": {"lines": 450, "functions": 23}
                },
                {
                    "path": "config/model_weights.json",
                    "type": "config",
                    "size_bytes": 512,
                    "last_modified": "2026-01-11T23:07:00Z"
                }
            ]
        """
        pass
    
    @abstractmethod
    def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """
        Apply an optimization suggestion
        
        Args:
            optimization: Dictionary containing:
                - type: Optimization type (e.g., 'model_weight_adjustment')
                - parameters: Parameters for the optimization
                - expected_improvement: Expected improvement description
                - confidence: Confidence score (0.0-1.0)
        
        Returns:
            True if optimization was successfully applied, False otherwise
        
        Example optimization:
            {
                "type": "model_weight_adjustment",
                "parameters": {
                    "lstm": 0.32,
                    "transformer": 0.24,
                    "automl": 0.26,
                    "ensemble": 0.18
                },
                "expected_improvement": "5-10% accuracy improvement",
                "confidence": 0.85
            }
        """
        pass
    
    # Optional methods (can be overridden for enhanced functionality)
    
    def get_dependencies(self) -> Dict[str, List[str]]:
        """
        Return agent dependencies
        
        Returns:
            Dictionary with 'internal' and 'external' dependency lists
        """
        return {
            'internal': [],
            'external': []
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Return current agent configuration
        
        Returns:
            Dictionary of configuration parameters
        """
        return {}
    
    def validate_state(self) -> Dict[str, Any]:
        """
        Validate agent state and return health check
        
        Returns:
            Dictionary with validation results:
            - healthy: Boolean indicating if agent is healthy
            - issues: List of any issues found
            - warnings: List of warnings
        """
        return {
            'healthy': True,
            'issues': [],
            'warnings': []
        }
    
    def get_learning_history(self) -> List[Dict[str, Any]]:
        """
        Return agent's learning history
        
        Returns:
            List of learning events with timestamps and insights
        """
        return []
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export complete agent state for backup/migration
        
        Returns:
            Dictionary containing full agent state
        """
        return {
            'metadata': self.get_metadata(),
            'metrics': self.get_performance_metrics(),
            'configuration': self.get_configuration(),
            'assets': self.get_assets()
        }
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        Restore agent state from exported data
        
        Args:
            state: State dictionary from export_state()
        
        Returns:
            True if restoration was successful
        """
        return False


class BaseAgent(AgentInterface):
    """
    Base implementation with sensible defaults
    
    Agents can inherit from this instead of AgentInterface for convenience.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", agent_type: str = "generic"):
        self.name = name
        self.version = version
        self.agent_type = agent_type
    
    def get_metadata(self) -> Dict[str, Any]:
        """Default metadata implementation"""
        return {
            'name': self.name,
            'version': self.version,
            'type': self.agent_type,
            'repository': None,
            'capabilities': [],
            'description': f"{self.name} agent",
            'author': "Manus AI"
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Default metrics implementation"""
        return {
            'accuracy': None,
            'latency_ms': None,
            'error_rate': 0.0,
            'custom_metrics': {}
        }
    
    def get_assets(self) -> List[Dict[str, Any]]:
        """Default assets implementation"""
        return []
    
    def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Default optimization implementation (no-op)"""
        return False


if __name__ == '__main__':
    print("Agent Interface v1.0.0")
    print("This module defines the interface that all agents must implement.")
