"""
Universal Self-Reflection System - Core Module

Author: Manus AI
Version: 1.0.0
"""

from .agent_interface import AgentInterface, BaseAgent, AgentMetadata, PerformanceMetrics, AgentAsset
from .reflection_engine import ReflectionEngine, ReflectionScheduler, ReflectionResult
from .config_manager import ConfigManager, ReflectionConfig

__all__ = [
    'AgentInterface',
    'BaseAgent',
    'AgentMetadata',
    'PerformanceMetrics',
    'AgentAsset',
    'ReflectionEngine',
    'ReflectionScheduler',
    'ReflectionResult',
    'ConfigManager',
    'ReflectionConfig'
]

__version__ = '1.0.0'
