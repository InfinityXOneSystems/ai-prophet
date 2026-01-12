#!/usr/bin/env python3
"""
Universal Self-Reflection Engine
Core orchestration engine for agent self-reflection and continuous improvement.

Author: Manus AI
Version: 1.0.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import importlib

@dataclass
class ReflectionResult:
    """Results from a reflection cycle"""
    timestamp: str
    agent_name: str
    modules_executed: List[str]
    inventory: Optional[Dict] = None
    taxonomy: Optional[Dict] = None
    documentation: Optional[Dict] = None
    indexing: Optional[Dict] = None
    evolution: Optional[Dict] = None
    reports: Optional[Dict] = None
    errors: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: Path):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ReflectionEngine:
    """
    Universal Self-Reflection Engine
    
    Orchestrates the self-reflection process for any AI agent by:
    1. Loading agent configuration
    2. Executing enabled reflection modules
    3. Collecting and aggregating results
    4. Generating reports and insights
    5. Applying optimizations (with approval if needed)
    """
    
    def __init__(self, agent, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the reflection engine
        
        Args:
            agent: Agent instance implementing AgentInterface
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        self.agent = agent
        self.config = self._load_config(config_path, config_dict)
        self.logger = self._setup_logging()
        self.modules = {}
        self.results = None
        
        self.logger.info(f"Reflection Engine initialized for agent: {self.config['agent']['name']}")
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict:
        """Load configuration from file or dictionary"""
        if config_dict:
            return config_dict
        
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'agent': {
                'name': 'Unknown Agent',
                'type': 'generic',
                'repository': None
            },
            'reflection': {
                'schedule': 'daily',
                'modules': ['inventory', 'documentation', 'reporting']
            },
            'reporting': {
                'formats': ['json', 'markdown'],
                'save_to': 'data/reflection/'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the reflection engine"""
        logger = logging.getLogger('ReflectionEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_modules(self):
        """Dynamically load enabled reflection modules"""
        enabled_modules = self.config['reflection']['modules']
        
        for module_name in enabled_modules:
            try:
                # Import module dynamically
                module_path = f"modules.{module_name}"
                module = importlib.import_module(module_path)
                
                # Get the main class (e.g., InventoryModule)
                class_name = f"{module_name.capitalize()}Module"
                module_class = getattr(module, class_name)
                
                # Instantiate the module
                self.modules[module_name] = module_class(
                    agent=self.agent,
                    config=self.config.get(module_name, {})
                )
                
                self.logger.info(f"Loaded module: {module_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load module {module_name}: {e}")
    
    def run_daily_reflection(self) -> ReflectionResult:
        """
        Run a complete daily reflection cycle
        
        Returns:
            ReflectionResult object with all findings
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING DAILY SELF-REFLECTION CYCLE")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        errors = []
        
        # Initialize result object
        result = ReflectionResult(
            timestamp=start_time.isoformat(),
            agent_name=self.config['agent']['name'],
            modules_executed=[],
            errors=[]
        )
        
        # Phase 1: Data Collection
        self.logger.info("\n📊 Phase 1: Data Collection")
        try:
            agent_metadata = self.agent.get_metadata()
            agent_metrics = self.agent.get_performance_metrics()
            agent_assets = self.agent.get_assets()
            
            self.logger.info(f"   Agent: {agent_metadata.get('name', 'Unknown')}")
            self.logger.info(f"   Version: {agent_metadata.get('version', 'Unknown')}")
            self.logger.info(f"   Assets: {len(agent_assets)}")
        except Exception as e:
            error_msg = f"Data collection failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
        
        # Phase 2: Module Execution
        self.logger.info("\n🔧 Phase 2: Module Execution")
        
        enabled_modules = self.config['reflection']['modules']
        
        for module_name in enabled_modules:
            try:
                self.logger.info(f"\n   Executing: {module_name}")
                
                # Get module instance
                if module_name not in self.modules:
                    # Try to load it if not already loaded
                    self._load_modules()
                
                if module_name in self.modules:
                    module = self.modules[module_name]
                    module_result = module.execute()
                    
                    # Store result
                    setattr(result, module_name, module_result)
                    result.modules_executed.append(module_name)
                    
                    self.logger.info(f"   ✅ {module_name} completed")
                else:
                    error_msg = f"Module {module_name} not available"
                    self.logger.warning(f"   ⚠️  {error_msg}")
                    errors.append(error_msg)
                    
            except Exception as e:
                error_msg = f"Module {module_name} failed: {e}"
                self.logger.error(f"   ❌ {error_msg}")
                errors.append(error_msg)
        
        # Phase 3: Report Generation
        self.logger.info("\n📝 Phase 3: Report Generation")
        try:
            if 'reporting' in self.modules:
                reporting_module = self.modules['reporting']
                reports = reporting_module.generate_reports(result)
                result.reports = reports
                self.logger.info("   ✅ Reports generated")
        except Exception as e:
            error_msg = f"Report generation failed: {e}"
            self.logger.error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        
        # Store errors
        result.errors = errors
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"✅ REFLECTION CYCLE COMPLETE")
        self.logger.info(f"   Duration: {duration:.2f} seconds")
        self.logger.info(f"   Modules executed: {len(result.modules_executed)}")
        self.logger.info(f"   Errors: {len(errors)}")
        self.logger.info("=" * 60)
        
        # Save results
        self.results = result
        self._save_results(result)
        
        return result
    
    def _save_results(self, result: ReflectionResult):
        """Save reflection results to disk"""
        save_dir = Path(self.config['reporting']['save_to'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = save_dir / f"reflection_{timestamp}.json"
        result.to_json(json_file)
        self.logger.info(f"\n💾 Results saved to: {json_file}")
    
    def run_on_event(self, event_type: str, event_data: Dict) -> ReflectionResult:
        """
        Run reflection triggered by a specific event
        
        Args:
            event_type: Type of event (e.g., 'trade_completed', 'error_occurred')
            event_data: Data associated with the event
        """
        self.logger.info(f"Running event-triggered reflection: {event_type}")
        
        # Filter modules based on event type
        # For now, run all modules, but could be optimized
        return self.run_daily_reflection()
    
    def get_latest_results(self) -> Optional[ReflectionResult]:
        """Get the most recent reflection results"""
        return self.results
    
    def apply_optimizations(self, optimizations: List[Dict], require_approval: bool = True) -> Dict:
        """
        Apply optimization suggestions to the agent
        
        Args:
            optimizations: List of optimization dictionaries
            require_approval: Whether to require human approval
            
        Returns:
            Dictionary with application results
        """
        results = {
            'applied': [],
            'skipped': [],
            'failed': []
        }
        
        for optimization in optimizations:
            try:
                if require_approval and optimization.get('requires_approval', True):
                    self.logger.info(f"Skipping optimization (requires approval): {optimization.get('type')}")
                    results['skipped'].append(optimization)
                    continue
                
                # Apply optimization through agent interface
                success = self.agent.apply_optimization(optimization)
                
                if success:
                    self.logger.info(f"Applied optimization: {optimization.get('type')}")
                    results['applied'].append(optimization)
                else:
                    self.logger.warning(f"Failed to apply optimization: {optimization.get('type')}")
                    results['failed'].append(optimization)
                    
            except Exception as e:
                self.logger.error(f"Error applying optimization: {e}")
                results['failed'].append(optimization)
        
        return results


class ReflectionScheduler:
    """
    Scheduler for automated reflection cycles
    """
    
    def __init__(self):
        self.scheduled_agents = {}
        self.logger = logging.getLogger('ReflectionScheduler')
    
    def schedule_daily(self, agent_name: str, time: str, modules: List[str]):
        """
        Schedule daily reflection for an agent
        
        Args:
            agent_name: Name of the agent
            time: Time to run (HH:MM format)
            modules: List of modules to execute
        """
        self.logger.info(f"Scheduling daily reflection for {agent_name} at {time}")
        
        self.scheduled_agents[agent_name] = {
            'schedule': 'daily',
            'time': time,
            'modules': modules
        }
    
    def schedule_hourly(self, agent_name: str, modules: List[str]):
        """Schedule hourly reflection for an agent"""
        self.logger.info(f"Scheduling hourly reflection for {agent_name}")
        
        self.scheduled_agents[agent_name] = {
            'schedule': 'hourly',
            'modules': modules
        }
    
    def start(self):
        """Start the scheduler (would integrate with cron/systemd in production)"""
        self.logger.info("Scheduler started")
        # In production, this would set up actual scheduled tasks
        pass
    
    def stop(self):
        """Stop the scheduler"""
        self.logger.info("Scheduler stopped")


if __name__ == '__main__':
    # Example usage
    print("Universal Self-Reflection Engine v1.0.0")
    print("This module should be imported and used with an agent implementation.")
