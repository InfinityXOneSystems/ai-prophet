#!/usr/bin/env python3
"""
Evolution Module
Implements continuous learning and optimization for AI agents.

Author: Manus AI
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class EvolutionModule:
    """
    Main evolution module that analyzes performance and generates optimizations
    """
    
    def __init__(self, agent, config: Optional[Dict] = None):
        """
        Initialize evolution module
        
        Args:
            agent: Agent instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.config = config or {}
        self.root_dir = Path(self.config.get('root_dir', '.'))
        
        self.auto_optimize = self.config.get('auto_optimize', False)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.performance_threshold = self.config.get('performance_threshold', 0.7)
        self.track_learning = self.config.get('track_learning', True)
        
        self.learning_history_path = self.root_dir / 'data' / 'learning' / 'evolution_history.json'
        self.learning_history_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.results = None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete evolution analysis
        
        Returns:
            Dictionary with evolution results
        """
        print("   🧬 Analyzing performance evolution...")
        
        # Get current performance metrics
        current_metrics = self.agent.get_performance_metrics()
        
        # Load historical performance
        historical_metrics = self._load_historical_metrics()
        
        # Analyze performance trends
        print("      Analyzing trends...")
        trends = self._analyze_trends(current_metrics, historical_metrics)
        
        # Identify improvement opportunities
        print("      Identifying opportunities...")
        opportunities = self._identify_opportunities(current_metrics, trends)
        
        # Generate optimizations
        print("      Generating optimizations...")
        optimizations = self._generate_optimizations(opportunities)
        
        # Calculate evolution score
        evolution_score = self._calculate_evolution_score(trends)
        print(f"      Evolution score: {evolution_score:.2f}/100")
        
        # Predict future performance
        predictions = self._predict_future_performance(historical_metrics, current_metrics)
        
        # Save current metrics to history
        self._save_to_history(current_metrics)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'trends': trends,
            'opportunities': opportunities,
            'optimizations': optimizations,
            'evolution_score': evolution_score,
            'predictions': predictions
        }
        
        return self.results
    
    def _load_historical_metrics(self) -> List[Dict[str, Any]]:
        """Load historical performance metrics"""
        if not self.learning_history_path.exists():
            return []
        
        try:
            with open(self.learning_history_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _save_to_history(self, metrics: Dict[str, Any]):
        """Save current metrics to learning history"""
        history = self._load_historical_metrics()
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        # Keep only last 100 entries
        history = history[-100:]
        
        with open(self.learning_history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _analyze_trends(self, current: Dict[str, Any], historical: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        trends = {
            'improving': [],
            'declining': [],
            'stable': [],
            'insufficient_data': []
        }
        
        if len(historical) < 2:
            return {
                **trends,
                'status': 'insufficient_data',
                'message': 'Need at least 2 historical data points to analyze trends'
            }
        
        # Extract metrics over time
        metric_series = defaultdict(list)
        for entry in historical[-10:]:  # Last 10 entries
            for key, value in entry.get('metrics', {}).items():
                if isinstance(value, (int, float)) and value is not None:
                    metric_series[key].append(value)
        
        # Analyze each metric
        for metric_name, values in metric_series.items():
            if len(values) < 2:
                trends['insufficient_data'].append(metric_name)
                continue
            
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(values))
            y = np.array(values)
            
            if len(x) > 1 and np.std(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                
                # Determine trend direction
                if abs(slope) < 0.01:  # Stable
                    trends['stable'].append({
                        'metric': metric_name,
                        'current_value': values[-1],
                        'change': 0.0
                    })
                elif slope > 0:  # Improving
                    trends['improving'].append({
                        'metric': metric_name,
                        'current_value': values[-1],
                        'change': slope,
                        'improvement_rate': (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                    })
                else:  # Declining
                    trends['declining'].append({
                        'metric': metric_name,
                        'current_value': values[-1],
                        'change': slope,
                        'decline_rate': (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                    })
        
        return trends
    
    def _identify_opportunities(self, metrics: Dict[str, Any], trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for improvement"""
        opportunities = []
        
        # Check for low accuracy
        accuracy = metrics.get('accuracy')
        if accuracy is not None and accuracy < self.performance_threshold:
            opportunities.append({
                'type': 'low_accuracy',
                'severity': 'high',
                'current_value': accuracy,
                'target_value': self.performance_threshold,
                'description': f"Accuracy ({accuracy:.2%}) is below threshold ({self.performance_threshold:.2%})",
                'potential_actions': [
                    'Adjust model weights',
                    'Increase training data',
                    'Feature engineering',
                    'Hyperparameter tuning'
                ]
            })
        
        # Check for declining trends
        for declining_metric in trends.get('declining', []):
            opportunities.append({
                'type': 'declining_metric',
                'severity': 'medium',
                'metric': declining_metric['metric'],
                'current_value': declining_metric['current_value'],
                'decline_rate': declining_metric.get('decline_rate', 0),
                'description': f"{declining_metric['metric']} is declining",
                'potential_actions': [
                    'Investigate root cause',
                    'Review recent changes',
                    'Adjust learning parameters'
                ]
            })
        
        # Check for high error rate
        error_rate = metrics.get('error_rate')
        if error_rate is not None and error_rate > 0.05:
            opportunities.append({
                'type': 'high_error_rate',
                'severity': 'high',
                'current_value': error_rate,
                'target_value': 0.02,
                'description': f"Error rate ({error_rate:.2%}) is above acceptable threshold",
                'potential_actions': [
                    'Implement error handling',
                    'Add validation checks',
                    'Review error logs'
                ]
            })
        
        # Check for high latency
        latency = metrics.get('latency_ms')
        if latency is not None and latency > 1000:
            opportunities.append({
                'type': 'high_latency',
                'severity': 'medium',
                'current_value': latency,
                'target_value': 500,
                'description': f"Latency ({latency}ms) is higher than optimal",
                'potential_actions': [
                    'Optimize algorithms',
                    'Add caching',
                    'Parallel processing',
                    'Database query optimization'
                ]
            })
        
        return opportunities
    
    def _generate_optimizations(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations"""
        optimizations = []
        
        for opportunity in opportunities:
            if opportunity['type'] == 'low_accuracy':
                # Generate model weight adjustment
                optimizations.append({
                    'id': f"opt_{datetime.now().strftime('%Y%m%d%H%M%S')}_001",
                    'type': 'model_weight_adjustment',
                    'priority': 'high',
                    'opportunity': opportunity['type'],
                    'description': 'Adjust model weights to improve accuracy',
                    'parameters': self._suggest_weight_adjustments(),
                    'expected_improvement': '5-10% accuracy increase',
                    'confidence': 0.75,
                    'requires_approval': True
                })
            
            elif opportunity['type'] == 'declining_metric':
                optimizations.append({
                    'id': f"opt_{datetime.now().strftime('%Y%m%d%H%M%S')}_002",
                    'type': 'parameter_tuning',
                    'priority': 'medium',
                    'opportunity': opportunity['type'],
                    'description': f"Tune parameters to stabilize {opportunity['metric']}",
                    'parameters': {
                        'learning_rate': self.learning_rate * 0.9,
                        'metric': opportunity['metric']
                    },
                    'expected_improvement': 'Stabilize declining trend',
                    'confidence': 0.65,
                    'requires_approval': False
                })
            
            elif opportunity['type'] == 'high_error_rate':
                optimizations.append({
                    'id': f"opt_{datetime.now().strftime('%Y%m%d%H%M%S')}_003",
                    'type': 'error_handling_enhancement',
                    'priority': 'high',
                    'opportunity': opportunity['type'],
                    'description': 'Enhance error handling and validation',
                    'parameters': {
                        'add_retry_logic': True,
                        'add_validation': True,
                        'log_errors': True
                    },
                    'expected_improvement': 'Reduce error rate by 50%',
                    'confidence': 0.80,
                    'requires_approval': False
                })
            
            elif opportunity['type'] == 'high_latency':
                optimizations.append({
                    'id': f"opt_{datetime.now().strftime('%Y%m%d%H%M%S')}_004",
                    'type': 'performance_optimization',
                    'priority': 'medium',
                    'opportunity': opportunity['type'],
                    'description': 'Optimize performance to reduce latency',
                    'parameters': {
                        'enable_caching': True,
                        'optimize_queries': True,
                        'parallel_processing': True
                    },
                    'expected_improvement': 'Reduce latency by 30-40%',
                    'confidence': 0.70,
                    'requires_approval': False
                })
        
        return optimizations
    
    def _suggest_weight_adjustments(self) -> Dict[str, float]:
        """Suggest model weight adjustments based on performance"""
        # This is a placeholder - actual implementation would analyze
        # which models are performing best and adjust accordingly
        return {
            'model_a': 0.30,
            'model_b': 0.25,
            'model_c': 0.25,
            'model_d': 0.20
        }
    
    def _calculate_evolution_score(self, trends: Dict[str, Any]) -> float:
        """Calculate overall evolution score (0-100)"""
        if trends.get('status') == 'insufficient_data':
            return 50.0  # Neutral score
        
        improving_count = len(trends.get('improving', []))
        declining_count = len(trends.get('declining', []))
        stable_count = len(trends.get('stable', []))
        
        total = improving_count + declining_count + stable_count
        
        if total == 0:
            return 50.0
        
        # Calculate weighted score
        score = (
            (improving_count * 100) +
            (stable_count * 70) +
            (declining_count * 30)
        ) / total
        
        return score
    
    def _predict_future_performance(self, historical: List[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future performance based on trends"""
        predictions = {
            'timeline': '7_days',
            'confidence': 0.65,
            'scenarios': []
        }
        
        if len(historical) < 3:
            predictions['scenarios'].append({
                'name': 'insufficient_data',
                'description': 'Need more historical data for accurate predictions',
                'probability': 1.0
            })
            return predictions
        
        # Optimistic scenario
        predictions['scenarios'].append({
            'name': 'optimistic',
            'description': 'All optimizations applied successfully, trends improve',
            'probability': 0.25,
            'expected_metrics': {
                'accuracy': current.get('accuracy', 0.7) * 1.10,
                'error_rate': current.get('error_rate', 0.05) * 0.5
            }
        })
        
        # Realistic scenario
        predictions['scenarios'].append({
            'name': 'realistic',
            'description': 'Some optimizations applied, gradual improvement',
            'probability': 0.50,
            'expected_metrics': {
                'accuracy': current.get('accuracy', 0.7) * 1.05,
                'error_rate': current.get('error_rate', 0.05) * 0.8
            }
        })
        
        # Pessimistic scenario
        predictions['scenarios'].append({
            'name': 'pessimistic',
            'description': 'No optimizations applied, trends continue',
            'probability': 0.25,
            'expected_metrics': {
                'accuracy': current.get('accuracy', 0.7) * 0.98,
                'error_rate': current.get('error_rate', 0.05) * 1.2
            }
        })
        
        return predictions
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning history"""
        history = self._load_historical_metrics()
        
        if not history:
            return {'status': 'no_data'}
        
        return {
            'total_entries': len(history),
            'date_range': {
                'start': history[0]['timestamp'],
                'end': history[-1]['timestamp']
            },
            'metrics_tracked': list(history[-1].get('metrics', {}).keys())
        }


__all__ = ['EvolutionModule']
