#!/usr/bin/env python3
"""
AI Prophet Model Weight Update System
Updates model weights based on daily performance analysis and learning insights.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ModelWeightOptimizer:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.today = datetime.now().strftime('%Y%m%d')
        self.config_file = Path('config') / 'model_weights.json'
        
        # Default model weights
        self.default_weights = {
            'lstm': 0.30,
            'transformer': 0.25,
            'automl': 0.25,
            'ensemble': 0.20
        }
        
        # Load current weights
        self.current_weights = self.load_current_weights()
        
    def load_current_weights(self) -> Dict[str, float]:
        """Load current model weights from config"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('model_weights', self.default_weights)
            except Exception as e:
                print(f"⚠️  Error loading weights: {e}")
                return self.default_weights
        else:
            return self.default_weights
    
    def load_learning_report(self) -> Dict[str, Any]:
        """Load the latest learning report"""
        report_file = self.data_dir / 'learning' / f'daily_reflection_{self.today}.json'
        
        if not report_file.exists():
            print(f"⚠️  No learning report found for today")
            return None
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def calculate_weight_adjustments(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal weight adjustments based on performance"""
        
        new_weights = self.current_weights.copy()
        adjustments_made = []
        
        # Get key metrics from report
        overall_metrics = report.get('overall_metrics', {})
        pred_analysis = report.get('prediction_analysis', {})
        trade_analysis = report.get('trade_analysis', {})
        
        avg_confidence = pred_analysis.get('average_confidence', 0)
        confidence_std = pred_analysis.get('confidence_std', 0)
        closed_trades = trade_analysis.get('closed_trades', {}).get('count', 0)
        win_rate = trade_analysis.get('closed_trades', {}).get('win_rate', 0)
        
        print(f"\n📊 Analyzing Performance Metrics:")
        print(f"   Average Confidence: {avg_confidence:.2%}")
        print(f"   Confidence Std Dev: {confidence_std:.4f}")
        print(f"   Closed Trades: {closed_trades}")
        print(f"   Win Rate: {win_rate:.2%}")
        
        # Adjustment Strategy 1: Confidence-based tuning
        if avg_confidence > 0.75:
            # High confidence - increase ensemble weight slightly
            adjustment = 0.02
            new_weights['ensemble'] = min(0.30, new_weights['ensemble'] + adjustment)
            new_weights['lstm'] = max(0.20, new_weights['lstm'] - adjustment/2)
            new_weights['transformer'] = max(0.20, new_weights['transformer'] - adjustment/2)
            adjustments_made.append(f"High confidence detected: Increased ensemble weight by {adjustment:.2%}")
        
        elif avg_confidence < 0.65:
            # Low confidence - increase AutoML weight (more adaptive)
            adjustment = 0.03
            new_weights['automl'] = min(0.35, new_weights['automl'] + adjustment)
            new_weights['ensemble'] = max(0.15, new_weights['ensemble'] - adjustment)
            adjustments_made.append(f"Low confidence detected: Increased AutoML weight by {adjustment:.2%}")
        
        # Adjustment Strategy 2: Variance-based tuning
        if confidence_std > 0.15:
            # High variance - favor more stable models (LSTM)
            adjustment = 0.02
            new_weights['lstm'] = min(0.35, new_weights['lstm'] + adjustment)
            new_weights['transformer'] = max(0.20, new_weights['transformer'] - adjustment)
            adjustments_made.append(f"High variance detected: Increased LSTM weight by {adjustment:.2%}")
        
        # Adjustment Strategy 3: Win rate based (only if we have closed trades)
        if closed_trades >= 10:
            if win_rate > 0.60:
                # Good performance - maintain current strategy with slight boost to best performer
                adjustment = 0.01
                # Find the model with highest weight and boost it
                max_model = max(new_weights, key=new_weights.get)
                new_weights[max_model] = min(0.40, new_weights[max_model] + adjustment)
                adjustments_made.append(f"Strong win rate: Boosted {max_model} by {adjustment:.2%}")
            
            elif win_rate < 0.45:
                # Poor performance - rebalance toward default weights
                for model in new_weights:
                    new_weights[model] = (new_weights[model] + self.default_weights[model]) / 2
                adjustments_made.append(f"Low win rate: Rebalancing toward default weights")
        
        # Normalize weights to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Round to 2 decimal places
        new_weights = {k: round(v, 2) for k, v in new_weights.items()}
        
        return new_weights, adjustments_made
    
    def save_updated_weights(self, new_weights: Dict[str, float], adjustments: list):
        """Save updated model weights to config"""
        
        # Create config directory if it doesn't exist
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        # Prepare config data
        config = {
            'model_weights': new_weights,
            'last_updated': datetime.now().isoformat(),
            'previous_weights': self.current_weights,
            'adjustments_made': adjustments,
            'update_reason': 'Daily self-reflection optimization'
        }
        
        # Save to config file
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Updated weights saved to: {self.config_file}")
        
        # Also save to learning directory for historical tracking
        learning_dir = self.data_dir / 'learning'
        history_file = learning_dir / f'weight_update_{self.today}.json'
        
        with open(history_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📊 Weight history saved to: {history_file}")
        
        return config
    
    def generate_weight_comparison_report(self, new_weights: Dict[str, float], adjustments: list) -> str:
        """Generate a markdown report comparing old and new weights"""
        
        report = []
        report.append(f"# Model Weight Update Report")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"")
        report.append(f"---")
        report.append(f"")
        
        # Weight comparison table
        report.append(f"## ⚖️ Weight Adjustments")
        report.append(f"")
        report.append(f"| Model | Previous | New | Change |")
        report.append(f"|-------|----------|-----|--------|")
        
        for model in sorted(self.current_weights.keys()):
            old_weight = self.current_weights[model]
            new_weight = new_weights[model]
            change = new_weight - old_weight
            change_str = f"+{change:.2%}" if change > 0 else f"{change:.2%}"
            
            report.append(f"| {model} | {old_weight:.2%} | {new_weight:.2%} | {change_str} |")
        
        report.append(f"")
        
        # Adjustments made
        report.append(f"## 📝 Adjustments Made")
        report.append(f"")
        if adjustments:
            for i, adj in enumerate(adjustments, 1):
                report.append(f"{i}. {adj}")
        else:
            report.append(f"*No adjustments needed - weights remain optimal*")
        report.append(f"")
        
        # Rationale
        report.append(f"## 🎯 Optimization Rationale")
        report.append(f"")
        report.append(f"Model weights are continuously optimized based on:")
        report.append(f"")
        report.append(f"1. **Prediction Confidence**: Higher confidence predictions favor ensemble methods")
        report.append(f"2. **Confidence Variance**: High variance favors stable models like LSTM")
        report.append(f"3. **Win Rate Performance**: Successful strategies get reinforced")
        report.append(f"4. **Adaptive Learning**: AutoML weight increases during uncertain conditions")
        report.append(f"")
        report.append(f"These adjustments ensure AI Prophet continuously evolves and improves accuracy.")
        report.append(f"")
        
        report.append(f"---")
        report.append(f"")
        report.append(f"*Generated by AI Prophet Model Weight Optimizer*")
        
        markdown_content = "\n".join(report)
        
        # Save markdown report
        learning_dir = self.data_dir / 'learning'
        markdown_file = learning_dir / f'weight_update_{self.today}.md'
        
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Weight update report saved to: {markdown_file}")
        
        return markdown_content
    
    def run_optimization(self):
        """Run the complete model weight optimization"""
        
        print("=" * 60)
        print("AI PROPHET MODEL WEIGHT OPTIMIZATION")
        print("Updating model weights based on performance insights")
        print("=" * 60)
        
        # Load learning report
        report = self.load_learning_report()
        
        if not report:
            print("\n⚠️  No learning report available. Cannot optimize weights.")
            return None
        
        print(f"\n📖 Loaded learning report from {self.today}")
        
        # Display current weights
        print(f"\n⚖️  Current Model Weights:")
        for model, weight in self.current_weights.items():
            print(f"   {model}: {weight:.2%}")
        
        # Calculate new weights
        new_weights, adjustments = self.calculate_weight_adjustments(report)
        
        # Display new weights
        print(f"\n⚖️  Optimized Model Weights:")
        for model, weight in new_weights.items():
            change = weight - self.current_weights[model]
            change_str = f"(+{change:.2%})" if change > 0 else f"({change:.2%})" if change < 0 else "(no change)"
            print(f"   {model}: {weight:.2%} {change_str}")
        
        # Display adjustments
        if adjustments:
            print(f"\n📝 Adjustments Made:")
            for i, adj in enumerate(adjustments, 1):
                print(f"   {i}. {adj}")
        else:
            print(f"\n✅ No adjustments needed - current weights are optimal")
        
        # Save updated weights
        config = self.save_updated_weights(new_weights, adjustments)
        
        # Generate comparison report
        markdown_content = self.generate_weight_comparison_report(new_weights, adjustments)
        
        print("\n" + "=" * 60)
        print("✅ Model weight optimization complete!")
        print("=" * 60)
        
        return config

def main():
    optimizer = ModelWeightOptimizer()
    result = optimizer.run_optimization()
    
    if result:
        print(f"\n🎯 Model weights have been optimized and saved")
        return 0
    else:
        print(f"\n⚠️  Optimization incomplete")
        return 1

if __name__ == '__main__':
    exit(main())
