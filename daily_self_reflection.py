#!/usr/bin/env python3
"""
AI Prophet Daily Self-Reflection and Learning Optimization
Analyzes predictions, trades, and performance to continuously improve accuracy.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import statistics

class DailySelfReflection:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.today = datetime.now().strftime('%Y%m%d')
        self.analysis_results = {
            'date': datetime.now().isoformat(),
            'overall_metrics': {},
            'prediction_analysis': {},
            'trade_analysis': {},
            'model_performance': {},
            'asset_class_performance': {},
            'strategy_insights': {},
            'model_weight_adjustments': {},
            'recommendations': []
        }
        
    def load_all_cycles(self):
        """Load all trading cycles from today"""
        cycles = []
        cycle_dir = self.data_dir / 'day_trading_cycles'
        
        if not cycle_dir.exists():
            print(f"⚠️  No cycle directory found at {cycle_dir}")
            return cycles
            
        for cycle_file in sorted(cycle_dir.glob(f'*_{self.today}_*.json')):
            try:
                with open(cycle_file, 'r') as f:
                    cycle_data = json.load(f)
                    cycles.append({
                        'file': cycle_file.name,
                        'data': cycle_data
                    })
            except Exception as e:
                print(f"⚠️  Error loading {cycle_file}: {e}")
                
        print(f"📊 Loaded {len(cycles)} trading cycles from today")
        return cycles
    
    def load_day_trading_state(self):
        """Load current day trading state"""
        state_file = self.data_dir / 'day_trading' / f'state_{self.today}.json'
        
        if not state_file.exists():
            print(f"⚠️  No day trading state found for today")
            return None
            
        with open(state_file, 'r') as f:
            return json.load(f)
    
    def analyze_predictions(self, cycles):
        """Analyze all predictions made today"""
        all_predictions = []
        
        for cycle in cycles:
            predictions = cycle['data'].get('stages', {}).get('predictions', {}).get('predictions', [])
            for pred in predictions:
                pred['cycle'] = cycle['file']
                all_predictions.append(pred)
        
        if not all_predictions:
            print("⚠️  No predictions found in cycles")
            return
        
        # Analyze prediction patterns
        directions = defaultdict(int)
        confidences = []
        symbols = defaultdict(int)
        asset_classes = defaultdict(int)
        
        for pred in all_predictions:
            directions[pred.get('direction', 'UNKNOWN')] += 1
            confidences.append(pred.get('confidence', 0))
            symbols[pred.get('symbol', 'UNKNOWN')] += 1
            asset_classes[pred.get('asset_class', 'UNKNOWN')] += 1
        
        self.analysis_results['prediction_analysis'] = {
            'total_predictions': len(all_predictions),
            'direction_distribution': dict(directions),
            'average_confidence': statistics.mean(confidences) if confidences else 0,
            'median_confidence': statistics.median(confidences) if confidences else 0,
            'confidence_std': statistics.stdev(confidences) if len(confidences) > 1 else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'symbols_predicted': dict(symbols),
            'asset_class_distribution': dict(asset_classes)
        }
        
        print(f"\n📈 PREDICTION ANALYSIS")
        print(f"   Total Predictions: {len(all_predictions)}")
        print(f"   Average Confidence: {statistics.mean(confidences):.2%}")
        print(f"   Direction Distribution: {dict(directions)}")
        
        return all_predictions
    
    def analyze_trades(self, state):
        """Analyze open and closed trades"""
        if not state:
            print("⚠️  No state data available for trade analysis")
            return
        
        open_positions = state.get('open_positions', {})
        closed_trades = state.get('closed_trades', [])
        
        # Analyze open positions
        open_analysis = {
            'count': len(open_positions),
            'by_asset_class': defaultdict(int),
            'by_side': defaultdict(int),
            'by_strategy': defaultdict(int),
            'total_exposure': 0
        }
        
        for trade_id, position in open_positions.items():
            asset_class = position.get('asset_class', 'unknown')
            side = position.get('side', 'unknown')
            strategy = position.get('strategy', 'unknown')
            
            open_analysis['by_asset_class'][asset_class] += 1
            open_analysis['by_side'][side] += 1
            open_analysis['by_strategy'][strategy] += 1
            
            # Calculate exposure (entry_price * quantity)
            exposure = position.get('entry_price', 0) * position.get('quantity', 0)
            open_analysis['total_exposure'] += exposure
        
        # Analyze closed trades
        closed_analysis = {
            'count': len(closed_trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'by_asset_class': defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0}),
            'by_strategy': defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
        }
        
        for trade in closed_trades:
            pnl = trade.get('pnl', 0)
            asset_class = trade.get('asset_class', 'unknown')
            strategy = trade.get('strategy', 'unknown')
            
            closed_analysis['total_pnl'] += pnl
            
            if pnl > 0:
                closed_analysis['winning_trades'] += 1
                closed_analysis['by_asset_class'][asset_class]['wins'] += 1
                closed_analysis['by_strategy'][strategy]['wins'] += 1
            elif pnl < 0:
                closed_analysis['losing_trades'] += 1
            
            closed_analysis['by_asset_class'][asset_class]['count'] += 1
            closed_analysis['by_asset_class'][asset_class]['pnl'] += pnl
            closed_analysis['by_strategy'][strategy]['count'] += 1
            closed_analysis['by_strategy'][strategy]['pnl'] += pnl
        
        # Calculate win rate
        if closed_analysis['count'] > 0:
            closed_analysis['win_rate'] = closed_analysis['winning_trades'] / closed_analysis['count']
        else:
            closed_analysis['win_rate'] = 0
        
        self.analysis_results['trade_analysis'] = {
            'open_positions': {k: v for k, v in open_analysis.items() if k != 'by_asset_class' and k != 'by_side' and k != 'by_strategy'},
            'open_by_asset_class': dict(open_analysis['by_asset_class']),
            'open_by_side': dict(open_analysis['by_side']),
            'open_by_strategy': dict(open_analysis['by_strategy']),
            'closed_trades': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in closed_analysis.items() if k != 'by_asset_class' and k != 'by_strategy'},
            'closed_by_asset_class': {k: dict(v) for k, v in closed_analysis['by_asset_class'].items()},
            'closed_by_strategy': {k: dict(v) for k, v in closed_analysis['by_strategy'].items()}
        }
        
        print(f"\n💼 TRADE ANALYSIS")
        print(f"   Open Positions: {open_analysis['count']}")
        print(f"   Closed Trades: {closed_analysis['count']}")
        if closed_analysis['count'] > 0:
            print(f"   Win Rate: {closed_analysis['win_rate']:.2%}")
            print(f"   Total P&L: ${closed_analysis['total_pnl']:.2f}")
    
    def calculate_model_weights(self, predictions, state):
        """Calculate optimal model weight adjustments based on performance"""
        
        # Default model weights (baseline)
        current_weights = {
            'lstm': 0.30,
            'transformer': 0.25,
            'automl': 0.25,
            'ensemble': 0.20
        }
        
        # Since we don't have completed trades yet, we'll analyze prediction quality
        # based on confidence levels and distribution
        
        adjustments = {}
        
        if predictions:
            avg_confidence = statistics.mean([p.get('confidence', 0) for p in predictions])
            
            # If average confidence is high, we can be more aggressive
            if avg_confidence > 0.75:
                adjustments['confidence_multiplier'] = 1.1
                self.analysis_results['recommendations'].append(
                    "High prediction confidence detected. Consider increasing position sizes by 10%."
                )
            elif avg_confidence < 0.60:
                adjustments['confidence_multiplier'] = 0.9
                self.analysis_results['recommendations'].append(
                    "Lower prediction confidence detected. Consider reducing position sizes by 10%."
                )
            else:
                adjustments['confidence_multiplier'] = 1.0
        
        # Placeholder for future model-specific adjustments based on actual trade outcomes
        adjustments['model_weights'] = current_weights
        adjustments['adjustment_reason'] = "Baseline weights - awaiting trade completion for performance-based adjustments"
        
        self.analysis_results['model_weight_adjustments'] = adjustments
        
        print(f"\n⚖️  MODEL WEIGHT ANALYSIS")
        print(f"   Current Weights: {current_weights}")
        print(f"   Confidence Multiplier: {adjustments.get('confidence_multiplier', 1.0):.2f}")
    
    def generate_strategy_insights(self, cycles):
        """Generate insights about trading strategies"""
        
        strategies_used = defaultdict(int)
        session_analysis = defaultdict(lambda: {'count': 0, 'strategies': []})
        
        for cycle in cycles:
            stages = cycle['data'].get('stages', {})
            session_info = stages.get('session_analysis', {})
            
            # Analyze stock session
            stock_session = session_info.get('stock_session', {})
            if stock_session.get('active'):
                session_name = stock_session.get('session', 'unknown')
                session_analysis[session_name]['count'] += 1
                session_analysis[session_name]['strategies'].extend(stock_session.get('strategies', []))
            
            # Analyze crypto session
            crypto_session = session_info.get('crypto_session', {})
            if crypto_session.get('active'):
                session_name = crypto_session.get('session', 'unknown')
                session_analysis[session_name]['count'] += 1
                session_analysis[session_name]['strategies'].extend(crypto_session.get('strategies', []))
        
        self.analysis_results['strategy_insights'] = {
            'sessions_active': {k: v['count'] for k, v in session_analysis.items()},
            'strategies_by_session': {k: list(set(v['strategies'])) for k, v in session_analysis.items()}
        }
        
        print(f"\n🎯 STRATEGY INSIGHTS")
        for session, data in session_analysis.items():
            print(f"   {session}: {data['count']} cycles, strategies: {list(set(data['strategies']))}")
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        
        pred_analysis = self.analysis_results.get('prediction_analysis', {})
        trade_analysis = self.analysis_results.get('trade_analysis', {})
        
        # Recommendation: Prediction diversity
        direction_dist = pred_analysis.get('direction_distribution', {})
        if direction_dist:
            long_count = direction_dist.get('LONG', 0)
            short_count = direction_dist.get('SHORT', 0)
            neutral_count = direction_dist.get('NEUTRAL', 0)
            total = sum(direction_dist.values())
            
            if long_count / total > 0.6:
                self.analysis_results['recommendations'].append(
                    "⚠️  Prediction bias toward LONG positions detected (>60%). Consider reviewing market conditions for potential overoptimism."
                )
            elif short_count / total > 0.6:
                self.analysis_results['recommendations'].append(
                    "⚠️  Prediction bias toward SHORT positions detected (>60%). Consider reviewing market conditions for potential overpessimism."
                )
        
        # Recommendation: Confidence levels
        avg_confidence = pred_analysis.get('average_confidence', 0)
        if avg_confidence < 0.65:
            self.analysis_results['recommendations'].append(
                "📊 Average prediction confidence is below 65%. Consider reducing position sizes or waiting for higher-confidence setups."
            )
        elif avg_confidence > 0.80:
            self.analysis_results['recommendations'].append(
                "✅ Strong prediction confidence detected (>80%). System is operating with high conviction."
            )
        
        # Recommendation: Open positions
        open_count = trade_analysis.get('open_positions', {}).get('count', 0)
        if open_count > 5:
            self.analysis_results['recommendations'].append(
                f"⚠️  {open_count} open positions detected. Monitor risk exposure and consider taking profits on winning positions."
            )
        elif open_count == 0:
            self.analysis_results['recommendations'].append(
                "💡 No open positions. System is in observation mode, waiting for high-quality setups."
            )
        
        # Recommendation: Asset class diversification
        asset_dist = pred_analysis.get('asset_class_distribution', {})
        if len(asset_dist) == 1:
            self.analysis_results['recommendations'].append(
                f"📌 All predictions focused on {list(asset_dist.keys())[0]}. Consider diversifying across asset classes for risk management."
            )
        
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(self.analysis_results['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    def calculate_overall_metrics(self):
        """Calculate overall system performance metrics"""
        
        pred_analysis = self.analysis_results.get('prediction_analysis', {})
        trade_analysis = self.analysis_results.get('trade_analysis', {})
        
        self.analysis_results['overall_metrics'] = {
            'total_predictions_today': pred_analysis.get('total_predictions', 0),
            'average_prediction_confidence': pred_analysis.get('average_confidence', 0),
            'open_positions': trade_analysis.get('open_positions', {}).get('count', 0),
            'closed_trades_today': trade_analysis.get('closed_trades', {}).get('count', 0),
            'win_rate': trade_analysis.get('closed_trades', {}).get('win_rate', 0),
            'total_pnl_today': trade_analysis.get('closed_trades', {}).get('total_pnl', 0),
            'system_status': 'LEARNING' if trade_analysis.get('closed_trades', {}).get('count', 0) == 0 else 'ACTIVE'
        }
        
        print(f"\n📊 OVERALL METRICS")
        print(f"   System Status: {self.analysis_results['overall_metrics']['system_status']}")
        print(f"   Predictions Today: {self.analysis_results['overall_metrics']['total_predictions_today']}")
        print(f"   Avg Confidence: {self.analysis_results['overall_metrics']['average_prediction_confidence']:.2%}")
        print(f"   Open Positions: {self.analysis_results['overall_metrics']['open_positions']}")
        print(f"   Closed Trades: {self.analysis_results['overall_metrics']['closed_trades_today']}")
    
    def save_learning_report(self):
        """Save the complete learning report"""
        
        learning_dir = self.data_dir / 'learning'
        learning_dir.mkdir(exist_ok=True)
        
        report_file = learning_dir / f'daily_reflection_{self.today}.json'
        
        with open(report_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"\n💾 Learning report saved to: {report_file}")
        return report_file
    
    def generate_markdown_report(self):
        """Generate a human-readable markdown report"""
        
        report = []
        report.append(f"# AI Prophet Daily Self-Reflection Report")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"")
        report.append(f"---")
        report.append(f"")
        
        # Overall Metrics
        report.append(f"## 📊 Overall Performance")
        report.append(f"")
        metrics = self.analysis_results['overall_metrics']
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| System Status | **{metrics['system_status']}** |")
        report.append(f"| Total Predictions | {metrics['total_predictions_today']} |")
        report.append(f"| Average Confidence | {metrics['average_prediction_confidence']:.2%} |")
        report.append(f"| Open Positions | {metrics['open_positions']} |")
        report.append(f"| Closed Trades | {metrics['closed_trades_today']} |")
        report.append(f"| Win Rate | {metrics['win_rate']:.2%} |")
        report.append(f"| Total P&L Today | ${metrics['total_pnl_today']:.2f} |")
        report.append(f"")
        
        # Prediction Analysis
        report.append(f"## 📈 Prediction Analysis")
        report.append(f"")
        pred = self.analysis_results['prediction_analysis']
        report.append(f"**Direction Distribution:**")
        for direction, count in pred.get('direction_distribution', {}).items():
            percentage = (count / pred['total_predictions'] * 100) if pred['total_predictions'] > 0 else 0
            report.append(f"- {direction}: {count} ({percentage:.1f}%)")
        report.append(f"")
        report.append(f"**Confidence Statistics:**")
        report.append(f"- Average: {pred.get('average_confidence', 0):.2%}")
        report.append(f"- Median: {pred.get('median_confidence', 0):.2%}")
        report.append(f"- Std Dev: {pred.get('confidence_std', 0):.4f}")
        report.append(f"- Range: {pred.get('min_confidence', 0):.2%} - {pred.get('max_confidence', 0):.2%}")
        report.append(f"")
        
        # Trade Analysis
        report.append(f"## 💼 Trade Analysis")
        report.append(f"")
        trade = self.analysis_results['trade_analysis']
        
        if trade.get('open_by_asset_class'):
            report.append(f"**Open Positions by Asset Class:**")
            for asset_class, count in trade['open_by_asset_class'].items():
                report.append(f"- {asset_class}: {count}")
            report.append(f"")
        
        if trade.get('open_by_side'):
            report.append(f"**Open Positions by Side:**")
            for side, count in trade['open_by_side'].items():
                report.append(f"- {side}: {count}")
            report.append(f"")
        
        # Model Weight Adjustments
        report.append(f"## ⚖️ Model Weight Adjustments")
        report.append(f"")
        weights = self.analysis_results['model_weight_adjustments']
        report.append(f"**Current Model Weights:**")
        for model, weight in weights.get('model_weights', {}).items():
            report.append(f"- {model}: {weight:.2%}")
        report.append(f"")
        report.append(f"**Confidence Multiplier:** {weights.get('confidence_multiplier', 1.0):.2f}")
        report.append(f"")
        report.append(f"*{weights.get('adjustment_reason', '')}*")
        report.append(f"")
        
        # Strategy Insights
        report.append(f"## 🎯 Strategy Insights")
        report.append(f"")
        strategy = self.analysis_results['strategy_insights']
        if strategy.get('sessions_active'):
            report.append(f"**Active Trading Sessions:**")
            for session, count in strategy['sessions_active'].items():
                strategies = strategy['strategies_by_session'].get(session, [])
                report.append(f"- {session}: {count} cycles")
                if strategies:
                    report.append(f"  - Strategies: {', '.join(strategies)}")
            report.append(f"")
        
        # Recommendations
        report.append(f"## 💡 Recommendations")
        report.append(f"")
        if self.analysis_results['recommendations']:
            for i, rec in enumerate(self.analysis_results['recommendations'], 1):
                report.append(f"{i}. {rec}")
        else:
            report.append(f"*No specific recommendations at this time. System is operating within normal parameters.*")
        report.append(f"")
        
        # Footer
        report.append(f"---")
        report.append(f"")
        report.append(f"*Generated by AI Prophet Daily Self-Reflection System*")
        report.append(f"")
        report.append(f"**Accuracy is everything. This report reflects AI Prophet's continuous learning and improvement process.**")
        
        markdown_content = "\n".join(report)
        
        # Save markdown report
        learning_dir = self.data_dir / 'learning'
        markdown_file = learning_dir / f'daily_reflection_{self.today}.md'
        
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Markdown report saved to: {markdown_file}")
        return markdown_file, markdown_content
    
    def run_complete_analysis(self):
        """Run the complete daily self-reflection analysis"""
        
        print("=" * 60)
        print("AI PROPHET DAILY SELF-REFLECTION")
        print("Analyzing what went right, what went wrong, and how to improve")
        print("=" * 60)
        
        # Load data
        cycles = self.load_all_cycles()
        state = self.load_day_trading_state()
        
        if not cycles:
            print("\n⚠️  No trading cycles found for today. System may not have run yet.")
            return None, None
        
        # Run analyses
        predictions = self.analyze_predictions(cycles)
        self.analyze_trades(state)
        self.calculate_model_weights(predictions, state)
        self.generate_strategy_insights(cycles)
        self.calculate_overall_metrics()
        self.generate_recommendations()
        
        # Save reports
        json_report = self.save_learning_report()
        markdown_report, markdown_content = self.generate_markdown_report()
        
        print("\n" + "=" * 60)
        print("✅ Daily self-reflection complete!")
        print("=" * 60)
        
        return json_report, markdown_report

def main():
    reflector = DailySelfReflection()
    json_report, markdown_report = reflector.run_complete_analysis()
    
    if json_report and markdown_report:
        print(f"\n📊 Reports generated:")
        print(f"   JSON: {json_report}")
        print(f"   Markdown: {markdown_report}")
        return 0
    else:
        print(f"\n⚠️  Analysis incomplete - insufficient data")
        return 1

if __name__ == '__main__':
    exit(main())
