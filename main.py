#!/usr/bin/env python3
"""
AI PROPHET - Main Orchestrator
================================
The Wizard with Quantum AI Thinking Capabilities

AI Prophet is not a chatbot - he is a wizard with:
- Multi-timeline simulation (MAP parallel instances)
- Proven prediction models (AutoML, LSTM, Transformer)
- Recursive learning through trading feedback
- Accuracy tracking and self-reflection
- Full Auto / Hybrid / Manual trading modes

Everything AI Prophet says and does is calculated.
He always pulls up his data from the past to show users how accurate he is.
Accuracy is more important than anything.

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.prophet_core import AIProphet as AIProphetCore
from core.recursive_learning import RecursiveLearningEngine
from trading.paper_trading_engine import PaperTradingEngine, TradingMode, OrderSide, AITradingAgent
from predictions.vertex_automl_engine import VertexAutoMLEngine, ForecastHorizon
from simulations.timeline_simulator import TimelineSimulator, TimelineType
from scrapers.daily_scraper_pipeline import DailyScraperPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | AI_PROPHET | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_prophet.log')
    ]
)
logger = logging.getLogger('AI_PROPHET')


class AIProphet:
    """
    AI Prophet - The Prediction Wizard
    
    A fully autonomous prediction system that:
    1. Scrapes daily market data and events
    2. Generates predictions using proven models
    3. Simulates multiple future timelines
    4. Trades on predictions (paper trading)
    5. Tracks accuracy and learns from outcomes
    6. Self-reflects and evolves daily
    
    Accuracy is everything. If you're a talker but not accurate,
    you're just a chatbot. AI Prophet is a wizard.
    """
    
    VERSION = "1.0.0"
    CODENAME = "Quantum Wizard"
    
    def __init__(self, data_dir: str = None):
        """Initialize AI Prophet"""
        if data_dir is None:
            data_dir = str(Path(__file__).parent / 'data')
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"AI PROPHET v{self.VERSION} - {self.CODENAME}")
        logger.info("The Wizard with Quantum AI Thinking Capabilities")
        logger.info("="*60)
        
        # Initialize components
        self._init_components()
        
        # Track session
        self.session_start = datetime.now()
        self.predictions_made = 0
        self.trades_executed = 0
        
        logger.info("AI Prophet initialized and ready")
    
    def _init_components(self):
        """Initialize all AI Prophet components"""
        logger.info("Initializing components...")
        
        # Core prediction engine
        self.core = AIProphetCore(str(self.data_dir))
        
        # Paper trading engine
        self.trading_engine = PaperTradingEngine(str(self.data_dir))
        
        # AI trading agent
        self.trading_agent = AITradingAgent(self.trading_engine)
        
        # Recursive learning engine
        self.learning_engine = RecursiveLearningEngine(str(self.data_dir))
        
        # AutoML engine
        self.automl_engine = VertexAutoMLEngine(str(self.data_dir))
        
        # Timeline simulator
        self.simulator = TimelineSimulator(str(self.data_dir))
        
        # Daily scraper pipeline
        self.scraper = DailyScraperPipeline(str(self.data_dir))
        
        logger.info("All components initialized")
    
    async def run_daily_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete daily pipeline:
        1. Scrape latest data
        2. Generate predictions
        3. Simulate timelines
        4. Execute trades
        5. Evaluate past predictions
        6. Learn and adapt
        """
        logger.info("="*60)
        logger.info("STARTING DAILY PIPELINE")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        results = {
            'date': datetime.now().isoformat(),
            'pipeline_status': 'running',
            'stages': {}
        }
        
        try:
            # Stage 1: Scrape latest data
            logger.info("\n[STAGE 1] Scraping latest market data...")
            scrape_results = await self.scraper.run_all_scrapers()
            results['stages']['scraping'] = {
                'status': 'complete',
                'data_points': scrape_results.get('total_data_points', 0)
            }
            
            # Stage 2: Generate predictions
            logger.info("\n[STAGE 2] Generating predictions...")
            predictions = await self._generate_predictions()
            results['stages']['predictions'] = {
                'status': 'complete',
                'count': len(predictions)
            }
            
            # Stage 3: Simulate timelines
            logger.info("\n[STAGE 3] Simulating future timelines...")
            simulations = await self._run_simulations()
            results['stages']['simulations'] = {
                'status': 'complete',
                'timelines': len(simulations)
            }
            
            # Stage 4: Execute trades (AI Prophet's portfolio)
            logger.info("\n[STAGE 4] Executing trades...")
            trades = self._execute_trades(predictions)
            results['stages']['trading'] = {
                'status': 'complete',
                'trades_executed': len(trades)
            }
            
            # Stage 5: Evaluate past predictions
            logger.info("\n[STAGE 5] Evaluating past predictions...")
            evaluations = self._evaluate_past_predictions()
            results['stages']['evaluation'] = {
                'status': 'complete',
                'evaluated': evaluations.get('evaluated_count', 0)
            }
            
            # Stage 6: Learn and adapt
            logger.info("\n[STAGE 6] Running learning cycle...")
            learning_report = self.learning_engine.run_daily_learning_cycle()
            results['stages']['learning'] = {
                'status': 'complete',
                'accuracy': learning_report['overall_statistics']['overall_accuracy']
            }
            
            results['pipeline_status'] = 'complete'
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results['pipeline_status'] = 'error'
            results['error'] = str(e)
        
        # Save pipeline results
        self._save_pipeline_results(results)
        
        logger.info("\n" + "="*60)
        logger.info("DAILY PIPELINE COMPLETE")
        logger.info(f"Status: {results['pipeline_status']}")
        logger.info("="*60)
        
        return results
    
    async def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for all tracked assets"""
        predictions = []
        
        for symbol in list(self.trading_engine.assets.keys())[:10]:  # Top 10 assets
            try:
                prediction = self.core.predict(
                    symbol=symbol,
                    horizon_days=7,
                    model='ensemble'
                )
                
                if prediction:
                    predictions.append(prediction)
                    self.predictions_made += 1
                    
                    logger.info(f"  {symbol}: {prediction.get('direction', 'N/A')} "
                              f"(confidence: {prediction.get('confidence', 0):.1%})")
            
            except Exception as e:
                logger.error(f"  {symbol}: Prediction failed - {e}")
        
        return predictions
    
    async def _run_simulations(self) -> List[Any]:
        """Run multi-timeline simulations"""
        all_simulations = []
        
        # Simulate for top 5 assets
        for symbol in list(self.trading_engine.assets.keys())[:5]:
            try:
                asset = self.trading_engine.assets[symbol]
                simulations = await self.simulator.simulate_parallel_timelines(
                    target_asset=symbol,
                    num_timelines=5,
                    days_ahead=30,
                    initial_price=asset.current_price
                )
                all_simulations.extend(simulations)
                
                logger.info(f"  {symbol}: {len(simulations)} timelines simulated")
            
            except Exception as e:
                logger.error(f"  {symbol}: Simulation failed - {e}")
        
        return all_simulations
    
    def _execute_trades(self, predictions: List[Dict[str, Any]]) -> List[Any]:
        """Execute trades based on predictions"""
        trades = self.trading_agent.run_trading_cycle(predictions)
        self.trades_executed += len(trades)
        
        for trade in trades:
            logger.info(f"  Trade: {trade.side.value} {trade.quantity:.4f} {trade.symbol} @ ${trade.price:,.2f}")
        
        return trades
    
    def _evaluate_past_predictions(self) -> Dict[str, Any]:
        """Evaluate predictions that have reached their horizon"""
        # Get predictions from 7 days ago
        cutoff = datetime.now() - timedelta(days=7)
        
        # In production, load past predictions and compare with actual outcomes
        evaluated_count = 0
        
        return {
            'evaluated_count': evaluated_count,
            'cutoff_date': cutoff.isoformat()
        }
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file"""
        results_dir = self.data_dir / 'pipeline_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Pipeline results saved to {filepath}")
    
    def get_accuracy_report(self) -> Dict[str, Any]:
        """
        Get AI Prophet's accuracy report.
        This is what AI Prophet cares about most.
        """
        learning_report = self.learning_engine.generate_learning_report(days=30)
        simulation_stats = self.simulator.get_simulation_accuracy_stats()
        
        ai_portfolio = self.trading_engine.get_ai_portfolio()
        portfolio_stats = ai_portfolio.get_stats() if ai_portfolio else None
        
        return {
            'report_date': datetime.now().isoformat(),
            'prediction_accuracy': {
                'overall': learning_report['overall_statistics']['overall_accuracy'],
                'total_predictions': learning_report['overall_statistics']['total_predictions'],
                'correct_predictions': learning_report['overall_statistics']['correct_predictions']
            },
            'simulation_accuracy': simulation_stats,
            'trading_performance': {
                'total_value': portfolio_stats.total_value if portfolio_stats else 0,
                'total_pnl_pct': portfolio_stats.total_pnl_pct if portfolio_stats else 0,
                'win_rate': portfolio_stats.win_rate if portfolio_stats else 0,
                'max_drawdown': portfolio_stats.max_drawdown if portfolio_stats else 0
            },
            'model_performances': learning_report.get('model_performances', []),
            'recommendations': learning_report.get('recommendations', [])
        }
    
    def show_accuracy(self):
        """Display AI Prophet's accuracy - his most important metric"""
        report = self.get_accuracy_report()
        
        print("\n" + "="*60)
        print("AI PROPHET ACCURACY REPORT")
        print("Because accuracy is everything.")
        print("="*60)
        
        print(f"\n📊 PREDICTION ACCURACY")
        print(f"   Overall: {report['prediction_accuracy']['overall']:.1%}")
        print(f"   Predictions: {report['prediction_accuracy']['total_predictions']}")
        print(f"   Correct: {report['prediction_accuracy']['correct_predictions']}")
        
        print(f"\n💰 TRADING PERFORMANCE")
        print(f"   Portfolio Value: ${report['trading_performance']['total_value']:,.2f}")
        print(f"   Total P&L: {report['trading_performance']['total_pnl_pct']:.2f}%")
        print(f"   Win Rate: {report['trading_performance']['win_rate']:.1%}")
        print(f"   Max Drawdown: {report['trading_performance']['max_drawdown']:.1%}")
        
        if report['model_performances']:
            print(f"\n🤖 MODEL PERFORMANCES")
            for model in report['model_performances'][:3]:
                print(f"   {model['model_name']}: {model['accuracy']:.1%} accuracy")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in report['recommendations'][:3]:
                print(f"   • {rec}")
        
        print("\n" + "="*60)
    
    def predict(self, symbol: str, horizon_days: int = 7) -> Dict[str, Any]:
        """
        Make a prediction for a symbol.
        AI Prophet always shows his reasoning and confidence.
        """
        prediction = self.core.predict(
            symbol=symbol,
            horizon_days=horizon_days,
            model='ensemble'
        )
        
        if prediction:
            # Adjust confidence based on learning
            adjusted_confidence = self.learning_engine.get_adjusted_confidence(
                base_confidence=prediction.get('confidence', 0.5),
                model=prediction.get('model', 'ensemble'),
                symbol=symbol
            )
            prediction['adjusted_confidence'] = adjusted_confidence
            
            # Add historical accuracy for this symbol
            symbol_insights = self.learning_engine.analyze_symbol_insights(symbol=symbol)
            if symbol_insights:
                prediction['historical_accuracy'] = symbol_insights[0].accuracy
        
        return prediction
    
    async def simulate(self, symbol: str, num_timelines: int = 5, 
                      days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Simulate multiple future timelines.
        AI Prophet's quantum thinking capability.
        """
        if symbol not in self.trading_engine.assets:
            return []
        
        asset = self.trading_engine.assets[symbol]
        
        timelines = await self.simulator.simulate_parallel_timelines(
            target_asset=symbol,
            num_timelines=num_timelines,
            days_ahead=days_ahead,
            initial_price=asset.current_price
        )
        
        return [
            {
                'timeline_id': tl.timeline_id,
                'type': tl.timeline_type.value,
                'probability': tl.probability,
                'final_price': tl.final_prediction['price'],
                'change_pct': tl.final_prediction['change_pct'],
                'direction': tl.final_prediction['direction'],
                'theory_basis': tl.theory_basis
            }
            for tl in timelines
        ]


async def main():
    """Main entry point for AI Prophet"""
    parser = argparse.ArgumentParser(description='AI Prophet - The Prediction Wizard')
    parser.add_argument('--mode', choices=['pipeline', 'accuracy', 'predict', 'simulate', 'api'],
                       default='accuracy', help='Operation mode')
    parser.add_argument('--symbol', type=str, help='Symbol for prediction/simulation')
    parser.add_argument('--days', type=int, default=7, help='Horizon days')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    
    args = parser.parse_args()
    
    # Initialize AI Prophet
    prophet = AIProphet()
    
    if args.mode == 'pipeline':
        # Run daily pipeline
        await prophet.run_daily_pipeline()
    
    elif args.mode == 'accuracy':
        # Show accuracy report
        prophet.show_accuracy()
    
    elif args.mode == 'predict':
        # Make a prediction
        if not args.symbol:
            print("Error: --symbol required for predict mode")
            return
        
        prediction = prophet.predict(args.symbol, args.days)
        print(json.dumps(prediction, indent=2))
    
    elif args.mode == 'simulate':
        # Run simulation
        if not args.symbol:
            print("Error: --symbol required for simulate mode")
            return
        
        timelines = await prophet.simulate(args.symbol, num_timelines=5, days_ahead=args.days)
        print(json.dumps(timelines, indent=2))
    
    elif args.mode == 'api':
        # Start API server
        from api.dashboard_api import run_server
        run_server(port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
