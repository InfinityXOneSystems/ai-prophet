#!/usr/bin/env python3
"""
AI PROPHET - Day Trading Runner
================================
Professional Day Trading Execution | 2-Hour Cycles | Self-Reflection

This script runs AI Prophet in day trading mode with:
- Professional trading windows (Opening Bell, Power Hour, etc.)
- 2-hour execution cycles for fast results
- Continuous self-reflection and learning
- Optimized for crypto (24/7) and stocks (market hours)

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

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
from trading.paper_trading_engine import PaperTradingEngine, TradingMode, OrderSide
from trading.day_trading_engine import DayTradingEngine, AssetClass, ProTradingSchedule
from predictions.vertex_automl_engine import VertexAutoMLEngine
from simulations.timeline_simulator import TimelineSimulator
from scrapers.daily_scraper_pipeline import DailyScraperPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | AI_PROPHET_DAY_TRADE | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_prophet_day_trading.log')
    ]
)
logger = logging.getLogger('AI_PROPHET_DAY_TRADE')


class AIProphetDayTrader:
    """
    AI Prophet Day Trading System
    
    Professional day trading with:
    - 2-hour execution cycles
    - Pro trading windows (Opening Bell, Power Hour)
    - Continuous self-reflection
    - Recursive learning from every trade
    """
    
    # Day trading focused assets (high liquidity, good volatility)
    DAY_TRADING_ASSETS = {
        AssetClass.CRYPTO: [
            {"symbol": "BTC", "name": "Bitcoin", "volatility": "high"},
            {"symbol": "ETH", "name": "Ethereum", "volatility": "high"},
            {"symbol": "SOL", "name": "Solana", "volatility": "very_high"},
            {"symbol": "XRP", "name": "Ripple", "volatility": "high"},
            {"symbol": "DOGE", "name": "Dogecoin", "volatility": "very_high"},
            {"symbol": "ADA", "name": "Cardano", "volatility": "high"},
            {"symbol": "AVAX", "name": "Avalanche", "volatility": "very_high"},
            {"symbol": "LINK", "name": "Chainlink", "volatility": "high"},
        ],
        AssetClass.STOCKS: [
            {"symbol": "SPY", "name": "S&P 500 ETF", "volatility": "medium"},
            {"symbol": "QQQ", "name": "Nasdaq ETF", "volatility": "medium"},
            {"symbol": "TSLA", "name": "Tesla", "volatility": "high"},
            {"symbol": "NVDA", "name": "NVIDIA", "volatility": "high"},
            {"symbol": "AMD", "name": "AMD", "volatility": "high"},
            {"symbol": "AAPL", "name": "Apple", "volatility": "medium"},
            {"symbol": "AMZN", "name": "Amazon", "volatility": "medium"},
            {"symbol": "META", "name": "Meta", "volatility": "high"},
        ]
    }
    
    def __init__(self, starting_capital: float = 100000.0):
        """Initialize AI Prophet Day Trader"""
        self.data_dir = Path(__file__).parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("AI PROPHET - DAY TRADING MODE")
        logger.info("Professional Day Trading | 2-Hour Cycles")
        logger.info("="*60)
        
        # Initialize components
        self.day_trading_engine = DayTradingEngine(str(self.data_dir))
        self.day_trading_engine.config.starting_capital = starting_capital
        self.day_trading_engine.portfolio_value = starting_capital
        
        self.core = AIProphetCore(str(self.data_dir))
        self.learning_engine = RecursiveLearningEngine(str(self.data_dir))
        self.simulator = TimelineSimulator(str(self.data_dir))
        self.scraper = DailyScraperPipeline(str(self.data_dir))
        
        # Tracking
        self.cycle_count = 0
        self.total_predictions = 0
        self.correct_predictions = 0
        self.session_start = datetime.now()
        
        # Simulated prices (in production, these come from real APIs)
        self.current_prices = self._initialize_prices()
        
        logger.info(f"Starting Capital: ${starting_capital:,.2f}")
        logger.info("Day Trader initialized and ready")
    
    def _initialize_prices(self) -> Dict[str, float]:
        """Initialize simulated prices for assets"""
        import random
        
        prices = {}
        
        # Crypto prices (approximate)
        crypto_base = {
            "BTC": 45000, "ETH": 2500, "SOL": 100, "XRP": 0.60,
            "DOGE": 0.08, "ADA": 0.50, "AVAX": 35, "LINK": 15
        }
        
        # Stock prices (approximate)
        stock_base = {
            "SPY": 480, "QQQ": 410, "TSLA": 250, "NVDA": 500,
            "AMD": 140, "AAPL": 185, "AMZN": 155, "META": 350
        }
        
        for symbol, base in {**crypto_base, **stock_base}.items():
            # Add some random variation
            variation = random.uniform(-0.02, 0.02)
            prices[symbol] = base * (1 + variation)
        
        return prices
    
    def _update_prices(self):
        """Simulate price movements"""
        import random
        
        for symbol in self.current_prices:
            # Random walk with mean reversion
            change = random.gauss(0, 0.005)  # 0.5% std dev
            self.current_prices[symbol] *= (1 + change)
    
    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run a single 2-hour trading cycle.
        This is the core execution loop.
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info(f"CYCLE {self.cycle_count} STARTING")
        logger.info(f"Time: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        results = {
            'cycle': self.cycle_count,
            'start_time': cycle_start.isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Update market data
            logger.info("\n[STAGE 1] Updating market data...")
            self._update_prices()
            results['stages']['data_update'] = {'status': 'complete'}
            
            # Stage 2: Check current session
            logger.info("\n[STAGE 2] Analyzing current session...")
            session_info = self.day_trading_engine.get_current_session_info()
            results['stages']['session_analysis'] = session_info
            
            # Stage 3: Check stops on open positions
            logger.info("\n[STAGE 3] Checking stops...")
            self.day_trading_engine.check_stops(self.current_prices)
            results['stages']['stop_check'] = {
                'open_positions': len(self.day_trading_engine.open_positions)
            }
            
            # Stage 4: Generate predictions
            logger.info("\n[STAGE 4] Generating predictions...")
            predictions = await self._generate_predictions()
            results['stages']['predictions'] = {
                'count': len(predictions),
                'predictions': predictions
            }
            
            # Stage 5: Execute trades based on predictions
            logger.info("\n[STAGE 5] Executing trades...")
            trades = self._execute_trades(predictions)
            results['stages']['trades'] = {
                'executed': len(trades),
                'trades': [t.to_dict() for t in trades]
            }
            
            # Stage 6: Self-reflection
            logger.info("\n[STAGE 6] Self-reflection...")
            reflection = self._self_reflect()
            results['stages']['reflection'] = reflection
            
            # Stage 7: Learning update
            logger.info("\n[STAGE 7] Learning update...")
            learning = self._update_learning()
            results['stages']['learning'] = learning
            
            # Get daily summary
            summary = self.day_trading_engine.get_daily_summary()
            results['summary'] = summary
            
            # Log summary
            logger.info("\n" + "-"*40)
            logger.info("CYCLE SUMMARY")
            logger.info("-"*40)
            logger.info(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
            logger.info(f"Daily P&L: ${summary['daily_pnl']:,.2f} ({summary['daily_pnl_pct']:.2f}%)")
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
            logger.info(f"Open Positions: {summary['open_positions']}")
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            results['error'] = str(e)
        
        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = (datetime.now() - cycle_start).total_seconds()
        
        # Save cycle results
        self._save_cycle_results(results)
        
        return results
    
    async def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for day trading assets"""
        predictions = []
        
        # Determine which assets to analyze based on current session
        session_info = self.day_trading_engine.get_current_session_info()
        
        # Always analyze crypto (24/7)
        for asset in self.DAY_TRADING_ASSETS[AssetClass.CRYPTO]:
            prediction = self._predict_asset(asset['symbol'], AssetClass.CRYPTO)
            if prediction:
                predictions.append(prediction)
        
        # Analyze stocks if market is open
        if session_info['stock_session']['active']:
            for asset in self.DAY_TRADING_ASSETS[AssetClass.STOCKS]:
                prediction = self._predict_asset(asset['symbol'], AssetClass.STOCKS)
                if prediction:
                    predictions.append(prediction)
        
        return predictions
    
    def _predict_asset(self, symbol: str, asset_class: AssetClass) -> Optional[Dict[str, Any]]:
        """Generate prediction for a single asset"""
        import random
        
        current_price = self.current_prices.get(symbol, 0)
        if current_price == 0:
            return None
        
        # Simulate prediction (in production, use actual models)
        direction = random.choice(["LONG", "SHORT", "NEUTRAL"])
        confidence = random.uniform(0.5, 0.95)
        
        # Calculate targets
        if direction == "LONG":
            target_price = current_price * (1 + random.uniform(0.01, 0.03))
            stop_loss = current_price * (1 - random.uniform(0.005, 0.015))
        elif direction == "SHORT":
            target_price = current_price * (1 - random.uniform(0.01, 0.03))
            stop_loss = current_price * (1 + random.uniform(0.005, 0.015))
        else:
            target_price = current_price
            stop_loss = current_price
        
        self.total_predictions += 1
        
        return {
            'symbol': symbol,
            'asset_class': asset_class.value,
            'current_price': current_price,
            'direction': direction,
            'confidence': confidence,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_trades(self, predictions: List[Dict[str, Any]]) -> List:
        """Execute trades based on predictions"""
        trades = []
        
        for pred in predictions:
            # Skip neutral predictions
            if pred['direction'] == "NEUTRAL":
                continue
            
            # Check confidence threshold
            if pred['confidence'] < 0.70:
                continue
            
            # Execute trade
            asset_class = AssetClass(pred['asset_class'])
            trade = self.day_trading_engine.execute_day_trade(
                symbol=pred['symbol'],
                asset_class=asset_class,
                side=pred['direction'],
                entry_price=pred['current_price'],
                strategy="ai_prophet_prediction",
                confidence=pred['confidence']
            )
            
            if trade:
                trades.append(trade)
        
        return trades
    
    def _self_reflect(self) -> Dict[str, Any]:
        """
        AI Prophet's self-reflection.
        Analyzes what went right and wrong.
        """
        closed_trades = self.day_trading_engine.closed_trades
        
        if not closed_trades:
            return {'status': 'no_trades_to_analyze'}
        
        # Analyze recent trades
        recent = closed_trades[-10:]  # Last 10 trades
        
        wins = [t for t in recent if t.pnl and t.pnl > 0]
        losses = [t for t in recent if t.pnl and t.pnl < 0]
        
        # Identify patterns
        winning_strategies = {}
        losing_strategies = {}
        
        for t in wins:
            winning_strategies[t.strategy] = winning_strategies.get(t.strategy, 0) + 1
        
        for t in losses:
            losing_strategies[t.strategy] = losing_strategies.get(t.strategy, 0) + 1
        
        # Generate insights
        insights = []
        
        if len(losses) > len(wins):
            insights.append("More losses than wins - consider tightening entry criteria")
        
        if losses:
            avg_loss = sum(t.pnl for t in losses) / len(losses)
            insights.append(f"Average loss: ${avg_loss:.2f} - review stop loss placement")
        
        if wins:
            avg_win = sum(t.pnl for t in wins) / len(wins)
            insights.append(f"Average win: ${avg_win:.2f}")
        
        reflection = {
            'analyzed_trades': len(recent),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(recent) * 100 if recent else 0,
            'winning_strategies': winning_strategies,
            'losing_strategies': losing_strategies,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Self-reflection: {len(wins)} wins, {len(losses)} losses")
        for insight in insights:
            logger.info(f"  Insight: {insight}")
        
        return reflection
    
    def _update_learning(self) -> Dict[str, Any]:
        """Update learning based on recent performance"""
        # In production, this would update model weights
        
        summary = self.day_trading_engine.get_daily_summary()
        
        learning_update = {
            'win_rate': summary['win_rate'],
            'profit_factor': summary['profit_factor'],
            'adjustments': []
        }
        
        # Make adjustments based on performance
        if summary['win_rate'] < 50:
            learning_update['adjustments'].append("Increase confidence threshold")
        
        if summary['profit_factor'] < 1.5:
            learning_update['adjustments'].append("Review risk/reward ratio")
        
        return learning_update
    
    def _save_cycle_results(self, results: Dict[str, Any]):
        """Save cycle results to file"""
        results_dir = self.data_dir / 'day_trading_cycles'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"cycle_{self.cycle_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        summary = self.day_trading_engine.get_daily_summary()
        
        return {
            'session_start': self.session_start.isoformat(),
            'cycles_completed': self.cycle_count,
            'total_predictions': self.total_predictions,
            'portfolio': {
                'starting_capital': self.day_trading_engine.config.starting_capital,
                'current_value': summary['portfolio_value'],
                'total_pnl': summary['portfolio_value'] - self.day_trading_engine.config.starting_capital,
                'total_pnl_pct': ((summary['portfolio_value'] / self.day_trading_engine.config.starting_capital) - 1) * 100
            },
            'trading': {
                'total_trades': summary['total_trades'],
                'winning_trades': summary['winning_trades'],
                'losing_trades': summary['losing_trades'],
                'win_rate': summary['win_rate'],
                'profit_factor': summary['profit_factor'],
                'largest_win': summary['largest_win'],
                'largest_loss': summary['largest_loss']
            },
            'current_session': summary['current_session']
        }
    
    def show_performance(self):
        """Display performance report"""
        report = self.get_performance_report()
        
        print("\n" + "="*60)
        print("AI PROPHET - DAY TRADING PERFORMANCE")
        print("="*60)
        
        print(f"\n📊 SESSION INFO")
        print(f"   Started: {report['session_start']}")
        print(f"   Cycles: {report['cycles_completed']}")
        print(f"   Predictions: {report['total_predictions']}")
        
        print(f"\n💰 PORTFOLIO")
        print(f"   Starting: ${report['portfolio']['starting_capital']:,.2f}")
        print(f"   Current: ${report['portfolio']['current_value']:,.2f}")
        print(f"   P&L: ${report['portfolio']['total_pnl']:,.2f} ({report['portfolio']['total_pnl_pct']:.2f}%)")
        
        print(f"\n📈 TRADING")
        print(f"   Total Trades: {report['trading']['total_trades']}")
        print(f"   Win Rate: {report['trading']['win_rate']:.1f}%")
        print(f"   Profit Factor: {report['trading']['profit_factor']:.2f}")
        print(f"   Largest Win: ${report['trading']['largest_win']:,.2f}")
        print(f"   Largest Loss: ${report['trading']['largest_loss']:,.2f}")
        
        print("\n" + "="*60)


async def run_single_cycle():
    """Run a single trading cycle"""
    trader = AIProphetDayTrader(starting_capital=100000.0)
    
    # Run one cycle
    results = await trader.run_cycle()
    
    # Show performance
    trader.show_performance()
    
    return results


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Prophet Day Trading')
    parser.add_argument('--cycles', type=int, default=1, help='Number of cycles to run')
    parser.add_argument('--capital', type=float, default=100000.0, help='Starting capital')
    
    args = parser.parse_args()
    
    trader = AIProphetDayTrader(starting_capital=args.capital)
    
    for i in range(args.cycles):
        await trader.run_cycle()
        
        if i < args.cycles - 1:
            logger.info(f"Waiting before next cycle...")
            await asyncio.sleep(5)  # Short delay between cycles for testing
    
    trader.show_performance()
    trader.day_trading_engine.save_state()


if __name__ == "__main__":
    asyncio.run(main())
