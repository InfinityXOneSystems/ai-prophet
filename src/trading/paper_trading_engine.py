#!/usr/bin/env python3
"""
AI PROPHET - Paper Trading & Portfolio Management Engine
=========================================================
Full Auto | Hybrid | Manual Trading Modes
Paper Trading | Testnet Crypto | Real-Time Statistics

Features:
- User portfolios with customizable starting capital
- Full Auto: AI Prophet trades autonomously
- Hybrid: User and AI Prophet trade together
- Manual: User has full control
- Real-time P&L tracking and statistics
- AI Prophet's own portfolio for recursive learning

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | PAPER_TRADING | %(levelname)s | %(message)s'
)
logger = logging.getLogger('PAPER_TRADING')


class TradingMode(Enum):
    """Trading control modes"""
    FULL_AUTO = "full_auto"      # AI Prophet trades autonomously
    HYBRID = "hybrid"            # User + AI Prophet collaborate
    MANUAL = "manual"            # User has full control


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class AssetClass(Enum):
    """Asset classes for trading"""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass
class Asset:
    """Tradeable asset"""
    symbol: str
    name: str
    asset_class: AssetClass
    current_price: float
    price_precision: int = 2
    quantity_precision: int = 8
    min_quantity: float = 0.0001
    
    # Prediction metadata
    predictability_score: float = 0.5  # 0-1, higher = easier to predict
    volatility: float = 0.0
    correlation_btc: float = 0.0  # For crypto
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'asset_class': self.asset_class.value,
            'current_price': self.current_price,
            'predictability_score': self.predictability_score,
            'volatility': self.volatility
        }


@dataclass
class Position:
    """A position in an asset"""
    position_id: str
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = ((new_price / self.entry_price) - 1) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'market_value': self.quantity * self.current_price
        }


@dataclass
class Order:
    """A trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # None for market orders
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    ai_generated: bool = False
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'ai_generated': self.ai_generated,
            'reasoning': self.reasoning
        }


@dataclass
class Trade:
    """A completed trade"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    value: float
    fee: float
    timestamp: datetime
    ai_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'value': self.value,
            'fee': self.fee,
            'timestamp': self.timestamp.isoformat(),
            'ai_generated': self.ai_generated
        }


@dataclass
class PortfolioStats:
    """Portfolio statistics"""
    total_value: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    total_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    best_trade: float
    worst_trade: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'positions_value': self.positions_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade
        }


class Portfolio:
    """
    User or AI Prophet portfolio for paper trading.
    Tracks all positions, orders, trades, and statistics.
    """
    
    def __init__(self, portfolio_id: str, owner_id: str, 
                 initial_capital: float, trading_mode: TradingMode,
                 is_ai_portfolio: bool = False):
        self.portfolio_id = portfolio_id
        self.owner_id = owner_id
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.trading_mode = trading_mode
        self.is_ai_portfolio = is_ai_portfolio
        self.created_at = datetime.now()
        
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        
        # Statistics
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio {portfolio_id} created with ${initial_capital:,.2f}")
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash_balance + positions_value
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   ai_generated: bool = False, reasoning: str = "") -> Order:
        """Place a new order"""
        order_id = f"ORD-{uuid.uuid4().hex[:8]}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            ai_generated=ai_generated,
            reasoning=reasoning
        )
        
        self.orders[order_id] = order
        logger.info(f"Order placed: {order_id} {side.value} {quantity} {symbol}")
        
        return order
    
    def execute_order(self, order_id: str, execution_price: float, 
                     fee_rate: float = 0.001) -> Optional[Trade]:
        """Execute a pending order"""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return None
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} is not pending")
            return None
        
        # Calculate trade value and fee
        trade_value = order.quantity * execution_price
        fee = trade_value * fee_rate
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = trade_value + fee
            if total_cost > self.cash_balance:
                order.status = OrderStatus.REJECTED
                logger.error(f"Insufficient funds for order {order_id}")
                return None
            
            self.cash_balance -= total_cost
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                avg_price = ((pos.quantity * pos.entry_price) + (order.quantity * execution_price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
                pos.current_price = execution_price
            else:
                self.positions[order.symbol] = Position(
                    position_id=f"POS-{uuid.uuid4().hex[:8]}",
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    entry_time=datetime.now(),
                    current_price=execution_price
                )
        
        else:  # SELL
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                logger.error(f"No position to sell for {order.symbol}")
                return None
            
            pos = self.positions[order.symbol]
            if order.quantity > pos.quantity:
                order.status = OrderStatus.REJECTED
                logger.error(f"Insufficient quantity to sell")
                return None
            
            # Calculate realized P&L
            pnl = (execution_price - pos.entry_price) * order.quantity
            self.realized_pnl += pnl
            
            # Update cash and position
            self.cash_balance += trade_value - fee
            pos.quantity -= order.quantity
            
            if pos.quantity <= 0:
                del self.positions[order.symbol]
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        
        # Create trade record
        trade = Trade(
            trade_id=f"TRD-{uuid.uuid4().hex[:8]}",
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            value=trade_value,
            fee=fee,
            timestamp=datetime.now(),
            ai_generated=order.ai_generated
        )
        
        self.trades.append(trade)
        self.total_fees += fee
        
        # Update equity history
        current_equity = self.get_total_value()
        self.equity_history.append((datetime.now(), current_equity))
        
        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        logger.info(f"Trade executed: {trade.trade_id} {order.side.value} {order.quantity} {order.symbol} @ ${execution_price}")
        
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def get_stats(self) -> PortfolioStats:
        """Calculate comprehensive portfolio statistics"""
        total_value = self.get_total_value()
        positions_value = total_value - self.cash_balance
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = self.realized_pnl + unrealized_pnl
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.side == OrderSide.SELL]
        # Simplified - in production, track P&L per trade
        win_count = len([t for t in winning_trades])
        loss_count = 0
        
        return PortfolioStats(
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            win_rate=win_count / len(self.trades) if self.trades else 0,
            total_trades=len(self.trades),
            winning_trades=win_count,
            losing_trades=loss_count,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=self.max_drawdown,
            best_trade=0.0,
            worst_trade=0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize portfolio to dictionary"""
        stats = self.get_stats()
        return {
            'portfolio_id': self.portfolio_id,
            'owner_id': self.owner_id,
            'initial_capital': self.initial_capital,
            'trading_mode': self.trading_mode.value,
            'is_ai_portfolio': self.is_ai_portfolio,
            'created_at': self.created_at.isoformat(),
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'open_orders': [o.to_dict() for o in self.orders.values() if o.status == OrderStatus.PENDING],
            'recent_trades': [t.to_dict() for t in self.trades[-10:]],
            'stats': stats.to_dict()
        }


class PaperTradingEngine:
    """
    Paper Trading Engine for AI Prophet
    
    Manages:
    - User portfolios
    - AI Prophet's own portfolio (recursive learning)
    - Order execution simulation
    - Real-time price feeds
    - Trading mode switching
    """
    
    # Diversified assets that are easier to predict
    TRACKED_ASSETS = [
        # Crypto - High predictability due to pattern recognition
        Asset("BTC", "Bitcoin", AssetClass.CRYPTO, 45000.0, predictability_score=0.75, volatility=0.04),
        Asset("ETH", "Ethereum", AssetClass.CRYPTO, 2500.0, predictability_score=0.72, volatility=0.05),
        Asset("SOL", "Solana", AssetClass.CRYPTO, 100.0, predictability_score=0.68, volatility=0.06),
        Asset("BNB", "Binance Coin", AssetClass.CRYPTO, 300.0, predictability_score=0.70, volatility=0.04),
        Asset("XRP", "Ripple", AssetClass.CRYPTO, 0.50, predictability_score=0.65, volatility=0.05),
        
        # Stocks - Blue chips with predictable patterns
        Asset("AAPL", "Apple Inc", AssetClass.STOCK, 180.0, predictability_score=0.78, volatility=0.02),
        Asset("MSFT", "Microsoft", AssetClass.STOCK, 380.0, predictability_score=0.80, volatility=0.02),
        Asset("GOOGL", "Alphabet", AssetClass.STOCK, 140.0, predictability_score=0.76, volatility=0.02),
        Asset("NVDA", "NVIDIA", AssetClass.STOCK, 500.0, predictability_score=0.65, volatility=0.04),
        Asset("AMZN", "Amazon", AssetClass.STOCK, 180.0, predictability_score=0.74, volatility=0.03),
        
        # ETFs - Highly predictable due to diversification
        Asset("SPY", "S&P 500 ETF", AssetClass.INDEX, 480.0, predictability_score=0.85, volatility=0.01),
        Asset("QQQ", "Nasdaq 100 ETF", AssetClass.INDEX, 420.0, predictability_score=0.82, volatility=0.015),
        Asset("GLD", "Gold ETF", AssetClass.COMMODITY, 190.0, predictability_score=0.80, volatility=0.01),
        
        # Forex - Major pairs with high liquidity
        Asset("EUR/USD", "Euro/Dollar", AssetClass.FOREX, 1.08, predictability_score=0.70, volatility=0.005),
        Asset("GBP/USD", "Pound/Dollar", AssetClass.FOREX, 1.27, predictability_score=0.68, volatility=0.006),
    ]
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / 'data')
        
        self.data_dir = Path(data_dir)
        self.portfolios_dir = self.data_dir / 'portfolios'
        self.portfolios_dir.mkdir(parents=True, exist_ok=True)
        
        self.portfolios: Dict[str, Portfolio] = {}
        self.assets: Dict[str, Asset] = {a.symbol: a for a in self.TRACKED_ASSETS}
        
        # Initialize AI Prophet's own portfolio
        self._init_ai_portfolio()
        
        logger.info(f"Paper Trading Engine initialized with {len(self.assets)} tracked assets")
    
    def _init_ai_portfolio(self):
        """Initialize AI Prophet's own portfolio for recursive learning"""
        ai_portfolio_id = "AI_PROPHET_MASTER"
        
        # Check if exists
        ai_portfolio_file = self.portfolios_dir / f"{ai_portfolio_id}.json"
        if ai_portfolio_file.exists():
            self._load_portfolio(ai_portfolio_id)
        else:
            # Create new AI portfolio with $1M starting capital
            self.create_portfolio(
                owner_id="AI_PROPHET",
                initial_capital=1000000.0,
                trading_mode=TradingMode.FULL_AUTO,
                portfolio_id=ai_portfolio_id,
                is_ai_portfolio=True
            )
        
        logger.info("AI Prophet's master portfolio initialized")
    
    def create_portfolio(self, owner_id: str, initial_capital: float,
                        trading_mode: TradingMode = TradingMode.HYBRID,
                        portfolio_id: str = None,
                        is_ai_portfolio: bool = False) -> Portfolio:
        """Create a new portfolio for a user"""
        if portfolio_id is None:
            portfolio_id = f"PORT-{uuid.uuid4().hex[:8]}"
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            owner_id=owner_id,
            initial_capital=initial_capital,
            trading_mode=trading_mode,
            is_ai_portfolio=is_ai_portfolio
        )
        
        self.portfolios[portfolio_id] = portfolio
        self._save_portfolio(portfolio)
        
        return portfolio
    
    def _save_portfolio(self, portfolio: Portfolio):
        """Save portfolio to storage"""
        file_path = self.portfolios_dir / f"{portfolio.portfolio_id}.json"
        with open(file_path, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)
    
    def _load_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Load portfolio from storage"""
        file_path = self.portfolios_dir / f"{portfolio_id}.json"
        if not file_path.exists():
            return None
        
        # Simplified loading - in production, fully deserialize
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        portfolio = Portfolio(
            portfolio_id=data['portfolio_id'],
            owner_id=data['owner_id'],
            initial_capital=data['initial_capital'],
            trading_mode=TradingMode(data['trading_mode']),
            is_ai_portfolio=data.get('is_ai_portfolio', False)
        )
        
        self.portfolios[portfolio_id] = portfolio
        return portfolio
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get a portfolio by ID"""
        if portfolio_id in self.portfolios:
            return self.portfolios[portfolio_id]
        return self._load_portfolio(portfolio_id)
    
    def set_trading_mode(self, portfolio_id: str, mode: TradingMode):
        """Change trading mode for a portfolio"""
        portfolio = self.get_portfolio(portfolio_id)
        if portfolio:
            portfolio.trading_mode = mode
            self._save_portfolio(portfolio)
            logger.info(f"Portfolio {portfolio_id} trading mode set to {mode.value}")
    
    def execute_market_order(self, portfolio_id: str, symbol: str,
                            side: OrderSide, quantity: float,
                            ai_generated: bool = False,
                            reasoning: str = "") -> Optional[Trade]:
        """Execute a market order immediately"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return None
        
        if symbol not in self.assets:
            logger.error(f"Unknown asset: {symbol}")
            return None
        
        # Place and execute order
        order = portfolio.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            ai_generated=ai_generated,
            reasoning=reasoning
        )
        
        # Execute at current price
        current_price = self.assets[symbol].current_price
        trade = portfolio.execute_order(order.order_id, current_price)
        
        if trade:
            self._save_portfolio(portfolio)
        
        return trade
    
    def update_asset_prices(self, prices: Dict[str, float]):
        """Update asset prices from market data"""
        for symbol, price in prices.items():
            if symbol in self.assets:
                self.assets[symbol].current_price = price
        
        # Update all portfolio positions
        for portfolio in self.portfolios.values():
            portfolio.update_prices(prices)
    
    def get_ai_portfolio(self) -> Portfolio:
        """Get AI Prophet's master portfolio"""
        return self.portfolios.get("AI_PROPHET_MASTER")
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get portfolio leaderboard by performance"""
        portfolios_with_stats = []
        
        for portfolio in self.portfolios.values():
            stats = portfolio.get_stats()
            portfolios_with_stats.append({
                'portfolio_id': portfolio.portfolio_id,
                'owner_id': portfolio.owner_id,
                'is_ai': portfolio.is_ai_portfolio,
                'total_value': stats.total_value,
                'total_pnl_pct': stats.total_pnl_pct,
                'win_rate': stats.win_rate,
                'total_trades': stats.total_trades
            })
        
        # Sort by P&L percentage
        portfolios_with_stats.sort(key=lambda x: x['total_pnl_pct'], reverse=True)
        
        return portfolios_with_stats[:limit]
    
    def get_tracked_assets(self) -> List[Dict[str, Any]]:
        """Get list of tracked assets with predictability scores"""
        return [
            {
                **asset.to_dict(),
                'rank': i + 1
            }
            for i, asset in enumerate(
                sorted(self.assets.values(), 
                      key=lambda x: x.predictability_score, 
                      reverse=True)
            )
        ]


class AITradingAgent:
    """
    AI Prophet's autonomous trading agent.
    Makes trading decisions based on predictions and executes them.
    """
    
    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.portfolio = engine.get_ai_portfolio()
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk management parameters
        self.max_position_size = 0.10  # 10% of portfolio per position
        self.max_daily_trades = 20
        self.min_confidence = 0.70  # Minimum prediction confidence to trade
        
        logger.info("AI Trading Agent initialized")
    
    def analyze_opportunity(self, symbol: str, 
                           prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a trading opportunity based on prediction"""
        if symbol not in self.engine.assets:
            return None
        
        asset = self.engine.assets[symbol]
        confidence = prediction.get('confidence', 0)
        direction = prediction.get('direction', 'NEUTRAL')
        
        # Check minimum confidence
        if confidence < self.min_confidence:
            return None
        
        # Calculate position size based on confidence and predictability
        base_size = self.max_position_size
        adjusted_size = base_size * confidence * asset.predictability_score
        
        # Determine action
        if direction == 'UP':
            action = OrderSide.BUY
        elif direction == 'DOWN':
            action = OrderSide.SELL
        else:
            return None
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'position_size_pct': adjusted_size,
            'reasoning': prediction.get('reasoning', ''),
            'predictability_score': asset.predictability_score
        }
    
    def execute_trade(self, opportunity: Dict[str, Any]) -> Optional[Trade]:
        """Execute a trade based on analyzed opportunity"""
        if not self.portfolio:
            return None
        
        symbol = opportunity['symbol']
        action = opportunity['action']
        position_size_pct = opportunity['position_size_pct']
        
        # Calculate quantity
        portfolio_value = self.portfolio.get_total_value()
        trade_value = portfolio_value * position_size_pct
        current_price = self.engine.assets[symbol].current_price
        quantity = trade_value / current_price
        
        # Execute trade
        trade = self.engine.execute_market_order(
            portfolio_id=self.portfolio.portfolio_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            ai_generated=True,
            reasoning=opportunity['reasoning']
        )
        
        if trade:
            self.trade_history.append({
                'trade': trade.to_dict(),
                'opportunity': opportunity,
                'timestamp': datetime.now().isoformat()
            })
        
        return trade
    
    def run_trading_cycle(self, predictions: List[Dict[str, Any]]) -> List[Trade]:
        """Run a complete trading cycle with multiple predictions"""
        executed_trades = []
        
        for prediction in predictions:
            symbol = prediction.get('symbol')
            if not symbol:
                continue
            
            # Analyze opportunity
            opportunity = self.analyze_opportunity(symbol, prediction)
            if not opportunity:
                continue
            
            # Execute trade
            trade = self.execute_trade(opportunity)
            if trade:
                executed_trades.append(trade)
            
            # Check daily trade limit
            if len(executed_trades) >= self.max_daily_trades:
                break
        
        logger.info(f"Trading cycle complete: {len(executed_trades)} trades executed")
        return executed_trades
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get AI agent's performance summary for recursive learning"""
        if not self.portfolio:
            return {}
        
        stats = self.portfolio.get_stats()
        
        return {
            'portfolio_id': self.portfolio.portfolio_id,
            'total_value': stats.total_value,
            'total_pnl': stats.total_pnl,
            'total_pnl_pct': stats.total_pnl_pct,
            'win_rate': stats.win_rate,
            'total_trades': stats.total_trades,
            'max_drawdown': stats.max_drawdown,
            'trade_history_count': len(self.trade_history),
            'last_updated': datetime.now().isoformat()
        }


def main():
    """Test the Paper Trading Engine"""
    engine = PaperTradingEngine()
    
    print("\n" + "="*60)
    print("AI PROPHET - PAPER TRADING ENGINE")
    print("Full Auto | Hybrid | Manual Trading Modes")
    print("="*60)
    
    # Show tracked assets
    print("\nTracked Assets (by Predictability):")
    for asset in engine.get_tracked_assets()[:10]:
        print(f"  {asset['rank']}. {asset['symbol']}: {asset['name']}")
        print(f"     Predictability: {asset['predictability_score']:.0%}")
        print(f"     Current Price: ${asset['current_price']:,.2f}")
    
    # Create a test user portfolio
    print("\n" + "="*60)
    print("Creating Test User Portfolio")
    print("="*60)
    
    user_portfolio = engine.create_portfolio(
        owner_id="test_user",
        initial_capital=10000.0,
        trading_mode=TradingMode.HYBRID
    )
    
    print(f"\nPortfolio ID: {user_portfolio.portfolio_id}")
    print(f"Initial Capital: ${user_portfolio.initial_capital:,.2f}")
    print(f"Trading Mode: {user_portfolio.trading_mode.value}")
    
    # Execute a test trade
    print("\n" + "="*60)
    print("Executing Test Trade")
    print("="*60)
    
    trade = engine.execute_market_order(
        portfolio_id=user_portfolio.portfolio_id,
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=0.1,
        reasoning="Test trade"
    )
    
    if trade:
        print(f"\nTrade Executed:")
        print(f"  Symbol: {trade.symbol}")
        print(f"  Side: {trade.side.value}")
        print(f"  Quantity: {trade.quantity}")
        print(f"  Price: ${trade.price:,.2f}")
        print(f"  Value: ${trade.value:,.2f}")
    
    # Show portfolio stats
    stats = user_portfolio.get_stats()
    print(f"\nPortfolio Stats:")
    print(f"  Total Value: ${stats.total_value:,.2f}")
    print(f"  Cash Balance: ${stats.cash_balance:,.2f}")
    print(f"  Positions Value: ${stats.positions_value:,.2f}")
    
    # Show AI Portfolio
    print("\n" + "="*60)
    print("AI Prophet's Master Portfolio")
    print("="*60)
    
    ai_portfolio = engine.get_ai_portfolio()
    if ai_portfolio:
        ai_stats = ai_portfolio.get_stats()
        print(f"\nPortfolio ID: {ai_portfolio.portfolio_id}")
        print(f"Total Value: ${ai_stats.total_value:,.2f}")
        print(f"Trading Mode: {ai_portfolio.trading_mode.value}")


if __name__ == "__main__":
    main()
