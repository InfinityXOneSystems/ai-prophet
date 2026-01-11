#!/usr/bin/env python3
"""
AI PROPHET - Professional Day Trading Engine
==============================================
Pro-Level Trading Windows | Optimized Execution | Fast Results

Professional day trading schedule based on how the pros do it:
- Pre-market analysis (4:00 AM - 9:30 AM EST)
- Opening bell volatility (9:30 AM - 10:30 AM EST) - HIGHEST VOLUME
- Mid-morning momentum (10:30 AM - 12:00 PM EST)
- Lunch lull (12:00 PM - 2:00 PM EST) - Lower activity
- Power hour (3:00 PM - 4:00 PM EST) - SECOND HIGHEST VOLUME
- After-hours (4:00 PM - 8:00 PM EST)

For Crypto (24/7):
- Asian session (7:00 PM - 4:00 AM EST)
- European session (3:00 AM - 12:00 PM EST)
- US session (8:00 AM - 5:00 PM EST)

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | DAY_TRADING | %(levelname)s | %(message)s'
)
logger = logging.getLogger('DAY_TRADING')


class TradingSession(Enum):
    """Professional trading sessions"""
    PRE_MARKET = "pre_market"           # 4:00 AM - 9:30 AM EST
    OPENING_BELL = "opening_bell"       # 9:30 AM - 10:30 AM EST (HIGHEST VOLUME)
    MID_MORNING = "mid_morning"         # 10:30 AM - 12:00 PM EST
    LUNCH_LULL = "lunch_lull"           # 12:00 PM - 2:00 PM EST
    AFTERNOON = "afternoon"             # 2:00 PM - 3:00 PM EST
    POWER_HOUR = "power_hour"           # 3:00 PM - 4:00 PM EST (HIGH VOLUME)
    AFTER_HOURS = "after_hours"         # 4:00 PM - 8:00 PM EST
    
    # Crypto sessions (24/7)
    CRYPTO_ASIAN = "crypto_asian"       # 7:00 PM - 4:00 AM EST
    CRYPTO_EUROPEAN = "crypto_european" # 3:00 AM - 12:00 PM EST
    CRYPTO_US = "crypto_us"             # 8:00 AM - 5:00 PM EST


class AssetClass(Enum):
    """Asset classes for day trading"""
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    FUTURES = "futures"


@dataclass
class TradingWindow:
    """A trading window with optimal parameters"""
    session: TradingSession
    start_time: time
    end_time: time
    volatility_level: str  # "high", "medium", "low"
    volume_level: str      # "high", "medium", "low"
    recommended_strategies: List[str]
    risk_multiplier: float  # Adjust risk based on session
    
    def is_active(self, current_time: time) -> bool:
        """Check if this window is currently active"""
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:  # Crosses midnight
            return current_time >= self.start_time or current_time <= self.end_time


@dataclass
class DayTradingConfig:
    """Configuration for professional day trading"""
    # Portfolio settings
    starting_capital: float = 100000.0
    max_daily_loss: float = 0.02  # 2% max daily loss (PDT rule)
    max_position_size: float = 0.05  # 5% max per position
    max_concurrent_positions: int = 5
    
    # Day trading specific
    scalping_enabled: bool = True
    swing_trading_enabled: bool = True
    momentum_trading_enabled: bool = True
    
    # Risk management
    stop_loss_pct: float = 0.01  # 1% stop loss
    take_profit_pct: float = 0.02  # 2% take profit (2:1 R/R)
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.005  # 0.5% trailing
    
    # Execution
    min_confidence: float = 0.70
    max_trades_per_session: int = 10
    cooldown_after_loss: int = 300  # 5 minutes after loss


class ProTradingSchedule:
    """
    Professional trading schedule based on how the pros do it.
    Optimized for maximum profit potential and risk management.
    """
    
    # Stock market trading windows (EST)
    STOCK_WINDOWS = [
        TradingWindow(
            session=TradingSession.PRE_MARKET,
            start_time=time(4, 0),
            end_time=time(9, 30),
            volatility_level="medium",
            volume_level="low",
            recommended_strategies=["gap_analysis", "news_trading"],
            risk_multiplier=0.5
        ),
        TradingWindow(
            session=TradingSession.OPENING_BELL,
            start_time=time(9, 30),
            end_time=time(10, 30),
            volatility_level="high",
            volume_level="high",
            recommended_strategies=["momentum", "breakout", "gap_fade"],
            risk_multiplier=1.0  # Full risk - best opportunity
        ),
        TradingWindow(
            session=TradingSession.MID_MORNING,
            start_time=time(10, 30),
            end_time=time(12, 0),
            volatility_level="medium",
            volume_level="medium",
            recommended_strategies=["trend_following", "pullback"],
            risk_multiplier=0.8
        ),
        TradingWindow(
            session=TradingSession.LUNCH_LULL,
            start_time=time(12, 0),
            end_time=time(14, 0),
            volatility_level="low",
            volume_level="low",
            recommended_strategies=["range_trading", "scalping"],
            risk_multiplier=0.3  # Reduced risk - choppy market
        ),
        TradingWindow(
            session=TradingSession.AFTERNOON,
            start_time=time(14, 0),
            end_time=time(15, 0),
            volatility_level="medium",
            volume_level="medium",
            recommended_strategies=["trend_continuation"],
            risk_multiplier=0.6
        ),
        TradingWindow(
            session=TradingSession.POWER_HOUR,
            start_time=time(15, 0),
            end_time=time(16, 0),
            volatility_level="high",
            volume_level="high",
            recommended_strategies=["momentum", "closing_range"],
            risk_multiplier=0.9  # High risk - second best opportunity
        ),
        TradingWindow(
            session=TradingSession.AFTER_HOURS,
            start_time=time(16, 0),
            end_time=time(20, 0),
            volatility_level="medium",
            volume_level="low",
            recommended_strategies=["earnings_plays", "news_trading"],
            risk_multiplier=0.4
        ),
    ]
    
    # Crypto trading windows (24/7)
    CRYPTO_WINDOWS = [
        TradingWindow(
            session=TradingSession.CRYPTO_ASIAN,
            start_time=time(19, 0),
            end_time=time(4, 0),
            volatility_level="medium",
            volume_level="medium",
            recommended_strategies=["range_trading", "accumulation"],
            risk_multiplier=0.7
        ),
        TradingWindow(
            session=TradingSession.CRYPTO_EUROPEAN,
            start_time=time(3, 0),
            end_time=time(12, 0),
            volatility_level="high",
            volume_level="high",
            recommended_strategies=["breakout", "trend_following"],
            risk_multiplier=0.9
        ),
        TradingWindow(
            session=TradingSession.CRYPTO_US,
            start_time=time(8, 0),
            end_time=time(17, 0),
            volatility_level="high",
            volume_level="high",
            recommended_strategies=["momentum", "news_trading", "whale_watching"],
            risk_multiplier=1.0  # Full risk - highest volume
        ),
    ]
    
    @classmethod
    def get_current_window(cls, asset_class: AssetClass) -> Optional[TradingWindow]:
        """Get the current active trading window"""
        current = datetime.now().time()
        
        windows = cls.CRYPTO_WINDOWS if asset_class == AssetClass.CRYPTO else cls.STOCK_WINDOWS
        
        for window in windows:
            if window.is_active(current):
                return window
        
        return None
    
    @classmethod
    def get_optimal_execution_times(cls) -> List[Dict[str, Any]]:
        """
        Get optimal execution times for day trading.
        Based on professional trading patterns.
        """
        return [
            # Pre-market prep
            {"time": "04:00", "action": "pre_market_scan", "priority": "medium"},
            
            # Opening bell - CRITICAL
            {"time": "09:25", "action": "opening_prep", "priority": "high"},
            {"time": "09:30", "action": "opening_execution", "priority": "critical"},
            {"time": "09:45", "action": "opening_momentum", "priority": "critical"},
            {"time": "10:00", "action": "opening_continuation", "priority": "high"},
            
            # Mid-morning
            {"time": "10:30", "action": "mid_morning_scan", "priority": "medium"},
            {"time": "11:00", "action": "trend_check", "priority": "medium"},
            
            # Lunch - reduced activity
            {"time": "12:00", "action": "lunch_scan", "priority": "low"},
            
            # Afternoon prep
            {"time": "14:00", "action": "afternoon_prep", "priority": "medium"},
            {"time": "14:30", "action": "power_hour_prep", "priority": "high"},
            
            # Power hour - CRITICAL
            {"time": "15:00", "action": "power_hour_start", "priority": "critical"},
            {"time": "15:30", "action": "power_hour_momentum", "priority": "high"},
            {"time": "15:50", "action": "closing_positions", "priority": "high"},
            
            # After hours
            {"time": "16:00", "action": "after_hours_scan", "priority": "medium"},
            {"time": "18:00", "action": "daily_review", "priority": "high"},
            
            # Crypto specific (runs 24/7)
            {"time": "00:00", "action": "crypto_asian_check", "priority": "medium"},
            {"time": "03:00", "action": "crypto_european_open", "priority": "high"},
            {"time": "08:00", "action": "crypto_us_prep", "priority": "high"},
            {"time": "20:00", "action": "crypto_evening_scan", "priority": "medium"},
            {"time": "22:00", "action": "crypto_night_check", "priority": "low"},
        ]


@dataclass
class DayTrade:
    """A day trade record"""
    trade_id: str
    symbol: str
    asset_class: AssetClass
    entry_time: datetime
    entry_price: float
    quantity: float
    side: str  # "LONG" or "SHORT"
    strategy: str
    session: TradingSession
    
    # Exit info (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "stop_loss", "take_profit", "trailing_stop", "manual", "eod"
    
    # P&L
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Risk metrics
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'asset_class': self.asset_class.value,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'side': self.side,
            'strategy': self.strategy,
            'session': self.session.value,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio
        }


class DayTradingEngine:
    """
    Professional Day Trading Engine for AI Prophet.
    Executes trades based on pro-level timing and strategies.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / 'data')
        
        self.data_dir = Path(data_dir)
        self.day_trading_dir = self.data_dir / 'day_trading'
        self.day_trading_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = DayTradingConfig()
        self.schedule = ProTradingSchedule()
        
        # Portfolio state
        self.portfolio_value = self.config.starting_capital
        self.daily_pnl = 0.0
        self.open_positions: Dict[str, DayTrade] = {}
        self.closed_trades: List[DayTrade] = []
        
        # Session tracking
        self.trades_this_session = 0
        self.losses_today = 0
        self.last_loss_time: Optional[datetime] = None
        
        logger.info(f"Day Trading Engine initialized with ${self.config.starting_capital:,.2f}")
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """Get information about the current trading session"""
        now = datetime.now()
        
        # Check stock market window
        stock_window = self.schedule.get_current_window(AssetClass.STOCKS)
        crypto_window = self.schedule.get_current_window(AssetClass.CRYPTO)
        
        return {
            'current_time': now.isoformat(),
            'stock_session': {
                'active': stock_window is not None,
                'session': stock_window.session.value if stock_window else None,
                'volatility': stock_window.volatility_level if stock_window else None,
                'volume': stock_window.volume_level if stock_window else None,
                'risk_multiplier': stock_window.risk_multiplier if stock_window else 0,
                'strategies': stock_window.recommended_strategies if stock_window else []
            },
            'crypto_session': {
                'active': crypto_window is not None,
                'session': crypto_window.session.value if crypto_window else None,
                'volatility': crypto_window.volatility_level if crypto_window else None,
                'volume': crypto_window.volume_level if crypto_window else None,
                'risk_multiplier': crypto_window.risk_multiplier if crypto_window else 0,
                'strategies': crypto_window.recommended_strategies if crypto_window else []
            }
        }
    
    def can_trade(self, asset_class: AssetClass) -> Tuple[bool, str]:
        """Check if trading is allowed based on rules"""
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.config.max_daily_loss * self.portfolio_value:
            return False, "Daily loss limit reached"
        
        # Check concurrent positions
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return False, "Max concurrent positions reached"
        
        # Check session trade limit
        if self.trades_this_session >= self.config.max_trades_per_session:
            return False, "Session trade limit reached"
        
        # Check cooldown after loss
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(seconds=self.config.cooldown_after_loss)
            if datetime.now() < cooldown_end:
                return False, f"Cooldown active until {cooldown_end.strftime('%H:%M:%S')}"
        
        # Check if market is open
        window = self.schedule.get_current_window(asset_class)
        if not window and asset_class != AssetClass.CRYPTO:
            return False, "Market closed"
        
        return True, "OK"
    
    def calculate_position_size(self, asset_class: AssetClass, 
                               entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size based on risk"""
        window = self.schedule.get_current_window(asset_class)
        risk_multiplier = window.risk_multiplier if window else 0.5
        
        # Risk per trade (adjusted by session)
        risk_amount = self.portfolio_value * self.config.stop_loss_pct * risk_multiplier
        
        # Calculate shares/units based on stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        # Apply max position size limit
        max_position_value = self.portfolio_value * self.config.max_position_size
        max_units = max_position_value / entry_price
        
        return min(position_size, max_units)
    
    def execute_day_trade(self, symbol: str, asset_class: AssetClass,
                         side: str, entry_price: float, strategy: str,
                         confidence: float) -> Optional[DayTrade]:
        """Execute a day trade"""
        import uuid
        
        # Check if we can trade
        can_trade, reason = self.can_trade(asset_class)
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            return None
        
        # Check confidence
        if confidence < self.config.min_confidence:
            logger.warning(f"Confidence {confidence:.1%} below minimum {self.config.min_confidence:.1%}")
            return None
        
        # Calculate stop loss and take profit
        if side == "LONG":
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 - self.config.take_profit_pct)
        
        # Calculate position size
        quantity = self.calculate_position_size(asset_class, entry_price, stop_loss)
        if quantity <= 0:
            logger.warning("Position size calculation returned 0")
            return None
        
        # Get current session
        window = self.schedule.get_current_window(asset_class)
        session = window.session if window else TradingSession.AFTER_HOURS
        
        # Create trade
        trade = DayTrade(
            trade_id=f"DT-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            asset_class=asset_class,
            entry_time=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            side=side,
            strategy=strategy,
            session=session,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=self.config.take_profit_pct / self.config.stop_loss_pct
        )
        
        # Add to open positions
        self.open_positions[trade.trade_id] = trade
        self.trades_this_session += 1
        
        logger.info(f"OPENED: {side} {quantity:.4f} {symbol} @ ${entry_price:,.2f} | "
                   f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f} | "
                   f"Strategy: {strategy} | Session: {session.value}")
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float, 
                   exit_reason: str) -> Optional[DayTrade]:
        """Close an open trade"""
        if trade_id not in self.open_positions:
            logger.warning(f"Trade {trade_id} not found in open positions")
            return None
        
        trade = self.open_positions[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.side == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # SHORT
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Update portfolio
        self.portfolio_value += trade.pnl
        self.daily_pnl += trade.pnl
        
        # Track losses
        if trade.pnl < 0:
            self.losses_today += 1
            self.last_loss_time = datetime.now()
        
        # Move to closed trades
        del self.open_positions[trade_id]
        self.closed_trades.append(trade)
        
        logger.info(f"CLOSED: {trade.side} {trade.symbol} @ ${exit_price:,.2f} | "
                   f"P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:.2%}) | "
                   f"Reason: {exit_reason}")
        
        return trade
    
    def check_stops(self, current_prices: Dict[str, float]):
        """Check and execute stop losses and take profits"""
        for trade_id, trade in list(self.open_positions.items()):
            if trade.symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.symbol]
            
            if trade.side == "LONG":
                if current_price <= trade.stop_loss:
                    self.close_trade(trade_id, current_price, "stop_loss")
                elif current_price >= trade.take_profit:
                    self.close_trade(trade_id, current_price, "take_profit")
            else:  # SHORT
                if current_price >= trade.stop_loss:
                    self.close_trade(trade_id, current_price, "stop_loss")
                elif current_price <= trade.take_profit:
                    self.close_trade(trade_id, current_price, "take_profit")
    
    def end_of_day_close(self, current_prices: Dict[str, float]):
        """Close all positions at end of day (day trading rule)"""
        for trade_id, trade in list(self.open_positions.items()):
            if trade.symbol in current_prices:
                self.close_trade(trade_id, current_prices[trade.symbol], "eod")
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily trading summary"""
        winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl < 0]
        
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.config.starting_capital) * 100,
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100 if self.closed_trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'largest_win': max((t.pnl for t in winning_trades), default=0),
            'largest_loss': min((t.pnl for t in losing_trades), default=0),
            'open_positions': len(self.open_positions),
            'current_session': self.get_current_session_info()
        }
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0.0
        self.trades_this_session = 0
        self.losses_today = 0
        self.last_loss_time = None
        logger.info("Daily counters reset")
    
    def save_state(self):
        """Save engine state to file"""
        state = {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'open_positions': {k: v.to_dict() for k, v in self.open_positions.items()},
            'closed_trades': [t.to_dict() for t in self.closed_trades],
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = self.day_trading_dir / f"state_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filepath}")


def main():
    """Test the Day Trading Engine"""
    engine = DayTradingEngine()
    
    print("\n" + "="*60)
    print("AI PROPHET - PROFESSIONAL DAY TRADING ENGINE")
    print("Pro-Level Trading Windows | Optimized Execution")
    print("="*60)
    
    # Show current session
    session_info = engine.get_current_session_info()
    print(f"\n📊 CURRENT SESSION INFO")
    print(f"   Time: {session_info['current_time']}")
    
    print(f"\n   STOCKS:")
    stock = session_info['stock_session']
    if stock['active']:
        print(f"     Session: {stock['session']}")
        print(f"     Volatility: {stock['volatility']}")
        print(f"     Volume: {stock['volume']}")
        print(f"     Risk Multiplier: {stock['risk_multiplier']}")
        print(f"     Strategies: {', '.join(stock['strategies'])}")
    else:
        print("     Market Closed")
    
    print(f"\n   CRYPTO:")
    crypto = session_info['crypto_session']
    print(f"     Session: {crypto['session']}")
    print(f"     Volatility: {crypto['volatility']}")
    print(f"     Volume: {crypto['volume']}")
    print(f"     Risk Multiplier: {crypto['risk_multiplier']}")
    print(f"     Strategies: {', '.join(crypto['strategies'])}")
    
    # Show optimal execution times
    print("\n" + "="*60)
    print("OPTIMAL EXECUTION TIMES (PRO SCHEDULE)")
    print("="*60)
    
    for exec_time in ProTradingSchedule.get_optimal_execution_times():
        priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        print(f"   {exec_time['time']} | {priority_emoji.get(exec_time['priority'], '⚪')} {exec_time['action']}")


if __name__ == "__main__":
    main()
