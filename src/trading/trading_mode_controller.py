#!/usr/bin/env python3
"""
AI PROPHET - Trading Mode Controller
=====================================
Full Auto | Hybrid | Manual Trading Control System

Three modes of operation:
1. FULL AUTO: AI Prophet trades autonomously - zero human hands
2. HYBRID: User and AI Prophet collaborate - AI suggests, user approves
3. MANUAL: User has full control - AI provides insights only

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | MODE_CONTROLLER | %(levelname)s | %(message)s'
)
logger = logging.getLogger('MODE_CONTROLLER')


class TradingMode(Enum):
    """Trading control modes"""
    FULL_AUTO = "full_auto"      # AI Prophet trades autonomously
    HYBRID = "hybrid"            # User + AI Prophet collaborate
    MANUAL = "manual"            # User has full control


class SignalStrength(Enum):
    """Strength of trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    """A trading signal from AI Prophet"""
    signal_id: str
    symbol: str
    strength: SignalStrength
    direction: str  # "BUY" or "SELL"
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str
    model_used: str
    timestamp: datetime
    expires_at: datetime
    
    # Approval tracking (for HYBRID mode)
    approved: Optional[bool] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'strength': self.strength.value,
            'direction': self.direction,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'reasoning': self.reasoning,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'approved': self.approved,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'approved_by': self.approved_by
        }


@dataclass
class ModeConfig:
    """Configuration for a trading mode"""
    mode: TradingMode
    max_position_size: float  # % of portfolio
    max_daily_trades: int
    min_confidence: float
    require_approval: bool
    auto_stop_loss: bool
    auto_take_profit: bool
    risk_per_trade: float  # % of portfolio
    
    @classmethod
    def full_auto(cls) -> 'ModeConfig':
        """Configuration for full auto mode"""
        return cls(
            mode=TradingMode.FULL_AUTO,
            max_position_size=0.10,  # 10% max per position
            max_daily_trades=20,
            min_confidence=0.70,
            require_approval=False,
            auto_stop_loss=True,
            auto_take_profit=True,
            risk_per_trade=0.02  # 2% risk per trade
        )
    
    @classmethod
    def hybrid(cls) -> 'ModeConfig':
        """Configuration for hybrid mode"""
        return cls(
            mode=TradingMode.HYBRID,
            max_position_size=0.15,  # 15% max per position
            max_daily_trades=10,
            min_confidence=0.60,
            require_approval=True,
            auto_stop_loss=True,
            auto_take_profit=False,
            risk_per_trade=0.03  # 3% risk per trade
        )
    
    @classmethod
    def manual(cls) -> 'ModeConfig':
        """Configuration for manual mode"""
        return cls(
            mode=TradingMode.MANUAL,
            max_position_size=0.25,  # 25% max per position
            max_daily_trades=50,
            min_confidence=0.0,  # No minimum
            require_approval=True,
            auto_stop_loss=False,
            auto_take_profit=False,
            risk_per_trade=0.05  # 5% risk per trade
        )


class TradingModeStrategy(ABC):
    """Abstract base class for trading mode strategies"""
    
    @abstractmethod
    def process_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Process a trading signal according to the mode"""
        pass
    
    @abstractmethod
    def can_execute(self, signal: TradingSignal) -> bool:
        """Check if signal can be executed"""
        pass


class FullAutoStrategy(TradingModeStrategy):
    """
    Full Auto Trading Strategy
    
    AI Prophet trades autonomously with zero human intervention.
    All signals meeting criteria are executed immediately.
    """
    
    def __init__(self, config: ModeConfig):
        self.config = config
        self.trades_today = 0
        self.last_reset = datetime.now().date()
    
    def _reset_daily_counter(self):
        """Reset daily trade counter"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.trades_today = 0
            self.last_reset = today
    
    def can_execute(self, signal: TradingSignal) -> bool:
        """Check if signal meets auto-execution criteria"""
        self._reset_daily_counter()
        
        # Check daily trade limit
        if self.trades_today >= self.config.max_daily_trades:
            return False
        
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check if signal has expired
        if datetime.now() > signal.expires_at:
            return False
        
        return True
    
    def process_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Process signal for auto execution"""
        if not self.can_execute(signal):
            return {
                'action': 'SKIP',
                'reason': 'Signal does not meet auto-execution criteria',
                'signal_id': signal.signal_id
            }
        
        self.trades_today += 1
        
        return {
            'action': 'EXECUTE',
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'auto_stop_loss': self.config.auto_stop_loss,
            'stop_loss_price': signal.stop_loss,
            'auto_take_profit': self.config.auto_take_profit,
            'take_profit_price': signal.target_price,
            'max_position_size': self.config.max_position_size,
            'risk_per_trade': self.config.risk_per_trade
        }


class HybridStrategy(TradingModeStrategy):
    """
    Hybrid Trading Strategy
    
    AI Prophet suggests trades, user approves or rejects.
    Combines AI intelligence with human judgment.
    """
    
    def __init__(self, config: ModeConfig):
        self.config = config
        self.pending_signals: Dict[str, TradingSignal] = {}
    
    def can_execute(self, signal: TradingSignal) -> bool:
        """Check if signal can be executed (requires approval)"""
        if signal.approved is None:
            return False
        return signal.approved
    
    def process_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Process signal - queue for approval"""
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            return {
                'action': 'SKIP',
                'reason': f'Confidence {signal.confidence:.1%} below threshold {self.config.min_confidence:.1%}',
                'signal_id': signal.signal_id
            }
        
        # Queue for approval
        self.pending_signals[signal.signal_id] = signal
        
        return {
            'action': 'PENDING_APPROVAL',
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'expires_at': signal.expires_at.isoformat()
        }
    
    def approve_signal(self, signal_id: str, user_id: str) -> Dict[str, Any]:
        """Approve a pending signal"""
        if signal_id not in self.pending_signals:
            return {'error': 'Signal not found'}
        
        signal = self.pending_signals[signal_id]
        
        # Check if expired
        if datetime.now() > signal.expires_at:
            del self.pending_signals[signal_id]
            return {'error': 'Signal has expired'}
        
        signal.approved = True
        signal.approved_at = datetime.now()
        signal.approved_by = user_id
        
        return {
            'action': 'EXECUTE',
            'signal_id': signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'approved_by': user_id,
            'max_position_size': self.config.max_position_size
        }
    
    def reject_signal(self, signal_id: str, user_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject a pending signal"""
        if signal_id not in self.pending_signals:
            return {'error': 'Signal not found'}
        
        signal = self.pending_signals[signal_id]
        signal.approved = False
        signal.approved_at = datetime.now()
        signal.approved_by = user_id
        
        del self.pending_signals[signal_id]
        
        return {
            'action': 'REJECTED',
            'signal_id': signal_id,
            'rejected_by': user_id,
            'reason': reason
        }
    
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get all pending signals awaiting approval"""
        # Clean expired signals
        now = datetime.now()
        expired = [sid for sid, s in self.pending_signals.items() if now > s.expires_at]
        for sid in expired:
            del self.pending_signals[sid]
        
        return [s.to_dict() for s in self.pending_signals.values()]


class ManualStrategy(TradingModeStrategy):
    """
    Manual Trading Strategy
    
    User has full control. AI Prophet provides insights and suggestions
    but does not execute any trades automatically.
    """
    
    def __init__(self, config: ModeConfig):
        self.config = config
        self.insights: List[Dict[str, Any]] = []
    
    def can_execute(self, signal: TradingSignal) -> bool:
        """Manual mode never auto-executes"""
        return False
    
    def process_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Process signal as insight only"""
        insight = {
            'type': 'INSIGHT',
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'potential_return': ((signal.target_price - signal.entry_price) / signal.entry_price) * 100,
            'risk_reward_ratio': abs(signal.target_price - signal.entry_price) / abs(signal.entry_price - signal.stop_loss) if signal.stop_loss != signal.entry_price else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.insights.append(insight)
        
        # Keep only last 100 insights
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]
        
        return {
            'action': 'INSIGHT_PROVIDED',
            **insight
        }
    
    def get_insights(self, symbol: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent insights"""
        insights = self.insights
        if symbol:
            insights = [i for i in insights if i['symbol'] == symbol]
        return insights[-limit:]


class TradingModeController:
    """
    Main controller for trading modes.
    Manages mode switching and signal processing.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / 'data')
        
        self.data_dir = Path(data_dir)
        self.modes_dir = self.data_dir / 'trading_modes'
        self.modes_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategies
        self.strategies = {
            TradingMode.FULL_AUTO: FullAutoStrategy(ModeConfig.full_auto()),
            TradingMode.HYBRID: HybridStrategy(ModeConfig.hybrid()),
            TradingMode.MANUAL: ManualStrategy(ModeConfig.manual())
        }
        
        # User mode assignments
        self.user_modes: Dict[str, TradingMode] = {}
        
        self._load_user_modes()
        logger.info("Trading Mode Controller initialized")
    
    def _load_user_modes(self):
        """Load user mode assignments"""
        modes_file = self.modes_dir / 'user_modes.json'
        if modes_file.exists():
            with open(modes_file, 'r') as f:
                data = json.load(f)
                self.user_modes = {
                    k: TradingMode(v) for k, v in data.items()
                }
    
    def _save_user_modes(self):
        """Save user mode assignments"""
        modes_file = self.modes_dir / 'user_modes.json'
        with open(modes_file, 'w') as f:
            json.dump({k: v.value for k, v in self.user_modes.items()}, f, indent=2)
    
    def set_user_mode(self, user_id: str, mode: TradingMode) -> Dict[str, Any]:
        """Set trading mode for a user"""
        old_mode = self.user_modes.get(user_id)
        self.user_modes[user_id] = mode
        self._save_user_modes()
        
        logger.info(f"User {user_id} mode changed: {old_mode} -> {mode.value}")
        
        return {
            'user_id': user_id,
            'previous_mode': old_mode.value if old_mode else None,
            'new_mode': mode.value,
            'config': self._get_mode_config(mode)
        }
    
    def get_user_mode(self, user_id: str) -> TradingMode:
        """Get trading mode for a user (default: HYBRID)"""
        return self.user_modes.get(user_id, TradingMode.HYBRID)
    
    def _get_mode_config(self, mode: TradingMode) -> Dict[str, Any]:
        """Get configuration for a mode"""
        configs = {
            TradingMode.FULL_AUTO: ModeConfig.full_auto(),
            TradingMode.HYBRID: ModeConfig.hybrid(),
            TradingMode.MANUAL: ModeConfig.manual()
        }
        config = configs[mode]
        return {
            'mode': config.mode.value,
            'max_position_size': config.max_position_size,
            'max_daily_trades': config.max_daily_trades,
            'min_confidence': config.min_confidence,
            'require_approval': config.require_approval,
            'auto_stop_loss': config.auto_stop_loss,
            'auto_take_profit': config.auto_take_profit,
            'risk_per_trade': config.risk_per_trade
        }
    
    def process_signal(self, user_id: str, signal: TradingSignal) -> Dict[str, Any]:
        """Process a trading signal for a user based on their mode"""
        mode = self.get_user_mode(user_id)
        strategy = self.strategies[mode]
        
        result = strategy.process_signal(signal)
        result['user_id'] = user_id
        result['mode'] = mode.value
        
        return result
    
    def approve_signal(self, user_id: str, signal_id: str) -> Dict[str, Any]:
        """Approve a signal (HYBRID mode only)"""
        mode = self.get_user_mode(user_id)
        
        if mode != TradingMode.HYBRID:
            return {'error': 'Approval only available in HYBRID mode'}
        
        strategy = self.strategies[TradingMode.HYBRID]
        return strategy.approve_signal(signal_id, user_id)
    
    def reject_signal(self, user_id: str, signal_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject a signal (HYBRID mode only)"""
        mode = self.get_user_mode(user_id)
        
        if mode != TradingMode.HYBRID:
            return {'error': 'Rejection only available in HYBRID mode'}
        
        strategy = self.strategies[TradingMode.HYBRID]
        return strategy.reject_signal(signal_id, user_id, reason)
    
    def get_pending_approvals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending approvals for a user (HYBRID mode)"""
        mode = self.get_user_mode(user_id)
        
        if mode != TradingMode.HYBRID:
            return []
        
        strategy = self.strategies[TradingMode.HYBRID]
        return strategy.get_pending_signals()
    
    def get_insights(self, user_id: str, symbol: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get AI insights for a user (MANUAL mode)"""
        mode = self.get_user_mode(user_id)
        
        if mode != TradingMode.MANUAL:
            return []
        
        strategy = self.strategies[TradingMode.MANUAL]
        return strategy.get_insights(symbol, limit)
    
    def get_mode_summary(self) -> Dict[str, Any]:
        """Get summary of all trading modes"""
        return {
            'modes': {
                'full_auto': {
                    'name': 'Full Auto',
                    'description': 'AI Prophet trades autonomously - zero human hands',
                    'config': self._get_mode_config(TradingMode.FULL_AUTO)
                },
                'hybrid': {
                    'name': 'Hybrid',
                    'description': 'AI suggests, user approves - collaborative trading',
                    'config': self._get_mode_config(TradingMode.HYBRID)
                },
                'manual': {
                    'name': 'Manual',
                    'description': 'User has full control - AI provides insights only',
                    'config': self._get_mode_config(TradingMode.MANUAL)
                }
            },
            'user_distribution': {
                mode.value: len([u for u, m in self.user_modes.items() if m == mode])
                for mode in TradingMode
            }
        }


def main():
    """Test the Trading Mode Controller"""
    controller = TradingModeController()
    
    print("\n" + "="*60)
    print("AI PROPHET - TRADING MODE CONTROLLER")
    print("Full Auto | Hybrid | Manual")
    print("="*60)
    
    # Show mode summary
    summary = controller.get_mode_summary()
    print("\nAvailable Modes:")
    for mode_id, mode_info in summary['modes'].items():
        print(f"\n  {mode_info['name']}:")
        print(f"    {mode_info['description']}")
        print(f"    Max Position: {mode_info['config']['max_position_size']:.0%}")
        print(f"    Min Confidence: {mode_info['config']['min_confidence']:.0%}")
        print(f"    Require Approval: {mode_info['config']['require_approval']}")
    
    # Test signal processing
    print("\n" + "="*60)
    print("Testing Signal Processing")
    print("="*60)
    
    import uuid
    
    test_signal = TradingSignal(
        signal_id=f"SIG-{uuid.uuid4().hex[:8]}",
        symbol="BTC",
        strength=SignalStrength.STRONG_BUY,
        direction="BUY",
        confidence=0.85,
        entry_price=45000.0,
        target_price=48000.0,
        stop_loss=43000.0,
        reasoning="Strong momentum with bullish divergence",
        model_used="LSTM",
        timestamp=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=4)
    )
    
    # Test in each mode
    for mode in TradingMode:
        controller.set_user_mode("test_user", mode)
        result = controller.process_signal("test_user", test_signal)
        print(f"\n{mode.value.upper()} Mode:")
        print(f"  Action: {result['action']}")


if __name__ == "__main__":
    main()
