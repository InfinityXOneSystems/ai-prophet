#!/usr/bin/env python3
"""
AI PROPHET - Dashboard API
===========================
Real-Time Portfolio Statistics | User Dashboard | Leaderboard

FastAPI backend for the AI Prophet trading dashboard.
Provides real-time statistics, portfolio management, and AI trading controls.

110% Protocol | FAANG Enterprise-Grade | Zero Human Hands
"""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import AI Prophet modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from trading.paper_trading_engine import (
    PaperTradingEngine, Portfolio, TradingMode, 
    OrderSide, OrderType, Asset
)
from core.recursive_learning import RecursiveLearningEngine
from simulations.timeline_simulator import TimelineSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | DASHBOARD_API | %(levelname)s | %(message)s'
)
logger = logging.getLogger('DASHBOARD_API')

# Initialize FastAPI app
app = FastAPI(
    title="AI Prophet Dashboard API",
    description="Real-time trading dashboard for AI Prophet prediction system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
trading_engine = PaperTradingEngine()
learning_engine = RecursiveLearningEngine()
simulator = TimelineSimulator()


# ============== Pydantic Models ==============

class CreatePortfolioRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    initial_capital: float = Field(..., ge=100, description="Starting capital")
    trading_mode: str = Field(default="hybrid", description="Trading mode: full_auto, hybrid, manual")


class PlaceOrderRequest(BaseModel):
    portfolio_id: str
    symbol: str
    side: str = Field(..., description="buy or sell")
    quantity: float = Field(..., gt=0)
    order_type: str = Field(default="market")
    price: Optional[float] = None


class SetTradingModeRequest(BaseModel):
    portfolio_id: str
    mode: str = Field(..., description="full_auto, hybrid, or manual")


class PredictionRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(default=7, ge=1, le=90)


class SimulationRequest(BaseModel):
    symbol: str
    num_timelines: int = Field(default=5, ge=1, le=10)
    days_ahead: int = Field(default=30, ge=1, le=365)
    initial_price: Optional[float] = None


# ============== Dashboard Endpoints ==============

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "AI Prophet Dashboard API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "trading_engine": "operational",
            "learning_engine": "operational",
            "simulator": "operational"
        },
        "timestamp": datetime.now().isoformat()
    }


# ============== Portfolio Endpoints ==============

@app.post("/portfolio/create")
async def create_portfolio(request: CreatePortfolioRequest):
    """Create a new user portfolio"""
    try:
        mode = TradingMode(request.trading_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid trading mode")
    
    portfolio = trading_engine.create_portfolio(
        owner_id=request.user_id,
        initial_capital=request.initial_capital,
        trading_mode=mode
    )
    
    return {
        "success": True,
        "portfolio_id": portfolio.portfolio_id,
        "message": f"Portfolio created with ${request.initial_capital:,.2f}"
    }


@app.get("/portfolio/{portfolio_id}")
async def get_portfolio(portfolio_id: str):
    """Get portfolio details and statistics"""
    portfolio = trading_engine.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return portfolio.to_dict()


@app.get("/portfolio/{portfolio_id}/stats")
async def get_portfolio_stats(portfolio_id: str):
    """Get detailed portfolio statistics"""
    portfolio = trading_engine.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    stats = portfolio.get_stats()
    return stats.to_dict()


@app.get("/portfolio/{portfolio_id}/positions")
async def get_portfolio_positions(portfolio_id: str):
    """Get all open positions"""
    portfolio = trading_engine.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return {
        "portfolio_id": portfolio_id,
        "positions": [pos.to_dict() for pos in portfolio.positions.values()]
    }


@app.get("/portfolio/{portfolio_id}/trades")
async def get_portfolio_trades(portfolio_id: str, limit: int = 50):
    """Get trade history"""
    portfolio = trading_engine.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    trades = portfolio.trades[-limit:] if limit else portfolio.trades
    return {
        "portfolio_id": portfolio_id,
        "total_trades": len(portfolio.trades),
        "trades": [t.to_dict() for t in reversed(trades)]
    }


@app.post("/portfolio/mode")
async def set_trading_mode(request: SetTradingModeRequest):
    """Change trading mode for a portfolio"""
    try:
        mode = TradingMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid trading mode")
    
    trading_engine.set_trading_mode(request.portfolio_id, mode)
    
    return {
        "success": True,
        "portfolio_id": request.portfolio_id,
        "new_mode": mode.value,
        "message": f"Trading mode changed to {mode.value}"
    }


# ============== Trading Endpoints ==============

@app.post("/trade/order")
async def place_order(request: PlaceOrderRequest):
    """Place a trading order"""
    try:
        side = OrderSide(request.side)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid order side")
    
    trade = trading_engine.execute_market_order(
        portfolio_id=request.portfolio_id,
        symbol=request.symbol,
        side=side,
        quantity=request.quantity,
        ai_generated=False,
        reasoning="Manual order"
    )
    
    if not trade:
        raise HTTPException(status_code=400, detail="Order execution failed")
    
    return {
        "success": True,
        "trade": trade.to_dict()
    }


@app.get("/assets")
async def get_tracked_assets():
    """Get list of tracked assets with predictability scores"""
    return {
        "assets": trading_engine.get_tracked_assets(),
        "total": len(trading_engine.assets)
    }


@app.get("/assets/{symbol}")
async def get_asset(symbol: str):
    """Get details for a specific asset"""
    if symbol not in trading_engine.assets:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    asset = trading_engine.assets[symbol]
    return asset.to_dict()


# ============== AI Prophet Endpoints ==============

@app.get("/ai-prophet/portfolio")
async def get_ai_portfolio():
    """Get AI Prophet's master portfolio"""
    portfolio = trading_engine.get_ai_portfolio()
    if not portfolio:
        raise HTTPException(status_code=404, detail="AI portfolio not found")
    
    return portfolio.to_dict()


@app.get("/ai-prophet/performance")
async def get_ai_performance():
    """Get AI Prophet's trading performance summary"""
    portfolio = trading_engine.get_ai_portfolio()
    if not portfolio:
        raise HTTPException(status_code=404, detail="AI portfolio not found")
    
    stats = portfolio.get_stats()
    
    return {
        "portfolio_id": portfolio.portfolio_id,
        "initial_capital": portfolio.initial_capital,
        "current_value": stats.total_value,
        "total_pnl": stats.total_pnl,
        "total_pnl_pct": stats.total_pnl_pct,
        "win_rate": stats.win_rate,
        "total_trades": stats.total_trades,
        "max_drawdown": stats.max_drawdown,
        "profit_factor": stats.profit_factor,
        "sharpe_ratio": stats.sharpe_ratio
    }


@app.get("/ai-prophet/learning-report")
async def get_learning_report(days: int = 30):
    """Get AI Prophet's learning report"""
    report = learning_engine.generate_learning_report(days=days)
    return report


@app.get("/ai-prophet/accuracy")
async def get_ai_accuracy():
    """Get AI Prophet's prediction accuracy statistics"""
    report = learning_engine.generate_learning_report(days=30)
    
    return {
        "overall_accuracy": report['overall_statistics']['overall_accuracy'],
        "total_predictions": report['overall_statistics']['total_predictions'],
        "correct_predictions": report['overall_statistics']['correct_predictions'],
        "signal_distribution": report['signal_distribution'],
        "model_performances": report['model_performances']
    }


# ============== Simulation Endpoints ==============

@app.post("/simulate/timelines")
async def simulate_timelines(request: SimulationRequest):
    """Simulate multiple future timelines for an asset"""
    import asyncio
    
    if request.symbol not in trading_engine.assets:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    initial_price = request.initial_price or trading_engine.assets[request.symbol].current_price
    
    timelines = await simulator.simulate_parallel_timelines(
        target_asset=request.symbol,
        num_timelines=request.num_timelines,
        days_ahead=request.days_ahead,
        initial_price=initial_price
    )
    
    return {
        "symbol": request.symbol,
        "num_timelines": len(timelines),
        "timelines": [
            {
                "timeline_id": tl.timeline_id,
                "type": tl.timeline_type.value,
                "probability": tl.probability,
                "final_price": tl.final_prediction['price'],
                "change_pct": tl.final_prediction['change_pct'],
                "direction": tl.final_prediction['direction'],
                "theory_basis": tl.theory_basis
            }
            for tl in timelines
        ]
    }


@app.get("/simulate/active")
async def get_active_simulations():
    """Get all active (pending) simulations"""
    return {
        "active_simulations": simulator.get_active_simulations()
    }


@app.get("/simulate/accuracy")
async def get_simulation_accuracy():
    """Get simulation accuracy statistics"""
    return simulator.get_simulation_accuracy_stats()


# ============== Leaderboard Endpoints ==============

@app.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get portfolio leaderboard"""
    return {
        "leaderboard": trading_engine.get_leaderboard(limit=limit)
    }


# ============== Dashboard Data Endpoints ==============

@app.get("/dashboard/{portfolio_id}")
async def get_dashboard_data(portfolio_id: str):
    """Get all dashboard data for a user in one call"""
    portfolio = trading_engine.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    stats = portfolio.get_stats()
    ai_portfolio = trading_engine.get_ai_portfolio()
    ai_stats = ai_portfolio.get_stats() if ai_portfolio else None
    
    return {
        "user_portfolio": {
            "portfolio_id": portfolio.portfolio_id,
            "trading_mode": portfolio.trading_mode.value,
            "stats": stats.to_dict(),
            "positions": [pos.to_dict() for pos in portfolio.positions.values()],
            "recent_trades": [t.to_dict() for t in portfolio.trades[-5:]]
        },
        "ai_prophet": {
            "portfolio_id": ai_portfolio.portfolio_id if ai_portfolio else None,
            "stats": ai_stats.to_dict() if ai_stats else None,
            "accuracy": learning_engine.generate_learning_report(days=7)['overall_statistics']['overall_accuracy']
        },
        "market": {
            "tracked_assets": len(trading_engine.assets),
            "top_predictable": trading_engine.get_tracked_assets()[:5]
        },
        "leaderboard": trading_engine.get_leaderboard(limit=5),
        "timestamp": datetime.now().isoformat()
    }


# ============== WebSocket for Real-Time Updates ==============

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


@app.websocket("/ws/{portfolio_id}")
async def websocket_endpoint(websocket: WebSocket, portfolio_id: str):
    """WebSocket for real-time portfolio updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Get latest portfolio data
            portfolio = trading_engine.get_portfolio(portfolio_id)
            if portfolio:
                stats = portfolio.get_stats()
                await websocket.send_json({
                    "type": "portfolio_update",
                    "data": {
                        "total_value": stats.total_value,
                        "total_pnl": stats.total_pnl,
                        "total_pnl_pct": stats.total_pnl_pct,
                        "positions": [pos.to_dict() for pos in portfolio.positions.values()]
                    },
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
