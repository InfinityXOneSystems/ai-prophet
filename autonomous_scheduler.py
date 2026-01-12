#!/usr/bin/env python3
"""
AI PROPHET - AUTONOMOUS SCHEDULER
==================================
110% Protocol | FAANG Enterprise-Grade | Zero Human Hands

This script runs AI Prophet autonomously without requiring Manus execution.
It handles:
- 2-hour cycle execution
- Priority trading windows (Opening Bell, Power Hour)
- 24/7 crypto monitoring
- Auto-recovery from failures
- Health monitoring
- GitHub auto-commit

Usage:
    python3 autonomous_scheduler.py --mode daemon    # Run as background daemon
    python3 autonomous_scheduler.py --mode cron      # Run single cycle (for cron)
    python3 autonomous_scheduler.py --mode priority  # Run priority windows only
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'autonomous_scheduler.log')
    ]
)
logger = logging.getLogger('AutonomousScheduler')


class TradingWindow:
    """Trading window definition"""
    def __init__(self, name: str, start_hour: int, start_minute: int, 
                 end_hour: int, end_minute: int, priority: str, days: list = None):
        self.name = name
        self.start_hour = start_hour
        self.start_minute = start_minute
        self.end_hour = end_hour
        self.end_minute = end_minute
        self.priority = priority
        self.days = days or [0, 1, 2, 3, 4]  # Mon-Fri by default
    
    def is_active(self, dt: datetime) -> bool:
        """Check if window is currently active"""
        if dt.weekday() not in self.days:
            return False
        
        current_time = dt.hour * 60 + dt.minute
        start_time = self.start_hour * 60 + self.start_minute
        end_time = self.end_hour * 60 + self.end_minute
        
        return start_time <= current_time < end_time


# Define priority trading windows (EST)
TRADING_WINDOWS = [
    TradingWindow("Opening Bell", 9, 30, 10, 30, "CRITICAL", [0, 1, 2, 3, 4]),
    TradingWindow("Power Hour", 15, 0, 16, 0, "HIGH", [0, 1, 2, 3, 4]),
    TradingWindow("Crypto US Session", 8, 0, 17, 0, "HIGH", [0, 1, 2, 3, 4, 5, 6]),
]


class AIAutonomousScheduler:
    """Autonomous scheduler for AI Prophet"""
    
    def __init__(self, mode: str = "daemon"):
        self.mode = mode
        self.running = True
        self.repo_path = Path(__file__).parent
        self.last_execution = None
        self.execution_count = 0
        self.error_count = 0
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def get_current_priority_window(self) -> Optional[TradingWindow]:
        """Get the current priority trading window if any"""
        now = datetime.now()
        for window in TRADING_WINDOWS:
            if window.is_active(now):
                return window
        return None
    
    def should_execute_cycle(self) -> tuple[bool, str]:
        """Determine if a cycle should be executed now"""
        now = datetime.now()
        
        # Check if we're in a priority window
        priority_window = self.get_current_priority_window()
        if priority_window:
            reason = f"Priority window: {priority_window.name} ({priority_window.priority})"
            return True, reason
        
        # Check if 2 hours have passed since last execution
        if self.last_execution:
            time_since_last = now - self.last_execution
            if time_since_last < timedelta(hours=2):
                return False, f"Last execution {time_since_last.total_seconds()/60:.1f} minutes ago"
        
        # Crypto runs 24/7, execute every 2 hours
        return True, "2-hour cycle interval"
    
    def execute_trading_cycle(self) -> bool:
        """Execute a single trading cycle"""
        logger.info("=" * 80)
        logger.info("EXECUTING AI PROPHET TRADING CYCLE")
        logger.info("=" * 80)
        
        try:
            # Run the day trading script
            cmd = [
                sys.executable,
                str(self.repo_path / "run_day_trading.py"),
                "--cycles", "1",
                "--capital", "100000"
            ]
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Log output
            if result.stdout:
                logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr}")
            
            if result.returncode == 0:
                logger.info("✅ Trading cycle completed successfully")
                self.last_execution = datetime.now()
                self.execution_count += 1
                self.error_count = 0  # Reset error count on success
                
                # Auto-commit to GitHub
                self.auto_commit_results()
                
                return True
            else:
                logger.error(f"❌ Trading cycle failed with return code {result.returncode}")
                self.error_count += 1
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Trading cycle timed out after 10 minutes")
            self.error_count += 1
            return False
        except Exception as e:
            logger.error(f"❌ Error executing trading cycle: {e}")
            self.error_count += 1
            return False
    
    def auto_commit_results(self):
        """Automatically commit results to GitHub"""
        try:
            logger.info("Committing results to GitHub...")
            
            # Check for new files
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                logger.info("No changes to commit")
                return
            
            # Add new data files
            subprocess.run(
                ["git", "add", "data/day_trading/", "data/day_trading_cycles/"],
                cwd=str(self.repo_path),
                check=True
            )
            
            # Commit with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"AI Prophet Autonomous Cycle - {timestamp} | Execution #{self.execution_count}"
            
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(self.repo_path),
                check=True
            )
            
            # Push to remote
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=str(self.repo_path),
                check=True
            )
            
            logger.info("✅ Results committed and pushed to GitHub")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to commit to GitHub: {e}")
        except Exception as e:
            logger.warning(f"Error during auto-commit: {e}")
    
    def run_daemon_mode(self):
        """Run in daemon mode with continuous monitoring"""
        logger.info("🚀 AI PROPHET AUTONOMOUS SCHEDULER - DAEMON MODE")
        logger.info("=" * 80)
        logger.info("Priority Windows:")
        for window in TRADING_WINDOWS:
            logger.info(f"  - {window.name}: {window.start_hour:02d}:{window.start_minute:02d}-"
                       f"{window.end_hour:02d}:{window.end_minute:02d} ({window.priority})")
        logger.info("=" * 80)
        
        while self.running:
            try:
                should_run, reason = self.should_execute_cycle()
                
                if should_run:
                    logger.info(f"📊 Executing cycle: {reason}")
                    success = self.execute_trading_cycle()
                    
                    if not success and self.error_count >= 3:
                        logger.error("⚠️ 3 consecutive failures detected. Waiting 30 minutes before retry...")
                        time.sleep(1800)  # Wait 30 minutes
                        self.error_count = 0  # Reset error count
                else:
                    logger.debug(f"⏸️  Skipping cycle: {reason}")
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in daemon loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
        
        logger.info("🛑 AI Prophet Autonomous Scheduler stopped")
    
    def run_cron_mode(self):
        """Run a single cycle (for cron job execution)"""
        logger.info("🚀 AI PROPHET AUTONOMOUS SCHEDULER - CRON MODE")
        
        should_run, reason = self.should_execute_cycle()
        
        if should_run:
            logger.info(f"📊 Executing cycle: {reason}")
            self.execute_trading_cycle()
        else:
            logger.info(f"⏸️  Skipping cycle: {reason}")
    
    def run_priority_mode(self):
        """Run only during priority windows"""
        logger.info("🚀 AI PROPHET AUTONOMOUS SCHEDULER - PRIORITY MODE")
        
        priority_window = self.get_current_priority_window()
        
        if priority_window:
            logger.info(f"📊 Executing priority window: {priority_window.name}")
            self.execute_trading_cycle()
        else:
            logger.info("⏸️  No priority window active")


def main():
    parser = argparse.ArgumentParser(description="AI Prophet Autonomous Scheduler")
    parser.add_argument(
        "--mode",
        choices=["daemon", "cron", "priority"],
        default="daemon",
        help="Execution mode: daemon (continuous), cron (single run), priority (priority windows only)"
    )
    
    args = parser.parse_args()
    
    scheduler = AIAutonomousScheduler(mode=args.mode)
    
    if args.mode == "daemon":
        scheduler.run_daemon_mode()
    elif args.mode == "cron":
        scheduler.run_cron_mode()
    elif args.mode == "priority":
        scheduler.run_priority_mode()


if __name__ == "__main__":
    main()
