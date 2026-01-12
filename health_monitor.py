#!/usr/bin/env python3
"""
AI PROPHET - HEALTH MONITORING & AUTO-RECOVERY
===============================================
110% Protocol | FAANG Enterprise-Grade

Monitors AI Prophet system health and performs auto-recovery actions:
- Checks if scheduler is running
- Monitors log file growth
- Validates recent execution
- Auto-restarts on failure
- Sends alerts (optional)

Usage:
    python3 health_monitor.py --check          # One-time health check
    python3 health_monitor.py --monitor        # Continuous monitoring
    python3 health_monitor.py --recover        # Force recovery
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'health_monitor.log')
    ]
)
logger = logging.getLogger('HealthMonitor')


class HealthStatus:
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AIHealthMonitor:
    """Health monitoring and auto-recovery for AI Prophet"""
    
    def __init__(self):
        self.repo_path = Path(__file__).parent
        self.log_dir = self.repo_path / "logs"
        self.data_dir = self.repo_path / "data"
        self.scheduler_log = self.log_dir / "autonomous_scheduler.log"
        self.state_dir = self.data_dir / "day_trading"
    
    def check_process_running(self) -> bool:
        """Check if autonomous scheduler is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "autonomous_scheduler.py"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking process: {e}")
            return False
    
    def check_log_activity(self, hours: int = 3) -> bool:
        """Check if log file has recent activity"""
        if not self.scheduler_log.exists():
            logger.warning("Scheduler log file does not exist")
            return False
        
        try:
            # Check file modification time
            mtime = datetime.fromtimestamp(self.scheduler_log.stat().st_mtime)
            age = datetime.now() - mtime
            
            if age > timedelta(hours=hours):
                logger.warning(f"Log file not updated in {age.total_seconds()/3600:.1f} hours")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking log activity: {e}")
            return False
    
    def check_recent_execution(self, hours: int = 3) -> bool:
        """Check if there was a recent execution"""
        try:
            # Find most recent state file
            if not self.state_dir.exists():
                logger.warning("State directory does not exist")
                return False
            
            state_files = list(self.state_dir.glob("state_*.json"))
            if not state_files:
                logger.warning("No state files found")
                return False
            
            # Get most recent state file
            latest_state = max(state_files, key=lambda p: p.stat().st_mtime)
            
            # Check age
            mtime = datetime.fromtimestamp(latest_state.stat().st_mtime)
            age = datetime.now() - mtime
            
            if age > timedelta(hours=hours):
                logger.warning(f"Last execution was {age.total_seconds()/3600:.1f} hours ago")
                return False
            
            # Validate state file content
            with open(latest_state, 'r') as f:
                state = json.load(f)
                if 'portfolio_value' not in state:
                    logger.warning("Invalid state file format")
                    return False
            
            logger.info(f"Recent execution found: {latest_state.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking recent execution: {e}")
            return False
    
    def check_disk_space(self, min_gb: float = 1.0) -> bool:
        """Check if sufficient disk space is available"""
        try:
            stat = os.statvfs(str(self.repo_path))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            if free_gb < min_gb:
                logger.warning(f"Low disk space: {free_gb:.2f} GB available")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False
    
    def perform_health_check(self) -> Dict:
        """Perform comprehensive health check"""
        logger.info("=" * 80)
        logger.info("AI PROPHET HEALTH CHECK")
        logger.info("=" * 80)
        
        checks = {
            "process_running": self.check_process_running(),
            "log_activity": self.check_log_activity(hours=3),
            "recent_execution": self.check_recent_execution(hours=3),
            "disk_space": self.check_disk_space(min_gb=1.0),
        }
        
        # Determine overall status
        if all(checks.values()):
            status = HealthStatus.HEALTHY
        elif checks["process_running"] and checks["recent_execution"]:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "checks": checks
        }
        
        # Log results
        logger.info(f"Status: {status}")
        for check, passed in checks.items():
            symbol = "✅" if passed else "❌"
            logger.info(f"  {symbol} {check}: {'PASS' if passed else 'FAIL'}")
        
        return result
    
    def auto_recover(self) -> bool:
        """Attempt to auto-recover the system"""
        logger.info("=" * 80)
        logger.info("ATTEMPTING AUTO-RECOVERY")
        logger.info("=" * 80)
        
        try:
            # Kill any existing processes
            logger.info("Stopping existing processes...")
            subprocess.run(
                ["pkill", "-f", "autonomous_scheduler.py"],
                capture_output=True
            )
            time.sleep(2)
            
            # Start new process
            logger.info("Starting autonomous scheduler...")
            subprocess.Popen(
                [sys.executable, str(self.repo_path / "autonomous_scheduler.py"), "--mode", "daemon"],
                cwd=str(self.repo_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait and verify
            time.sleep(5)
            if self.check_process_running():
                logger.info("✅ Auto-recovery successful")
                return True
            else:
                logger.error("❌ Auto-recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Auto-recovery error: {e}")
            return False
    
    def monitor_continuous(self, interval_minutes: int = 15):
        """Continuous monitoring with auto-recovery"""
        logger.info("🚀 Starting continuous health monitoring")
        logger.info(f"Check interval: {interval_minutes} minutes")
        logger.info("=" * 80)
        
        consecutive_failures = 0
        max_failures = 3
        
        while True:
            try:
                result = self.perform_health_check()
                
                if result["status"] == HealthStatus.CRITICAL:
                    consecutive_failures += 1
                    logger.warning(f"⚠️  Critical status detected ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Maximum consecutive failures reached. Attempting recovery...")
                        if self.auto_recover():
                            consecutive_failures = 0
                        else:
                            logger.error("Recovery failed. Manual intervention required.")
                else:
                    consecutive_failures = 0
                
                # Sleep until next check
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="AI Prophet Health Monitor")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Perform one-time health check"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuous monitoring with auto-recovery"
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Force auto-recovery"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Monitoring interval in minutes (default: 15)"
    )
    
    args = parser.parse_args()
    
    monitor = AIHealthMonitor()
    
    if args.recover:
        monitor.auto_recover()
    elif args.monitor:
        monitor.monitor_continuous(interval_minutes=args.interval)
    else:
        # Default to single health check
        result = monitor.perform_health_check()
        sys.exit(0 if result["status"] == HealthStatus.HEALTHY else 1)


if __name__ == "__main__":
    main()
