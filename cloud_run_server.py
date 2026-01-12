#!/usr/bin/env python3
"""
AI PROPHET - CLOUD RUN WEB SERVER
==================================
Lightweight web server that keeps Cloud Run container alive
while autonomous scheduler runs in the background.
"""

import os
import subprocess
import threading
import time
from datetime import datetime
from flask import Flask, jsonify
from pathlib import Path

app = Flask(__name__)

# Global state
scheduler_process = None
scheduler_status = {"status": "starting", "last_check": None, "uptime_seconds": 0}
start_time = time.time()

def run_autonomous_scheduler():
    """Run autonomous scheduler in background thread"""
    global scheduler_process, scheduler_status
    
    try:
        print("🚀 Starting autonomous scheduler...")
        scheduler_status["status"] = "running"
        scheduler_status["last_check"] = datetime.now().isoformat()
        
        # Run autonomous scheduler
        scheduler_process = subprocess.Popen(
            ["python3", "/app/autonomous_scheduler.py", "--mode", "daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor scheduler
        while True:
            if scheduler_process.poll() is not None:
                # Process ended, restart it
                print("⚠️  Scheduler stopped, restarting...")
                scheduler_status["status"] = "restarting"
                time.sleep(5)
                
                scheduler_process = subprocess.Popen(
                    ["python3", "/app/autonomous_scheduler.py", "--mode", "daemon"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                scheduler_status["status"] = "running"
            
            scheduler_status["last_check"] = datetime.now().isoformat()
            scheduler_status["uptime_seconds"] = int(time.time() - start_time)
            time.sleep(60)  # Check every minute
            
    except Exception as e:
        print(f"❌ Error in scheduler thread: {e}")
        scheduler_status["status"] = "error"
        scheduler_status["error"] = str(e)

@app.route('/')
def index():
    """Root endpoint - health check"""
    return jsonify({
        "service": "AI Prophet Autonomous",
        "status": "healthy",
        "scheduler_status": scheduler_status,
        "uptime_seconds": int(time.time() - start_time),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Health check endpoint for Cloud Run"""
    is_healthy = scheduler_status["status"] in ["running", "starting"]
    
    return jsonify({
        "status": "healthy" if is_healthy else "unhealthy",
        "scheduler": scheduler_status,
        "uptime_seconds": int(time.time() - start_time)
    }), 200 if is_healthy else 503

@app.route('/status')
def status():
    """Detailed status endpoint"""
    # Check for recent logs
    log_file = Path("/app/logs/autonomous_scheduler.log")
    last_log_time = None
    log_size = 0
    
    if log_file.exists():
        stat = log_file.stat()
        last_log_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
        log_size = stat.st_size
    
    # Check for recent trading cycles
    cycle_dir = Path("/app/data/day_trading_cycles")
    cycle_count = 0
    if cycle_dir.exists():
        cycle_count = len(list(cycle_dir.glob("*.json")))
    
    return jsonify({
        "service": "AI Prophet Autonomous",
        "scheduler": scheduler_status,
        "uptime_seconds": int(time.time() - start_time),
        "logs": {
            "last_updated": last_log_time,
            "size_bytes": log_size
        },
        "trading_cycles": {
            "total_count": cycle_count
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/logs')
def logs():
    """Get recent log entries"""
    log_file = Path("/app/logs/autonomous_scheduler.log")
    
    if not log_file.exists():
        return jsonify({"error": "Log file not found"}), 404
    
    try:
        # Get last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-50:] if len(lines) > 50 else lines
        
        return jsonify({
            "log_file": str(log_file),
            "total_lines": len(lines),
            "recent_lines": recent_lines,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("AI PROPHET - CLOUD RUN SERVER")
    print("="*80)
    print("Starting autonomous scheduler in background...")
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_autonomous_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Give scheduler a moment to start
    time.sleep(2)
    
    # Start Flask web server
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting web server on port {port}...")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=False)
