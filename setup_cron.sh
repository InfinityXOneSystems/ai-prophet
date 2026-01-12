#!/bin/bash
#
# AI PROPHET - CRON SETUP SCRIPT
# ===============================
# Sets up cron jobs for autonomous AI Prophet execution
#
# Usage: bash setup_cron.sh
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH=$(which python3)

echo "============================================"
echo "AI PROPHET - CRON SETUP"
echo "============================================"
echo ""
echo "Script Directory: $SCRIPT_DIR"
echo "Python Path: $PYTHON_PATH"
echo ""

# Create cron job entries
CRON_ENTRY_2H="0 */2 * * * cd $SCRIPT_DIR && $PYTHON_PATH $SCRIPT_DIR/autonomous_scheduler.py --mode cron >> $SCRIPT_DIR/logs/cron.log 2>&1"
CRON_ENTRY_OPENING="30 9 * * 1-5 cd $SCRIPT_DIR && $PYTHON_PATH $SCRIPT_DIR/autonomous_scheduler.py --mode priority >> $SCRIPT_DIR/logs/cron_priority.log 2>&1"
CRON_ENTRY_POWER="0 15 * * 1-5 cd $SCRIPT_DIR && $PYTHON_PATH $SCRIPT_DIR/autonomous_scheduler.py --mode priority >> $SCRIPT_DIR/logs/cron_priority.log 2>&1"

# Backup existing crontab
echo "Backing up existing crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

# Check if AI Prophet cron jobs already exist
if crontab -l 2>/dev/null | grep -q "autonomous_scheduler.py"; then
    echo "⚠️  AI Prophet cron jobs already exist!"
    echo ""
    echo "Current AI Prophet cron jobs:"
    crontab -l | grep "autonomous_scheduler.py"
    echo ""
    read -p "Do you want to replace them? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Remove existing AI Prophet cron jobs
    crontab -l | grep -v "autonomous_scheduler.py" | crontab -
fi

# Add new cron jobs
echo "Adding AI Prophet cron jobs..."
(crontab -l 2>/dev/null; echo "# AI Prophet - 2-Hour Cycle"; echo "$CRON_ENTRY_2H") | crontab -
(crontab -l 2>/dev/null; echo "# AI Prophet - Opening Bell Priority"; echo "$CRON_ENTRY_OPENING") | crontab -
(crontab -l 2>/dev/null; echo "# AI Prophet - Power Hour Priority"; echo "$CRON_ENTRY_POWER") | crontab -

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "Installed cron jobs:"
echo "-------------------"
crontab -l | grep -A 1 "AI Prophet"
echo ""
echo "Schedule:"
echo "  - Every 2 hours: General trading cycle"
echo "  - 9:30 AM (Mon-Fri): Opening Bell priority"
echo "  - 3:00 PM (Mon-Fri): Power Hour priority"
echo ""
echo "Logs will be written to:"
echo "  - $SCRIPT_DIR/logs/cron.log"
echo "  - $SCRIPT_DIR/logs/cron_priority.log"
echo ""
echo "To view cron jobs: crontab -l"
echo "To remove cron jobs: crontab -e (then delete AI Prophet lines)"
echo ""
echo "============================================"
