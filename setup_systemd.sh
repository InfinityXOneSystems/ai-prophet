#!/bin/bash
#
# AI PROPHET - SYSTEMD SERVICE SETUP
# ===================================
# Sets up AI Prophet as a systemd service for persistent daemon mode
#
# Usage: sudo bash setup_systemd.sh
#

set -e

if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_FILE="$SCRIPT_DIR/ai-prophet.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "============================================"
echo "AI PROPHET - SYSTEMD SERVICE SETUP"
echo "============================================"
echo ""
echo "Script Directory: $SCRIPT_DIR"
echo "Service File: $SERVICE_FILE"
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Service file not found: $SERVICE_FILE"
    exit 1
fi

# Stop existing service if running
if systemctl is-active --quiet ai-prophet.service; then
    echo "Stopping existing AI Prophet service..."
    systemctl stop ai-prophet.service
fi

# Copy service file to systemd directory
echo "Installing service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/ai-prophet.service"
chmod 644 "$SYSTEMD_DIR/ai-prophet.service"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service to start on boot
echo "Enabling service to start on boot..."
systemctl enable ai-prophet.service

# Start the service
echo "Starting AI Prophet service..."
systemctl start ai-prophet.service

# Wait a moment for service to start
sleep 2

# Check service status
echo ""
echo "============================================"
echo "SERVICE STATUS"
echo "============================================"
systemctl status ai-prophet.service --no-pager || true

echo ""
echo "✅ AI Prophet systemd service installed successfully!"
echo ""
echo "Useful commands:"
echo "  - Check status:  sudo systemctl status ai-prophet"
echo "  - View logs:     sudo journalctl -u ai-prophet -f"
echo "  - Stop service:  sudo systemctl stop ai-prophet"
echo "  - Start service: sudo systemctl start ai-prophet"
echo "  - Restart:       sudo systemctl restart ai-prophet"
echo "  - Disable:       sudo systemctl disable ai-prophet"
echo ""
echo "Log files:"
echo "  - $SCRIPT_DIR/logs/systemd.log"
echo "  - $SCRIPT_DIR/logs/systemd_error.log"
echo "  - $SCRIPT_DIR/logs/autonomous_scheduler.log"
echo ""
echo "============================================"
