# AI PROPHET - AUTONOMOUS DOCKER CONTAINER
# =========================================
# 110% Protocol | FAANG Enterprise-Grade
#
# This Dockerfile creates a fully autonomous AI Prophet container
# that runs independently without requiring Manus execution.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cron \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs /app/data/day_trading /app/data/day_trading_cycles

# Make scripts executable
RUN chmod +x /app/autonomous_scheduler.py

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('/app/logs/autonomous_scheduler.log') else 1)"

# Run in daemon mode by default
CMD ["python3", "/app/autonomous_scheduler.py", "--mode", "daemon"]
