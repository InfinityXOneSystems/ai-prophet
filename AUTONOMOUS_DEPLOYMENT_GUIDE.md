# AI Prophet: Autonomous Deployment Guide

**Version:** 1.0  
**Date:** January 12, 2026  
**Author:** Manus AI

---

## 🎯 Overview

This guide provides complete instructions for deploying AI Prophet to run **autonomously and independently** without requiring Manus execution. The system will execute 2-hour trading cycles 24/7, focusing on priority trading windows, and automatically commit results to GitHub.

**Cost-Effective Solution:** No Manus execution costs - runs on your own infrastructure.

---

## 📋 Prerequisites

### Required
- Python 3.11+
- Git with GitHub authentication configured
- 1GB+ free disk space
- Internet connection

### Optional (for cloud deployment)
- Docker & Docker Compose
- Google Cloud SDK (for Cloud Run deployment)
- GCP project with billing enabled

### Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GCP_SA_KEY="your-gcp-service-account-key"
export OPENAI_API_KEY="your-openai-api-key"  # Optional
```

---

## 🚀 Deployment Options

Choose the deployment method that best fits your needs:

### Option 1: Cron Job (Recommended for Local/VPS)
Best for: Personal machines, VPS, or any Linux server

### Option 2: Systemd Service (Recommended for Servers)
Best for: Dedicated servers requiring persistent daemon mode

### Option 3: Docker Container (Recommended for Portability)
Best for: Containerized environments, easy deployment

### Option 4: Google Cloud Run (Recommended for 24/7 Cloud)
Best for: Fully managed, scalable, 24/7 cloud deployment

---

## 📦 Option 1: Cron Job Deployment

### Step 1: Clone Repository
```bash
cd ~
git clone https://github.com/InfinityXOneSystems/ai-prophet.git
cd ai-prophet
```

### Step 2: Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Step 3: Set Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export GEMINI_API_KEY="your-key-here"
export GCP_SA_KEY="your-key-here"
```

### Step 4: Run Setup Script
```bash
bash setup_cron.sh
```

### Step 5: Verify Installation
```bash
# Check cron jobs
crontab -l

# View logs
tail -f logs/cron.log
```

### Cron Schedule
- **Every 2 hours:** General trading cycle
- **9:30 AM (Mon-Fri):** Opening Bell priority window
- **3:00 PM (Mon-Fri):** Power Hour priority window

---

## 🔧 Option 2: Systemd Service Deployment

### Step 1: Clone Repository
```bash
cd ~
git clone https://github.com/InfinityXOneSystems/ai-prophet.git
cd ai-prophet
```

### Step 2: Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Step 3: Run Setup Script (as root)
```bash
sudo bash setup_systemd.sh
```

### Step 4: Verify Service
```bash
# Check status
sudo systemctl status ai-prophet

# View logs
sudo journalctl -u ai-prophet -f
```

### Service Management
```bash
# Start service
sudo systemctl start ai-prophet

# Stop service
sudo systemctl stop ai-prophet

# Restart service
sudo systemctl restart ai-prophet

# Enable auto-start on boot
sudo systemctl enable ai-prophet

# Disable auto-start
sudo systemctl disable ai-prophet
```

---

## 🐳 Option 3: Docker Deployment

### Step 1: Clone Repository
```bash
cd ~
git clone https://github.com/InfinityXOneSystems/ai-prophet.git
cd ai-prophet
```

### Step 2: Create .env File
```bash
cat > .env << EOF
GEMINI_API_KEY=your-key-here
GCP_SA_KEY=your-key-here
OPENAI_API_KEY=your-key-here
EOF
```

### Step 3: Build and Run
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using Docker directly
docker build -t ai-prophet .
docker run -d --name ai-prophet \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  ai-prophet
```

### Step 4: Manage Container
```bash
# View logs
docker logs -f ai-prophet

# Stop container
docker stop ai-prophet

# Start container
docker start ai-prophet

# Restart container
docker restart ai-prophet

# Remove container
docker rm -f ai-prophet
```

---

## ☁️ Option 4: Google Cloud Run Deployment

### Step 1: Prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project infinity-x-one-systems
```

### Step 2: Set Up Secrets
```bash
# Create secrets in Google Secret Manager
echo -n "your-gemini-key" | gcloud secrets create GEMINI_API_KEY --data-file=-
echo -n "your-gcp-key" | gcloud secrets create GCP_SA_KEY --data-file=-
```

### Step 3: Clone Repository
```bash
cd ~
git clone https://github.com/InfinityXOneSystems/ai-prophet.git
cd ai-prophet
```

### Step 4: Deploy to Cloud Run
```bash
bash deploy_cloud_run.sh
```

### Step 5: Verify Deployment
```bash
# View service
gcloud run services describe ai-prophet-autonomous --region us-central1

# View logs
gcloud run logs read ai-prophet-autonomous --region us-central1 --limit 50
```

### Cloud Run Features
- **Always Running:** Min instances = 1 (24/7 availability)
- **Auto-Scaling:** Scales up if needed
- **Managed Service:** No server maintenance
- **High Availability:** Built-in redundancy

---

## 🔍 Health Monitoring

### Manual Health Check
```bash
python3 health_monitor.py --check
```

### Continuous Monitoring
```bash
# Monitor every 15 minutes with auto-recovery
python3 health_monitor.py --monitor --interval 15
```

### Force Recovery
```bash
python3 health_monitor.py --recover
```

### Health Checks Performed
- ✅ Process running
- ✅ Log file activity (last 3 hours)
- ✅ Recent execution (last 3 hours)
- ✅ Disk space available (>1GB)

---

## 📊 Monitoring & Logs

### Log Files
```bash
# Autonomous scheduler log
tail -f logs/autonomous_scheduler.log

# Cron execution log
tail -f logs/cron.log

# Systemd service log
sudo journalctl -u ai-prophet -f

# Health monitor log
tail -f logs/health_monitor.log
```

### Data Files
```bash
# Latest trading state
cat data/day_trading/state_$(date +%Y%m%d).json

# Latest cycle results
ls -lt data/day_trading_cycles/ | head -5
```

### GitHub Commits
All cycle results are automatically committed to GitHub with timestamps.

---

## 🎯 Priority Trading Windows

The system prioritizes execution during these high-value windows:

| Window | Time (EST) | Days | Priority |
|--------|-----------|------|----------|
| **Opening Bell** | 9:30-10:30 AM | Mon-Fri | 🔴 CRITICAL |
| **Power Hour** | 3:00-4:00 PM | Mon-Fri | 🟠 HIGH |
| **Crypto US Session** | 8:00 AM - 5:00 PM | Daily | 🟠 HIGH |
| **Crypto 24/7** | All times | Daily | 🟡 MEDIUM |

---

## 🔧 Troubleshooting

### Issue: Process Not Running
```bash
# Check if process exists
pgrep -f autonomous_scheduler.py

# Force recovery
python3 health_monitor.py --recover
```

### Issue: No Recent Executions
```bash
# Check logs for errors
tail -100 logs/autonomous_scheduler.log

# Manually run single cycle
python3 run_day_trading.py --cycles 1 --capital 100000
```

### Issue: GitHub Push Failures
```bash
# Verify Git authentication
git config --list | grep user

# Test push
cd ~/ai-prophet
git push origin main
```

### Issue: Out of Memory
```bash
# Check memory usage
free -h

# Restart service
sudo systemctl restart ai-prophet  # For systemd
docker restart ai-prophet          # For Docker
```

---

## 💰 Cost Comparison

### Manus Execution (Current)
- **Cost:** ~$0.10-0.50 per execution
- **Daily Cost:** $1.20-6.00 (12 cycles)
- **Monthly Cost:** $36-180

### Autonomous Deployment (Recommended)
- **Local/VPS:** $0 (uses existing infrastructure)
- **Cloud Run:** ~$10-30/month (always-on instance)
- **Savings:** 80-100% cost reduction

---

## 🔐 Security Best Practices

1. **Environment Variables:** Never commit API keys to Git
2. **Secret Management:** Use Google Secret Manager for cloud deployments
3. **File Permissions:** Restrict access to config files
4. **Network Security:** Use firewall rules for cloud deployments
5. **Regular Updates:** Keep dependencies updated

---

## 📈 Performance Optimization

### For Local/VPS Deployment
- Use SSD storage for faster I/O
- Ensure stable internet connection
- Monitor disk space regularly

### For Cloud Deployment
- Use appropriate instance size (2GB RAM, 2 vCPU)
- Enable auto-scaling for peak times
- Use regional deployment close to data sources

---

## 🆘 Support & Maintenance

### Regular Maintenance
- **Weekly:** Review logs for errors
- **Monthly:** Update dependencies
- **Quarterly:** Review and optimize trading strategies

### Getting Help
- Check logs first: `logs/autonomous_scheduler.log`
- Run health check: `python3 health_monitor.py --check`
- Review GitHub commits for execution history

---

## 📝 Quick Reference

### Start Autonomous Mode
```bash
# Cron (automatic)
bash setup_cron.sh

# Systemd
sudo systemctl start ai-prophet

# Docker
docker-compose up -d

# Manual daemon
python3 autonomous_scheduler.py --mode daemon
```

### Stop Autonomous Mode
```bash
# Cron
crontab -e  # Remove AI Prophet lines

# Systemd
sudo systemctl stop ai-prophet

# Docker
docker-compose down

# Manual
pkill -f autonomous_scheduler.py
```

### View Status
```bash
# Health check
python3 health_monitor.py --check

# View logs
tail -f logs/autonomous_scheduler.log

# Check process
pgrep -f autonomous_scheduler.py
```

---

## ✅ Post-Deployment Checklist

- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Environment variables configured
- [ ] Deployment method chosen and executed
- [ ] Service running and verified
- [ ] Logs accessible and monitoring
- [ ] Health check passing
- [ ] First execution completed
- [ ] GitHub auto-commit working
- [ ] Backup/recovery plan in place

---

## 🎉 Success Criteria

Your autonomous deployment is successful when:

1. ✅ Process runs continuously without manual intervention
2. ✅ Trading cycles execute every 2 hours
3. ✅ Priority windows are prioritized correctly
4. ✅ Results automatically commit to GitHub
5. ✅ Health checks pass consistently
6. ✅ No Manus execution costs incurred

---

**Congratulations! AI Prophet is now running autonomously. 🚀**

*Zero human hands. Maximum efficiency. Cost-effective operation.*
