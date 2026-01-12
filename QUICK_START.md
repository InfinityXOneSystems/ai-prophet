# AI Prophet: Quick Start Guide

**Get AI Prophet running autonomously in under 5 minutes!**

---

## 🚀 Fastest Path to Autonomous Operation

### Step 1: Clone Repository (30 seconds)
```bash
cd ~
git clone https://github.com/InfinityXOneSystems/prophet-system.git ai-prophet
cd ai-prophet
```

### Step 2: Install Dependencies (2 minutes)
```bash
pip3 install -r requirements.txt
```

### Step 3: Set Environment Variables (1 minute)
```bash
# Add to ~/.bashrc or ~/.zshrc
export GEMINI_API_KEY="your-gemini-api-key-here"
export GCP_SA_KEY="your-gcp-service-account-key-here"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Step 4: Choose Deployment Method (1 minute)

#### Option A: Cron Job (Recommended for most users)
```bash
bash setup_cron.sh
```
✅ Done! AI Prophet will run every 2 hours automatically.

#### Option B: Systemd Service (For servers)
```bash
sudo bash setup_systemd.sh
```
✅ Done! AI Prophet runs as a persistent daemon.

#### Option C: Docker (For containerized environments)
```bash
# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your-key-here
GCP_SA_KEY=your-key-here
EOF

# Start container
docker-compose up -d
```
✅ Done! AI Prophet runs in a Docker container.

---

## ✅ Verify It's Working

### Check Process
```bash
# For cron
crontab -l | grep autonomous_scheduler

# For systemd
sudo systemctl status ai-prophet

# For Docker
docker ps | grep ai-prophet
```

### View Logs
```bash
tail -f logs/autonomous_scheduler.log
```

### Health Check
```bash
python3 health_monitor.py --check
```

---

## 📊 What Happens Next?

1. **Every 2 hours:** AI Prophet executes a trading cycle
2. **Priority windows:** Opening Bell (9:30 AM) and Power Hour (3:00 PM) get priority
3. **Auto-commit:** Results automatically pushed to GitHub
4. **24/7 operation:** Crypto markets monitored continuously
5. **Zero cost:** No Manus execution fees

---

## 💡 Pro Tips

### View Recent Execution
```bash
cat data/day_trading/state_$(date +%Y%m%d).json | jq
```

### Manual Test Run
```bash
python3 run_day_trading.py --cycles 1 --capital 100000
```

### Stop Autonomous Mode
```bash
# Cron: Remove from crontab
crontab -e

# Systemd: Stop service
sudo systemctl stop ai-prophet

# Docker: Stop container
docker-compose down
```

---

## 🆘 Troubleshooting

**Problem:** Process not running  
**Solution:** `python3 health_monitor.py --recover`

**Problem:** No logs appearing  
**Solution:** Check environment variables are set: `echo $GEMINI_API_KEY`

**Problem:** GitHub push fails  
**Solution:** Configure Git credentials: `git config --global user.name "Your Name"`

---

## 📚 Full Documentation

For advanced configuration, cloud deployment, and detailed guides:
- **Full Guide:** [AUTONOMOUS_DEPLOYMENT_GUIDE.md](AUTONOMOUS_DEPLOYMENT_GUIDE.md)
- **Repository:** https://github.com/InfinityXOneSystems/prophet-system

---

## 🎉 You're Done!

AI Prophet is now running autonomously. No more Manus costs!

**Check back in 2 hours to see your first autonomous execution results.**

---

*110% Protocol | FAANG Enterprise-Grade | Zero Human Hands*
