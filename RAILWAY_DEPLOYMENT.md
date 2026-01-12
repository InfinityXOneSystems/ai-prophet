# AI PROPHET - RAILWAY DEPLOYMENT GUIDE

**Cost:** $0-5/month (covered by Railway free tier)  
**Setup Time:** 5 minutes  
**Difficulty:** Easy

---

## 🚀 Quick Deploy to Railway

### Option 1: Deploy via Railway Dashboard (Recommended)

**Step 1: Create New Project**
1. Go to https://railway.app/dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose: `InfinityXOneSystems/prophet-system`
5. Click **"Deploy Now"**

**Step 2: Configure Environment Variables**

Add these environment variables in Railway dashboard:

```
GEMINI_API_KEY=<your_gemini_api_key>
GCP_SA_KEY=<your_gcp_service_account_json>
PORT=8080
```

**Step 3: Deploy**
- Railway will automatically detect the Dockerfile
- Build and deploy will start automatically
- Service will be live in ~3-5 minutes

**Step 4: Get Public URL**
- Go to **Settings** → **Networking**
- Click **"Generate Domain"**
- Your service will be available at: `https://your-app.up.railway.app`

---

### Option 2: Deploy via Railway CLI

**Step 1: Install Railway CLI**
```bash
npm install -g @railway/cli
```

**Step 2: Login**
```bash
railway login
```

**Step 3: Initialize Project**
```bash
cd /path/to/ai-prophet
railway init
```

**Step 4: Add Environment Variables**
```bash
railway variables set GEMINI_API_KEY="your_key_here"
railway variables set GCP_SA_KEY='{"type":"service_account",...}'
railway variables set PORT=8080
```

**Step 5: Deploy**
```bash
railway up
```

**Step 6: Get URL**
```bash
railway domain
```

---

## 🔧 Configuration Details

### Dockerfile
Railway uses the existing `Dockerfile` which:
- Runs Flask web server on port 8080
- Starts autonomous scheduler in background
- Includes health checks
- Auto-recovers on failures

### Environment Variables Required

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `GCP_SA_KEY` | GCP service account JSON | `{"type":"service_account",...}` |
| `PORT` | Web server port | `8080` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CAPITAL` | Trading capital | `100000` |
| `CYCLE_INTERVAL` | Hours between cycles | `2` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 💰 Railway Pricing

### Free Tier (Hobby Plan)
- **$5 free credit/month**
- 512 MB RAM
- Shared CPU
- 100 GB bandwidth
- Custom domains

### Estimated Usage
- **CPU:** Low (periodic tasks)
- **Memory:** ~256-512 MB
- **Bandwidth:** Minimal
- **Storage:** Minimal

**Expected Cost:** $0-5/month (covered by free tier)

---

## 📊 Monitoring

### Railway Dashboard
- **Logs:** Real-time logs in dashboard
- **Metrics:** CPU, memory, bandwidth usage
- **Deployments:** View deployment history
- **Health:** Service health status

### API Endpoints
- **Health:** `https://your-app.up.railway.app/health`
- **Status:** `https://your-app.up.railway.app/status`
- **Logs:** `https://your-app.up.railway.app/logs`

---

## 🔄 Auto-Deploy from GitHub

Railway can automatically deploy when you push to GitHub:

**Step 1: Enable Auto-Deploy**
1. Go to project settings
2. Enable **"Auto-Deploy"**
3. Select branch: `main`

**Step 2: Push Changes**
```bash
git push origin main
```

Railway will automatically:
- Detect changes
- Build new image
- Deploy with zero downtime
- Rollback on failure

---

## 🎯 Advantages of Railway

✅ **Simpler than Cloud Run** - No complex IAM/secrets setup  
✅ **Free tier** - $5/month covers AI Prophet completely  
✅ **Auto-deploy** - Push to GitHub = automatic deployment  
✅ **Better DX** - Cleaner dashboard, easier logs  
✅ **No cold starts** - Always warm (unlike Cloud Run)  
✅ **Custom domains** - Free SSL certificates  
✅ **Zero config** - Detects Dockerfile automatically  

---

## 🚨 Troubleshooting

### Service Won't Start
- Check environment variables are set correctly
- Verify `PORT=8080` is set
- Check Railway logs for errors

### Out of Memory
- Upgrade to Pro plan ($5/month for 8 GB RAM)
- Or optimize memory usage in code

### Deployment Failed
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Check Railway build logs

---

## 📈 Migration from Cloud Run

If migrating from Cloud Run:

1. **Keep Cloud Run running** (for now)
2. **Deploy to Railway** (following steps above)
3. **Test Railway deployment** (verify trades execute)
4. **Switch over** (update any external references)
5. **Shut down Cloud Run** (save costs)

No downtime required!

---

## 🎉 Result

After deployment, AI Prophet will:
- Run 24/7 on Railway
- Execute trades every 2 hours
- Auto-commit to GitHub
- Cost $0-5/month
- Zero human intervention

**Railway is perfect for AI Prophet - simple, cheap, and reliable.**

---

## 📞 Support

- **Railway Docs:** https://docs.railway.app
- **Railway Discord:** https://discord.gg/railway
- **Railway Status:** https://status.railway.app

---

*Last Updated: January 12, 2026*
