# Deploy CAPE to Railway

## Prerequisites
- Railway CLI: `npm install -g @railway/cli`
- Railway account linked: `railway login`

## Step 1: Prepare Database (Railway MySQL)

1. Create new Railway project
2. Add MySQL plugin → generates `MYSQLHOST`, `MYSQLPORT`, `MYSQL_DATABASE`, `MYSQLUSER`, `MYSQLPASSWORD`
3. Note the connection details for ESP32 config later

## Step 2: Push to GitHub

```bash
cd Detect
git add .
git commit -m "feat: CAPE with Railway deployment config"
git push origin main
```

## Step 3: Connect Railway to GitHub

1. Go to railway.app
2. New Project → Deploy from GitHub repo
3. Select your CAPE repo
4. Railway auto-detects `railway.json`

## Step 4: Configure Environment Variables

In Railway dashboard → Variables:

```env
# Database (from Railway MySQL plugin)
DB__HOST=<MYSQLHOST>
DB__PORT=<MYSQLPORT>
DB__NAME=<MYSQL_DATABASE>
DB__USER=<MYSQLUSER>
DB__PASSWORD=<MYSQLPASSWORD>

# App Settings
THRESHOLD=0.5
DEVICE=cpu
MODEL_DIR=models
ZONES_PATH=zones.json

# CORS (your frontend URL)
ALLOWED_ORIGINS=http://your-frontend.railway.app

# Optional
LOG_LEVEL=INFO
```

## Step 5: Upload Models & Zones

Railway persistent disks (`/data`):

```bash
railway run -- mkdir -p /app/models /app/data
# Then upload via Railway dashboard → Variables → File reference
```

Or in Dockerfile, volumes mounted from Railway disks.

## Step 6: ESP32 Configuration

Update ESP32 sketch:

```cpp
// Old (PHP hosting):
http.begin("http://your-old-host.com/upload_img.php");

// New (Railway):
http.begin("http://cape-api.railway.app/receive");
```

## Step 7: Deploy

Railway auto-deploys on git push. Manual trigger:
```bash
railway up
```

## Health Check

```bash
curl https://your-app.railway.app/health
```

Expected response:
```json
{"status":"ok","device":"cpu","models_loaded":["angle_1","angle_2"]}
```

## Cost Estimate

| Resource | Railway Starter |
|---|---|
| API container | $5/mo (always-on) |
| MySQL | Included in Starter tier |
| Disk (models + zones) | $0.10/GB/mo |
| Bandwidth | Free tier: 100GB/mo |

**Total: ~$5-7/mo** for single-camera setup
