# CAPE API

FastAPI backend for desk occupancy detection + 5 management indicators.

## Stack

- **FastAPI** + Python + PyTorch (MobileNetV2)
- **MySQL** — detection logs + stats
- **Prometheus** metrics
- **Docker** — Railway deploy
- **Railway** — hosting

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics (Basic auth) |
| `POST` | `/detect` | Detect from base64 image |
| `POST` | `/detect/file` | Detect from multipart file upload |
| `POST` | `/receive` | ESP32-CAM endpoint (auto-save to DB) |
| `GET` | `/stats/{person}` | 5 management indicators by person |
| `GET` | `/stats/desk/{desk}` | 5 management indicators by desk |

## Local Development

```bash
cd backend
pip install -r requirements-docker.txt
cp .env.example .env   # fill in DB credentials
uvicorn main:app --reload --port 8000
```

## Docker

```bash
cd backend
docker build -t cape-api .
docker run -p 8000:8000 --env-file .env cape-api
```

## Railway Deploy

1. Connect repo → Railway
2. Set env vars: `DB__HOST`, `DB__PORT`, `DB__NAME`, `DB__USER`, `DB__PASSWORD`, `DEVICE`, `MODEL_DIR`
3. Upload model files to Railway volume `/data/models`
4. Push a tag `v*` to trigger deploy

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `MODEL_DIR` | `models` | Path to model weights |
| `ZONES_PATH` | `zones.json` | Path to zones config |
| `THRESHOLD` | `0.5` | Detection confidence threshold |
| `METRICS_USER` | `prometheus` | Basic auth user for `/metrics` |
| `METRICS_PASSWORD` | _(empty)_ | Basic auth pass (unset = open) |
| `DB__HOST` | _(required)_ | MySQL host |
| `DB__PORT` | `3306` | MySQL port |
| `DB__NAME` | _(required)_ | MySQL database name |
| `DB__USER` | _(required)_ | MySQL username |
| `DB__PASSWORD` | _(required)_ | MySQL password |
