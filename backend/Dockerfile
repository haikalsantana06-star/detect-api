# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (try PyTorch CPU index first, fallback to PyPI)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0

# Ensure numpy<2 for PyTorch 2.1 compatibility
RUN pip install --no-cache-dir "numpy<2"

# Copy requirements and install other dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Force numpy<2 again to override any transitive upgrade
RUN pip install --no-cache-dir "numpy<2"

# ============================================
# Stage 2: Runtime - Production image
# ============================================
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY main.py config.py schemas.py stats_schemas.py stats_computer.py model_loader.py pytorch_inference.py angle_detector.py metrics.py zones.json start.sh ./

# Copy model weights and SSIM reference images (from repo root)
COPY models/ ./models/
COPY dataset/sorted/ ./dataset/sorted/

# Environment variables (override via Railway dashboard)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=cpu
ENV MODEL_DIR=models
ENV ZONES_PATH=zones.json
ENV THRESHOLD=0.5

# Railway provides PORT env var
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=cpu
ENV MODEL_DIR=models
ENV ZONES_PATH=zones.json
ENV THRESHOLD=0.5

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run startup script (reads PORT env)
CMD ["bash", "start.sh"]
