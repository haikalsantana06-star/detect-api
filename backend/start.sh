#!/bin/bash
# Railway startup script - reads PORT from environment
set -e

PORT=${PORT:-8000}
echo "Starting CAPE API on port $PORT..."

# Copy models from volume if present
if [ -d "/data/models" ]; then
    cp -r /data/models/* models/ 2>/dev/null || true
fi

# Copy zones.json if present
if [ -f "/data/zones.json" ]; then
    cp /data/zones.json zones.json
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT
