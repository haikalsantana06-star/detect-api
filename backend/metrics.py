"""Prometheus metrics for CAPE API."""

from prometheus_client import Counter, Gauge, Histogram

# Inference timing histogram (seconds)
cape_inference_duration_seconds = Histogram(
    "cape_inference_duration_seconds",
    "Time spent running inference per request",
    ("angle",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Total inference requests counter
cape_inference_requests_total = Counter(
    "cape_inference_requests_total",
    "Total inference requests",
    ("angle",),
)

# Detection counter by desk and occupancy status
cape_detections_total = Counter(
    "cape_detections_total",
    "Total detections by desk and status",
    ("desk", "status"),
)

# Model load status gauge per angle (1=loaded, 0=unloaded)
cape_model_load_status = Gauge(
    "cape_model_load_status",
    "Model load status per angle (1=loaded, 0=unloaded)",
    ("angle",),
)

# Active DB connection pool gauge
cape_db_connection_pool_active = Gauge(
    "cape_db_connection_pool_active",
    "Number of active DB connections in pool",
)
