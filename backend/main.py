"""FastAPI application for desk occupancy detection and management indicators."""
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBasicCredentials

from angle_detector import detect_angle
from config import (
    clear_stats_cache,
    get_cached_stats,
    get_config,
    get_db_connection,
    get_settings,
    init_db_pool,
)
from metrics import (
    cape_db_connection_pool_active,
    cape_detections_total,
    cape_inference_duration_seconds,
    cape_inference_requests_total,
    cape_model_load_status,
)
from prometheus_client import generate_latest
from model_loader import ModelLoader
from pytorch_inference import run_inference
from schemas import DeskResult, DetectRequest, DetectResponse, HealthResponse
from stats_computer import (
    compute_aggregate,
    compute_daily_stats,
    compute_desk_stats,
    compute_person_comparison,
    compute_stats,
    compute_trend_series,
    get_available_dates,
)
from stats_schemas import (
    AggregateStats,
    AvailableDates,
    DailyStats,
    DeskStatsResponse,
    PersonComparison,
    PresenceStatus,
    StatsResponse,
    TrendPoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_model_loader: ModelLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _model_loader
    settings = get_settings()
    config = get_config()

    logger.info(f"Initializing with device={settings.device}, model_dir={settings.model_dir}")

    # Initialize DB connection pool
    init_db_pool(config)
    logger.info("DB connection pool initialized")
    cape_db_connection_pool_active.set(0)

    _model_loader = ModelLoader(
        model_dir=Path(__file__).parent / settings.model_dir,
        device=settings.device,
    )

    available_angles = ["angle_1", "angle_2"]
    for angle in available_angles:
        try:
            _model_loader.load_model(angle)
            cape_model_load_status.labels(angle=angle).set(1)
        except FileNotFoundError:
            logger.warning(f"Model for {angle} not found, skipping preload")
            cape_model_load_status.labels(angle=angle).set(0)

    logger.info(f"API ready. Models loaded: {_model_loader.get_loaded_angles()}")

    yield

    # Shutdown
    clear_stats_cache()
    logger.info("Shutting down. Stats cache cleared.")


app = FastAPI(
    title="Desk Occupancy Detection API",
    description="Detect desk occupancy + 5 management indicators from ESP32-CAM images",
    version="1.0.0",
    lifespan=lifespan,
)

config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/metrics")
def metrics(credentials: HTTPBasicCredentials = None):
    """Prometheus metrics endpoint — protected by HTTP Basic auth if password is set."""
    import os
    expected_user = os.environ.get("METRICS_USER", "prometheus")
    expected_pass = os.environ.get("METRICS_PASSWORD", "")
    # If env vars not set, allow unauthenticated access (local dev)
    if expected_pass:
        if not credentials or credentials.username != expected_user or credentials.password != expected_pass:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic"},
            )
    return PlainTextResponse(generate_latest(), media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — returns status and loaded models."""
    return HealthResponse(
        status="ok",
        device=get_settings().device,
        models_loaded=_model_loader.get_loaded_angles() if _model_loader else [],
    )


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    """
    Detect occupancy from a base64-encoded image.
    If `angle` is not provided, auto-detects the angle using SSIM matching.
    """
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    config = get_config()
    threshold = req.threshold if req.threshold is not None else config.threshold

    angle = req.angle
    if angle is None:
        import base64

        import cv2
        import numpy as np
        try:
            raw = base64.b64decode(req.image_base64)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        detected = detect_angle(img)
        if detected is None:
            raise HTTPException(
                status_code=422,
                detail="Could not auto-detect angle. Please specify `angle` explicitly.",
            )
        angle = detected

    if angle not in config.zones:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown angle '{angle}'. Available: {list(config.zones.keys())}",
        )

    try:
        import time
        t0 = time.perf_counter()
        results = run_inference(
            image_source=req.image_base64,
            angle=angle,
            threshold=threshold,
            model_loader=_model_loader,
        )
        duration = time.perf_counter() - t0
        cape_inference_duration_seconds.labels(angle=angle).observe(duration)
        cape_inference_requests_total.labels(angle=angle).inc()
        for desk, data in results.items():
            status = "occupied" if data["occupied"] else "empty"
            cape_detections_total.labels(desk=desk, status=status).inc()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    desk_results = {desk: DeskResult(**data) for desk, data in results.items()}
    return DetectResponse(angle=angle, desks=desk_results)


@app.post("/detect/file", response_model=DetectResponse)
async def detect_file(
    file: UploadFile = File(...),
    angle: str | None = Form(None),
    threshold: float | None = Form(None),
):
    """Detect from multipart file upload."""
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    config = get_config()
    effective_threshold = threshold if threshold is not None else config.threshold

    content = await file.read()

    if angle is None:
        import cv2
        import numpy as np
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            detected = detect_angle(img)
            if detected is None:
                raise HTTPException(
                    status_code=422,
                    detail="Could not auto-detect angle. Please specify `angle` explicitly.",
                )
            angle = detected
        else:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    if angle not in config.zones:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown angle '{angle}'. Available: {list(config.zones.keys())}",
        )

    try:
        import time
        t0 = time.perf_counter()
        results = run_inference(
            image_source=content,
            angle=angle,
            threshold=effective_threshold,
            model_loader=_model_loader,
        )
        duration = time.perf_counter() - t0
        cape_inference_duration_seconds.labels(angle=angle).observe(duration)
        cape_inference_requests_total.labels(angle=angle).inc()
        for desk, data in results.items():
            status = "occupied" if data["occupied"] else "empty"
            cape_detections_total.labels(desk=desk, status=status).inc()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    desk_results = {desk: DeskResult(**data) for desk, data in results.items()}
    return DetectResponse(angle=angle, desks=desk_results)


@app.post("/receive", response_class=PlainTextResponse)
async def receive_esp32(
    file: UploadFile = File(...),
    angle: str | None = Form(None),
):
    """
    ESP32-CAM endpoint — receives image upload, runs inference,
    saves results to MySQL, returns simple 200 OK.

    ESP32-CAM POST multipart/form-data with field: imageFile
    Optional form field: angle (auto-detected if omitted)
    """
    if _model_loader is None:
        logger.error("Models not loaded")
        return "ERROR: Models not loaded"

    config = get_config()
    threshold = config.threshold
    timestamp = datetime.now()

    content = await file.read()

    # Decode image for angle detection
    import cv2
    import numpy as np
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Could not decode uploaded image")
        return "ERROR: Could not decode image"

    # Auto-detect angle if not provided
    if angle is None:
        detected = detect_angle(img)
        if detected is None:
            logger.warning("Could not auto-detect angle, defaulting to angle_1")
            angle = "angle_1"
        else:
            angle = detected
            logger.info(f"Auto-detected angle: {angle}")

    if angle not in config.zones:
        logger.error(f"Unknown angle: {angle}")
        return f"ERROR: Unknown angle '{angle}'"

    # Run inference
    try:
        import time
        t0 = time.perf_counter()
        results = run_inference(
            image_source=content,
            angle=angle,
            threshold=threshold,
            model_loader=_model_loader,
        )
        duration = time.perf_counter() - t0
        cape_inference_duration_seconds.labels(angle=angle).observe(duration)
        cape_inference_requests_total.labels(angle=angle).inc()
        for desk, data in results.items():
            status = "occupied" if data["occupied"] else "empty"
            cape_detections_total.labels(desk=desk, status=status).inc()
    except Exception:
        logger.exception("Inference failed")
        return "ERROR: Inference failed"

    # Save to MySQL detection_logs
    try:
        _save_detection_logs(results, angle, timestamp)
    except Exception:
        logger.exception("Failed to save to DB")
        return "ERROR: Database write failed"

    occupied_count = sum(1 for d in results.values() if d["occupied"])
    logger.info(f"ESP32 frame processed — angle={angle}, occupied={occupied_count}/{len(results)}")
    return "OK"


def _save_detection_logs(results: dict, angle: str, timestamp: datetime):
    """Insert detection results into MySQL detection_logs table."""
    cfg = get_config()
    person_map = cfg.person_map

    conn = get_db_connection()
    cape_db_connection_pool_active.inc()
    cursor = conn.cursor()

    for desk, data in results.items():
        person = person_map.get(desk, desk)
        occupied = 1 if data["occupied"] else 0
        confidence = float(data["confidence"])

        cursor.execute(
            """
            INSERT INTO detection_logs
                (image_timestamp, person, desk, occupied, confidence, angle)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (timestamp, person, desk, occupied, confidence, angle),
        )

    conn.commit()
    cursor.close()
    conn.close()
    cape_db_connection_pool_active.dec()


@app.get("/stats/{person}", response_model=StatsResponse)
def get_stats_by_person(
    person: str,
    date: date = Query(default=None, description="Date in YYYY-MM-DD format, defaults to today"),
):
    """Get 5 management indicators for a person on a given date."""
    target_date = date if date is not None else date.today()
    cache_key = f"stats:person:{person}:{target_date.isoformat()}"

    def compute():
        return compute_stats(person, target_date)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Stats computation failed for person={person}, date={target_date}")
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")


@app.get("/stats/desk/{desk}", response_model=DeskStatsResponse)
def get_stats_by_desk(
    desk: str,
    date: date = Query(default=None, description="Date in YYYY-MM-DD format, defaults to today"),
):
    """Get 5 management indicators for a desk on a given date."""
    target_date = date if date is not None else date.today()
    cache_key = f"stats:desk:{desk}:{target_date.isoformat()}"

    def compute():
        return compute_desk_stats(desk, target_date)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Stats computation failed for desk={desk}, date={target_date}")
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")


# ---------------------------------------------------------------------------
# Enhanced Stats Endpoints
# ---------------------------------------------------------------------------


@app.get("/stats/available-dates", response_model=AvailableDates)
def get_available_dates_endpoint(
    person: str | None = Query(None, description="Filter by person"),
    year_month: str | None = Query(None, description="YYYY-MM filter"),
):
    """
    Return list of dates that have detection data.
    Frontend uses this to populate date/week/month selectors.
    """
    cache_key = f"available_dates:{person}:{year_month}"

    def compute():
        dates = get_available_dates(person=person, year_month=year_month)
        return AvailableDates(dates=dates)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception("Failed to get available dates")
        raise HTTPException(status_code=500, detail=f"Failed to get available dates: {e}")


@app.get("/stats/daily/{person}", response_model=DailyStats)
def get_daily_stats_endpoint(
    person: str,
    date: date = Query(..., description="YYYY-MM-DD"),
):
    """
    Get DailyStats for one person on one date.
    Returns three-state presence_status.
    """
    cache_key = f"daily:{person}:{date.isoformat()}"

    def compute():
        return compute_daily_stats(person, date)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Failed to compute daily stats for person={person}, date={date}")
        raise HTTPException(status_code=500, detail=f"Failed to compute daily stats: {e}")


@app.get("/stats/aggregate", response_model=AggregateStats)
def get_aggregate_stats_endpoint(
    scope: str = Query(..., description="weekly | monthly"),
    period: str = Query(..., description="YYYY-Www (week) or YYYY-MM (month)"),
    person: str | None = Query(None, description="Specific person or omit for all"),
):
    """
    Get AggregateStats for a week or month.

    Query examples:
    - /stats/aggregate?scope=weekly&period=2025-W03&person=Asep
    - /stats/aggregate?scope=monthly&period=2025-01
    """
    import calendar
    from datetime import timedelta

    # Parse period
    if scope == "weekly":
        # period format: YYYY-Www
        try:
            year, week = period.split("-W")
            week_num = int(week)
            # ISO week: get the Monday of that week
            jan4 = date(int(year), 1, 4)  # Jan 4 is always in week 1
            start_date = jan4 + timedelta(weeks=week_num - 1, days=-jan4.weekday())
            end_date = start_date + timedelta(days=6)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid weekly period format. Use YYYY-Www")
    elif scope == "monthly":
        # period format: YYYY-MM
        try:
            year, month = period.split("-")
            year, month = int(year), int(month)
            _, last_day = calendar.monthrange(year, month)
            start_date = date(year, month, 1)
            end_date = date(year, month, last_day)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid monthly period format. Use YYYY-MM")
    else:
        raise HTTPException(status_code=422, detail="Scope must be 'weekly' or 'monthly'")

    cache_key = f"aggregate:{scope}:{period}:{person}"

    def compute():
        return compute_aggregate(person=person, start_date=start_date, end_date=end_date, scope=scope)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Failed to compute aggregate stats scope={scope} period={period} person={person}")
        raise HTTPException(status_code=500, detail=f"Failed to compute aggregate stats: {e}")


@app.get("/stats/trend", response_model=list[TrendPoint])
def get_trend_series_endpoint(
    scope: str = Query(..., description="weekly | monthly"),
    period: str = Query(...),
    person: str | None = Query(None),
):
    """
    Get daily TrendPoints for a period (for trend charts).
    Returns array even for days with no data (marked NO_DATA).
    """
    import calendar
    from datetime import timedelta

    # Parse period
    if scope == "weekly":
        try:
            year, week = period.split("-W")
            week_num = int(week)
            jan4 = date(int(year), 1, 4)
            start_date = jan4 + timedelta(weeks=week_num - 1, days=-jan4.weekday())
            end_date = start_date + timedelta(days=6)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid weekly period format. Use YYYY-Www")
    elif scope == "monthly":
        try:
            year, month = period.split("-")
            year, month = int(year), int(month)
            _, last_day = calendar.monthrange(year, month)
            start_date = date(year, month, 1)
            end_date = date(year, month, last_day)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid monthly period format. Use YYYY-MM")
    else:
        raise HTTPException(status_code=422, detail="Scope must be 'weekly' or 'monthly'")

    cache_key = f"trend:{scope}:{period}:{person}"

    def compute():
        return compute_trend_series(person=person, start_date=start_date, end_date=end_date)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Failed to compute trend series scope={scope} period={period} person={person}")
        raise HTTPException(status_code=500, detail=f"Failed to compute trend series: {e}")


@app.get("/stats/comparison", response_model=list[PersonComparison])
def get_person_comparison_endpoint(
    scope: str = Query(..., description="weekly | monthly"),
    period: str = Query(...),
):
    """
    Get PersonComparison for all people in person_map.
    Used when 'Semua orang' is selected.
    """
    import calendar
    from datetime import timedelta

    if scope == "weekly":
        try:
            year, week = period.split("-W")
            week_num = int(week)
            jan4 = date(int(year), 1, 4)
            start_date = jan4 + timedelta(weeks=week_num - 1, days=-jan4.weekday())
            end_date = start_date + timedelta(days=6)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid weekly period format. Use YYYY-Www")
    elif scope == "monthly":
        try:
            year, month = period.split("-")
            year, month = int(year), int(month)
            _, last_day = calendar.monthrange(year, month)
            start_date = date(year, month, 1)
            end_date = date(year, month, last_day)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Invalid monthly period format. Use YYYY-MM")
    else:
        raise HTTPException(status_code=422, detail="Scope must be 'weekly' or 'monthly'")

    cache_key = f"comparison:{scope}:{period}"

    def compute():
        return compute_person_comparison(start_date=start_date, end_date=end_date)

    try:
        return get_cached_stats(cache_key, compute, ttl_seconds=60)
    except Exception as e:
        logger.exception(f"Failed to compute person comparison scope={scope} period={period}")
        raise HTTPException(status_code=500, detail=f"Failed to compute person comparison: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
