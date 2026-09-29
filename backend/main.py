"""FastAPI application for desk occupancy detection and management indicators."""
import logging
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from config import get_config, get_settings
from schemas import DetectRequest, DetectResponse, DeskResult, HealthResponse
from stats_schemas import StatsResponse, DeskStatsResponse
from model_loader import ModelLoader
from pytorch_inference import run_inference
from angle_detector import detect_angle
from stats_computer import compute_stats, compute_desk_stats, save_detection_logs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Desk Occupancy Detection API",
    description="Detect desk occupancy + 5 management indicators from ESP32-CAM images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_loader: ModelLoader | None = None


@app.on_event("startup")
def startup():
    global _model_loader
    settings = get_settings()
    config = get_config()

    logger.info(f"Initializing with device={settings.device}, model_dir={settings.model_dir}")

    _model_loader = ModelLoader(
        model_dir=Path(__file__).parent / settings.model_dir,
        device=settings.device,
    )

    available_angles = ["angle_1", "angle_2"]
    for angle in available_angles:
        try:
            _model_loader.load_model(angle)
        except FileNotFoundError:
            logger.warning(f"Model for {angle} not found, skipping preload")

    logger.info(f"API ready. Models loaded: {_model_loader.get_loaded_angles()}")


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
        import base64, numpy as np, cv2
        try:
            raw = base64.b64decode(req.image_base64)
        except Exception:
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
        results = run_inference(
            image_source=req.image_base64,
            angle=angle,
            threshold=threshold,
            model_loader=_model_loader,
        )
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
        import numpy as np, cv2
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
        results = run_inference(
            image_source=content,
            angle=angle,
            threshold=effective_threshold,
            model_loader=_model_loader,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    desk_results = {desk: DeskResult(**data) for desk, data in results.items()}
    return DetectResponse(angle=angle, desks=desk_results)


@app.get("/stats/{person}", response_model=StatsResponse)
def get_stats_by_person(
    person: str,
    date: date = Query(default=None, description="Date in YYYY-MM-DD format, defaults to today"),
):
    """Get 5 management indicators for a person on a given date."""
    target_date = date if date is not None else date.today()

    try:
        return compute_stats(person, target_date)
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

    try:
        return compute_desk_stats(desk, target_date)
    except Exception as e:
        logger.exception(f"Stats computation failed for desk={desk}, date={target_date}")
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
