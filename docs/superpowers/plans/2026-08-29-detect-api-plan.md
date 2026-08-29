# Detect API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend service with desk occupancy detection endpoints AND 5 management indicator statistics, loading trained PyTorch models directly (no ONNX conversion needed).

**Architecture:** FastAPI app loads PyTorch MobileNetV2 classifiers per angle at startup, caches them in memory. Inference pipeline: receive image → SSIM angle detection (optional) → crop desk zones → preprocess → model inference → return structured occupancy results. PHP calls `/detect` for inference and writes to MySQL. FastAPI `/stats` endpoints read from MySQL and compute 5 management indicators.

**Tech Stack:** FastAPI, Uvicorn, PyTorch, torchvision, OpenCV, scikit-image, Pillow, Pydantic, mysql-connector-python

**Spec:** `docs/superpowers/specs/2026-08-29-detect-api-design.md`

---

## Global Constraints

- Image input size: 96×96
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Model architecture: MobileNetV2 + Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→1)
- Classifier path: `models/best_model_angle_{n}.pth`
- Zones config: `zones.json` (copied from repo root)
- Threshold default: 0.5
- Device default: cpu
- Work hours: 07:00 – 17:00 (600 minutes/day)
- Detection interval assumption: 5 minutes (used for time calculations)

---

## File Map

```
detect-api/
├── .env                     # Task 1 (updated)
├── requirements.txt         # Task 1 (updated)
├── config.py               # Task 1 (updated with DB config)
├── schemas.py               # Task 3
├── model_loader.py          # Task 4
├── angle_detector.py        # Task 5
├── pytorch_inference.py     # Task 6
├── stats_schemas.py         # Task 7
├── stats_computer.py        # Task 8
├── main.py                  # Task 9 (updated with stats endpoints)
├── sql/
│   └── schema.sql           # Task 2
└── README.md                # Task 10 (updated)
```

Shared model architecture lives in `detect.py` (existing) — will be reused by reference.

---

## Tasks

### Task 1: Project scaffold — config, dependencies, env (updated)

**Files:**
- Create: `detect-api/requirements.txt`
- Create: `detect-api/.env`
- Create: `detect-api/config.py`

**Interfaces:**
- Consumes: `zones.json` (from repo root), `.env` file
- Produces: `Config` dataclass with `zones`, `threshold`, `device`, `model_dir`, `db_config`

---

- [ ] **Step 1: Create `detect-api/` directory**

```bash
mkdir -p detect-api/sql
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
torch>=2.0.0
torchvision>=0.15.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
opencv-python>=4.8.0
scikit-image>=0.21.0
pillow>=10.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
mysql-connector-python>=8.0.0
```

- [ ] **Step 3: Write `.env`**

```
THRESHOLD=0.5
DEVICE=cpu
MODEL_DIR=models
ZONES_PATH=zones.json

# MySQL (for stats endpoint)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=upitas_mon
DB_USER=
DB_PASSWORD=
```

- [ ] **Step 4: Write `config.py`**

```python
"""Configuration loader — zones.json + .env"""
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import json


load_dotenv()


class DBSettings(BaseSettings):
    host: str = "localhost"
    port: int = 3306
    name: str = "upitas_mon"
    user: str = ""
    password: str = ""


class Settings(BaseSettings):
    threshold: float = 0.5
    device: str = "cpu"
    model_dir: str = "models"
    zones_path: str = "zones.json"
    db: DBSettings = DBSettings()

    class Config:
        env_nested = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@dataclass
class Zone:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Config:
    zones: dict[str, dict[str, Zone]]  # {angle: {desk: Zone}}
    desks: list[str]
    person_map: dict[str, str]
    threshold: float
    device: str
    model_dir: Path
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str


def load_config() -> Config:
    settings = get_settings()

    zones_path = Path(settings.zones_path)
    if not zones_path.exists():
        raise FileNotFoundError(f"zones.json not found at {zones_path}")

    with open(zones_path) as f:
        raw = json.load(f)

    # Parse zones into Zone dataclasses
    parsed_zones = {}
    for angle, angle_zones in raw["zones"].items():
        parsed_zones[angle] = {}
        for desk, zone_data in angle_zones.items():
            parsed_zones[angle][desk] = Zone(
                x=zone_data["x"],
                y=zone_data["y"],
                w=zone_data["w"],
                h=zone_data["h"],
            )

    return Config(
        zones=parsed_zones,
        desks=raw["desks"],
        person_map=raw.get("person_map", {}),
        threshold=settings.threshold,
        device=settings.device,
        model_dir=Path(settings.model_dir),
        db_host=settings.db.host,
        db_port=settings.db.port,
        db_name=settings.db.name,
        db_user=settings.db.user,
        db_password=settings.db.password,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
```

- [ ] **Step 5: Commit**

```bash
git add detect-api/requirements.txt detect-api/.env detect-api/config.py
git commit -m "feat(detect-api): scaffold project - requirements, config, env, DB config"
```

---

### Task 2: MySQL schema — detection_logs and presence_summary tables

**Files:**
- Create: `detect-api/sql/schema.sql`

---

- [ ] **Step 1: Write `sql/schema.sql`**

```sql
-- Run this once on your shared hosting MySQL (via phpMyAdmin or CLI)

CREATE DATABASE IF NOT EXISTS upitas_mon;
USE upitas_mon;

-- Raw detection log — one row per ESP32-CAM capture
CREATE TABLE IF NOT EXISTS detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_timestamp DATETIME NOT NULL,
    person VARCHAR(50) NOT NULL,
    desk VARCHAR(20) NOT NULL,
    occupied TINYINT(1) NOT NULL COMMENT '0=empty, 1=occupied',
    confidence FLOAT NOT NULL,
    angle VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_person_date (person, image_timestamp),
    INDEX idx_desk_date (desk, image_timestamp),
    INDEX idx_timestamp (image_timestamp)
);

-- Aggregated daily summary per person
CREATE TABLE IF NOT EXISTS presence_summary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    first_seen TIME DEFAULT NULL,
    last_seen TIME DEFAULT NULL,
    total_presence_minutes INT DEFAULT 0,
    work_hours_minutes INT DEFAULT 600,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_person_date (person, date)
);
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/sql/schema.sql
git commit -m "feat(detect-api): add MySQL schema for detection_logs and presence_summary"
```

---

### Task 3: Pydantic inference schemas — request/response models

**Files:**
- Create: `detect-api/schemas.py`

**Interfaces:**
- Consumes: nothing (standalone)
- Produces: `DetectRequest`, `DetectResponse`, `DeskResult`, `HealthResponse`

---

- [ ] **Step 1: Write `schemas.py`**

```python
"""Pydantic request/response schemas for inference endpoints."""
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG image")
    angle: str | None = Field(None, description="Angle name (auto-detected if omitted)")
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class DeskResult(BaseModel):
    occupied: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    person: str | None = None


class DetectResponse(BaseModel):
    angle: str
    desks: dict[str, DeskResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: list[str]
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/schemas.py
git commit -m "feat(detect-api): add Pydantic inference schemas"
```

---

### Task 4: Model loader — load & cache PyTorch models per angle

**Files:**
- Create: `detect-api/model_loader.py`

**Interfaces:**
- Consumes: `config.py` → `model_dir`, `device`
- Produces: `ModelLoader` class with `load_model(angle)` → `torch.nn.Module`

---

- [ ] **Step 1: Write `model_loader.py`**

```python
"""Load and cache PyTorch models per angle."""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MobileNetV2Classifier(nn.Module):
    """Must match the architecture used in train_classifier.py exactly."""

    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            self.model = models.mobilenet_v2(weights=None)

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ModelLoader:
    """Load and cache one model per angle."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self._cache: dict[str, nn.Module] = {}

    def load_model(self, angle: str) -> nn.Module:
        """Load model for given angle, cached in memory."""
        if angle in self._cache:
            return self._cache[angle]

        model_path = self.model_dir / f"best_model_{angle}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Available models: {list(self.model_dir.glob('best_model_*.pth'))}"
            )

        logger.info(f"Loading model from {model_path} on {self.device}")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        model = MobileNetV2Classifier(pretrained=False)
        state_dict = checkpoint["model_state_dict"]

        # Remove "model." prefix if present
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model for '{angle}' loaded. Val acc: {checkpoint.get('val_acc', 'N/A')}")
        self._cache[angle] = model
        return model

    def get_loaded_angles(self) -> list[str]:
        """Return list of angles with cached models."""
        return list(self._cache.keys())

    def preload_all(self, angles: list[str]) -> None:
        """Pre-load all specified angles at startup."""
        for angle in angles:
            try:
                self.load_model(angle)
            except FileNotFoundError as e:
                logger.warning(f"Could not preload {angle}: {e}")
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/model_loader.py
git commit -m "feat(detect-api): add ModelLoader - load & cache PyTorch models per angle"
```

---

### Task 5: Angle detector — SSIM-based auto-detection

**Files:**
- Create: `detect-api/angle_detector.py`

**Interfaces:**
- Consumes: `config.py` → `zones`, image numpy array
- Produces: `detect_angle(image) -> str | None`

---

- [ ] **Step 1: Write `angle_detector.py`**

```python
"""SSIM-based angle auto-detection."""
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import json
import logging

logger = logging.getLogger(__name__)

_refs_cache: dict[str, dict[str, np.ndarray]] = {}
_zones_cache: dict | None = None


def _load_zones() -> dict:
    global _zones_cache
    if _zones_cache is not None:
        return _zones_cache

    zones_path = Path(__file__).parent.parent / "zones.json"
    if not zones_path.exists():
        zones_path = Path("zones.json")

    if not zones_path.exists():
        raise FileNotFoundError("zones.json not found")

    with open(zones_path) as f:
        raw = json.load(f)
    _zones_cache = raw["zones"]
    return _zones_cache


def _build_ssim_references(sorted_dir: str = "dataset/sorted") -> dict[str, dict[str, np.ndarray]]:
    """
    Build average grayscale reference crops for each angle from labeled images.
    Returns: {angle_name: {desk_name: avg_grayscale_crop}}
    """
    global _refs_cache
    if _refs_cache:
        return _refs_cache

    sorted_path = Path(sorted_dir)
    if not sorted_path.exists():
        logger.warning(f"SSIM reference dir not found: {sorted_dir}")
        _refs_cache = {}
        return _refs_cache

    all_zones = _load_zones()
    desks = ["desk_a", "desk_b", "desk_c", "desk_d"]

    for angle_folder in sorted_path.iterdir():
        if not angle_folder.is_dir():
            continue
        if not angle_folder.name.startswith("angle_"):
            continue

        angle_name = angle_folder.name

        img_files = [
            f for f in angle_folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not img_files:
            continue

        if angle_name not in all_zones:
            continue

        desk_crops: dict[str, list[np.ndarray]] = {d: [] for d in desks}

        for img_path in img_files:
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            for d in desks:
                if d not in all_zones[angle_name]:
                    continue
                z = all_zones[angle_name][d]
                crop = gray[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]]
                if crop.size > 0:
                    desk_crops[d].append(crop)

        if all(desk_crops[d] for d in desks):
            _refs_cache[angle_name] = {}
            for d in desks:
                stack = np.stack(desk_crops[d])
                _refs_cache[angle_name][d] = np.mean(stack, axis=0)

    logger.info(f"Built SSIM references for angles: {list(_refs_cache.keys())}")
    return _refs_cache


def detect_angle(image_path: str | Path | np.ndarray) -> str | None:
    """
    Auto-detect which angle an image belongs to using SSIM reference matching.
    Returns angle name (e.g. 'angle_1') or None if no confident match.
    """
    refs = _build_ssim_references()
    if not refs:
        logger.warning("No SSIM references available, cannot auto-detect angle")
        return None

    if isinstance(image_path, (str, Path)):
        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            return None
    else:
        img_cv = image_path

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    all_zones = _load_zones()
    desks = ["desk_a", "desk_b", "desk_c", "desk_d"]

    angle_scores: dict[str, float] = {}

    for angle_name, ref_desks in refs.items():
        if angle_name not in all_zones:
            continue
        score = 0.0
        valid = 0
        for d in desks:
            if d not in ref_desks or d not in all_zones[angle_name]:
                continue
            z = all_zones[angle_name][d]
            crop = gray[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]]
            if crop.size == 0:
                continue
            try:
                s = ssim(crop, ref_desks[d], data_range=255)
                score += s
                valid += 1
            except Exception:
                continue
        if valid > 0:
            angle_scores[angle_name] = score / valid

    if not angle_scores:
        return None

    best_angle = max(angle_scores, key=angle_scores.get)
    best_score = angle_scores[best_angle]

    if best_score < 0.1:
        return None

    logger.info(f"Detected angle: {best_angle} (score: {best_score:.3f})")
    return best_angle
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/angle_detector.py
git commit -m "feat(detect-api): add SSIM-based angle auto-detection"
```

---

### Task 6: Inference engine — crop, preprocess, run model

**Files:**
- Create: `detect-api/pytorch_inference.py`

**Interfaces:**
- Consumes: `model_loader.py` → model, `config.py` → zones
- Produces: `run_inference(image_source, angle, threshold, model_loader) -> dict[str, dict]`

---

- [ ] **Step 1: Write `pytorch_inference.py`**

```python
"""Inference engine: crop desk zones, preprocess, run PyTorch model."""
import base64
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
import logging

from model_loader import ModelLoader
from config import get_config, Zone

logger = logging.getLogger(__name__)

IMG_SIZE = (96, 96)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _decode_image(source: str | bytes) -> np.ndarray:
    """Decode base64 string or raw bytes to OpenCV image (BGR)."""
    if isinstance(source, str):
        if "," in source:
            source = source.split(",", 1)[1]
        raw = base64.b64decode(source)
    else:
        raw = source

    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _crop_and_preprocess(img: np.ndarray, zone: Zone) -> torch.Tensor:
    """Crop a desk zone from the image and preprocess for model input."""
    x, y, w, h = zone.x, zone.y, zone.w, zone.h
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"Empty crop at zone ({x}, {y}, {w}, {h})")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensor = _transform(pil)
    return tensor


def run_inference(
    image_source: str | bytes,
    angle: str,
    threshold: float,
    model_loader: ModelLoader,
) -> dict[str, dict]:
    """
    Run inference on an image for a given angle.

    Args:
        image_source: Base64 string or raw bytes of the image
        angle: Angle name (e.g. 'angle_1')
        threshold: Occupancy threshold (0-1)
        model_loader: ModelLoader instance with cached models

    Returns:
        Dict mapping desk name -> {occupied: bool, confidence: float, person: str}
    """
    img = _decode_image(image_source)
    model = model_loader.load_model(angle)
    config = get_config()
    angle_zones = config.zones.get(angle, {})
    person_map = config.person_map

    results = {}

    for desk, zone in angle_zones.items():
        try:
            tensor = _crop_and_preprocess(img, zone)
            tensor = tensor.unsqueeze(0).to(model_loader.device)

            with torch.no_grad():
                output = model(tensor)
                output = output.view(-1)
                probability = torch.sigmoid(output).item()
                is_occupied = probability >= threshold

            results[desk] = {
                "occupied": is_occupied,
                "confidence": round(float(probability), 4),
                "person": person_map.get(desk, desk),
            }
        except Exception as e:
            logger.error(f"Inference failed for {desk} (angle {angle}): {e}")
            results[desk] = {
                "occupied": False,
                "confidence": 0.0,
                "person": person_map.get(desk, desk),
            }

    return results
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/pytorch_inference.py
git commit -m "feat(detect-api): add inference engine - crop, preprocess, PyTorch inference"
```

---

### Task 7: Stats Pydantic schemas

**Files:**
- Create: `detect-api/stats_schemas.py`

**Interfaces:**
- Consumes: nothing (standalone)
- Produces: `StatsResponse`, `Indicator`, `WorkHours`

---

- [ ] **Step 1: Write `stats_schemas.py`**

```python
"""Pydantic schemas for stats endpoints and 5 management indicators."""
from pydantic import BaseModel, Field
from typing import Any


class Indicator(BaseModel):
    label: str
    value: Any  # bool | str | int | float
    unit: str | None = None
    description: str


class WorkHours(BaseModel):
    start: str = "07:00"
    end: str = "17:00"
    total_minutes: int = 600


class StatsResponse(BaseModel):
    person: str
    date: str
    work_hours: WorkHours
    indicators: dict[str, Indicator]


class DeskStatsResponse(BaseModel):
    desk: str
    date: str
    work_hours: WorkHours
    indicators: dict[str, Indicator]
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/stats_schemas.py
git commit -m "feat(detect-api): add Pydantic schemas for stats endpoints"
```

---

### Task 8: Stats computer — compute 5 indicators from MySQL detection logs

**Files:**
- Create: `detect-api/stats_computer.py`

**Interfaces:**
- Consumes: MySQL detection_logs table, `config.py` → DB config
- Produces: `compute_stats(person, date) -> StatsResponse`

---

- [ ] **Step 1: Write `stats_computer.py`**

```python
"""Compute 5 management indicators from detection_logs in MySQL."""
import mysql.connector
from datetime import datetime, date, time, timedelta
import logging

from config import get_config
from stats_schemas import StatsResponse, Indicator, WorkHours

logger = logging.getLogger(__name__)

WORK_START = time(7, 0)
WORK_END = time(17, 0)
DETECTION_INTERVAL_MINUTES = 5


def _get_db_connection():
    """Create a MySQL connection using config."""
    cfg = get_config()
    return mysql.connector.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        database=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
    )


def _parse_datetime(dt_val) -> datetime | None:
    """Convert MySQL datetime result to Python datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, str):
        return datetime.fromisoformat(dt_val)
    return None


def compute_stats(person: str, target_date: date) -> StatsResponse:
    """
    Compute 5 management indicators for a person on a given date.

    Indicators:
    1. Tingkat Kehadiran    - Hadir jika ada >= 1 detection occupied di jam kerja
    2. Ketepatan Datang    - Waktu pertama occupied dalam jam kerja (HH:MM), "N/A" if none
    3. Lama Bekerja        - first_seen to last_seen dalam jam kerja (menit)
    4. Waktu Produktif     - Menit occupied di dalam jam kerja
    5. Waktu Tidak Produktif - Menit occupied di luar jam kerja

    Args:
        person: Person name (e.g. 'Asep')
        target_date: Date to compute stats for

    Returns:
        StatsResponse with all 5 indicators
    """
    conn = _get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Query all detections for this person on this date
    query = """
        SELECT image_timestamp, occupied
        FROM detection_logs
        WHERE person = %s
          AND DATE(image_timestamp) = %s
        ORDER BY image_timestamp ASC
    """
    cursor.execute(query, (person, target_date.isoformat()))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    # Default response (no data)
    work_hours = WorkHours(
        start="07:00",
        end="17:00",
        total_minutes=600,
    )

    if not rows:
        return StatsResponse(
            person=person,
            date=target_date.isoformat(),
            work_hours=work_hours,
            indicators={
                "tingkat_kehadiran": Indicator(
                    label="Tingkat Kehadiran",
                    value=False,
                    description="Hadir jika muncul >= 1x selama jam kerja"
                ),
                "ketepatan_datang": Indicator(
                    label="Ketepatan Waktu Datang",
                    value="N/A",
                    description="Waktu pertama muncul dalam jam kerja (HH:MM)"
                ),
                "lama_bekerja": Indicator(
                    label="Lama Berada di Kantor",
                    value=0,
                    unit="minutes",
                    description="Durasi dari first_seen sampai last_seen dalam jam kerja"
                ),
                "waktu_produktif": Indicator(
                    label="Waktu Produktif",
                    value=0,
                    unit="minutes",
                    description="Menit occupied selama jam kerja"
                ),
                "waktu_tidak_produktif": Indicator(
                    label="Waktu Tidak Produktif",
                    value=0,
                    unit="minutes",
                    description="Menit occupied di luar jam kerja"
                ),
            }
        )

    # Parse detections
    detections: list[tuple[datetime, bool]] = []
    for row in rows:
        dt = _parse_datetime(row["image_timestamp"])
        if dt:
            detections.append((dt, bool(row["occupied"])))

    if not detections:
        # All rows invalid
        return StatsResponse(
            person=person,
            date=target_date.isoformat(),
            work_hours=work_hours,
            indicators={
                "tingkat_kehadiran": Indicator(label="Tingkat Kehadiran", value=False, description="Hadir jika muncul >= 1x selama jam kerja"),
                "ketepatan_datang": Indicator(label="Ketepatan Waktu Datang", value="N/A", description="Waktu pertama muncul dalam jam kerja (HH:MM)"),
                "lama_bekerja": Indicator(label="Lama Berada di Kantor", value=0, unit="minutes", description="Durasi dari first_seen sampai last_seen dalam jam kerja"),
                "waktu_produktif": Indicator(label="Waktu Produktif", value=0, unit="minutes", description="Menit occupied selama jam kerja"),
                "waktu_tidak_produktif": Indicator(label="Waktu Tidak Produktif", value=0, unit="minutes", description="Menit occupied di luar jam kerja"),
            }
        )

    # --- Indicator 1: Tingkat Kehadiran ---
    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and WORK_START <= dt.time() <= WORK_END
    ]
    tingkat_kehadiran = len(occupied_in_work_hours) > 0

    # --- Indicator 2: Ketepatan Datang ---
    ketepatan_datang = "N/A"
    if occupied_in_work_hours:
        first = min(occupied_in_work_hours)
        ketepatan_datang = first.strftime("%H:%M")

    # --- Indicator 3: Lama Bekerja (within work hours) ---
    all_times = [dt for dt, _ in detections if WORK_START <= dt.time() <= WORK_END]
    lama_bekerja = 0
    if all_times:
        first_seen_dt = min(all_times)
        last_seen_dt = max(all_times)
        diff_minutes = int((last_seen_dt - first_seen_dt).total_seconds() / 60)

        # Clamp to work hours (max 600 minutes)
        lama_bekerja = min(diff_minutes, 600)

    # --- Indicator 4 & 5: Waktu Produktif vs Tidak Produktif ---
    productive_minutes = 0
    unproductive_minutes = 0

    for dt, occ in detections:
        if not occ:
            continue

        t = dt.time()
        if WORK_START <= t <= WORK_END:
            productive_minutes += DETECTION_INTERVAL_MINUTES
        else:
            unproductive_minutes += DETECTION_INTERVAL_MINUTES

    return StatsResponse(
        person=person,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators={
            "tingkat_kehadiran": Indicator(
                label="Tingkat Kehadiran",
                value=tingkat_kehadiran,
                description="Hadir jika muncul >= 1x selama jam kerja"
            ),
            "ketepatan_datang": Indicator(
                label="Ketepatan Waktu Datang",
                value=ketepatan_datang,
                description="Waktu pertama muncul dalam jam kerja (HH:MM)"
            ),
            "lama_bekerja": Indicator(
                label="Lama Berada di Kantor",
                value=lama_bekerja,
                unit="minutes",
                description="Durasi dari first_seen sampai last_seen dalam jam kerja"
            ),
            "waktu_produktif": Indicator(
                label="Waktu Produktif",
                value=productive_minutes,
                unit="minutes",
                description="Menit occupied selama jam kerja"
            ),
            "waktu_tidak_produktif": Indicator(
                label="Waktu Tidak Produktif",
                value=unproductive_minutes,
                unit="minutes",
                description="Menit occupied di luar jam kerja"
            ),
        }
    )


def compute_desk_stats(desk: str, target_date: date) -> StatsResponse:
    """
    Compute 5 management indicators for a desk on a given date.
    Same logic as compute_stats but queries by desk instead of person.
    """
    conn = _get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT image_timestamp, occupied
        FROM detection_logs
        WHERE desk = %s
          AND DATE(image_timestamp) = %s
        ORDER BY image_timestamp ASC
    """
    cursor.execute(query, (desk, target_date.isoformat()))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    work_hours = WorkHours(start="07:00", end="17:00", total_minutes=600)

    if not rows:
        return StatsResponse(
            person=desk,
            date=target_date.isoformat(),
            work_hours=work_hours,
            indicators={
                "tingkat_kehadiran": Indicator(label="Tingkat Kehadiran", value=False, description="Hadir jika muncul >= 1x selama jam kerja"),
                "ketepatan_datang": Indicator(label="Ketepatan Waktu Datang", value="N/A", description="Waktu pertama muncul dalam jam kerja (HH:MM)"),
                "lama_bekerja": Indicator(label="Lama Berada di Kantor", value=0, unit="minutes", description="Durasi dari first_seen sampai last_seen dalam jam kerja"),
                "waktu_produktif": Indicator(label="Waktu Produktif", value=0, unit="minutes", description="Menit occupied selama jam kerja"),
                "waktu_tidak_produktif": Indicator(label="Waktu Tidak Produktif", value=0, unit="minutes", description="Menit occupied di luar jam kerja"),
            }
        )

    detections: list[tuple[datetime, bool]] = []
    for row in rows:
        dt = _parse_datetime(row["image_timestamp"])
        if dt:
            detections.append((dt, bool(row["occupied"])))

    if not detections:
        return StatsResponse(
            person=desk,
            date=target_date.isoformat(),
            work_hours=work_hours,
            indicators={
                "tingkat_kehadiran": Indicator(label="Tingkat Kehadiran", value=False, description="Hadir jika muncul >= 1x selama jam kerja"),
                "ketepatan_datang": Indicator(label="Ketepatan Waktu Datang", value="N/A", description="Waktu pertama muncul dalam jam kerja (HH:MM)"),
                "lama_bekerja": Indicator(label="Lama Berada di Kantor", value=0, unit="minutes", description="Durasi dari first_seen sampai last_seen dalam jam kerja"),
                "waktu_produktif": Indicator(label="Waktu Produktif", value=0, unit="minutes", description="Menit occupied selama jam kerja"),
                "waktu_tidak_produktif": Indicator(label="Waktu Tidak Produktif", value=0, unit="minutes", description="Menit occupied di luar jam kerja"),
            }
        )

    occupied_in_work_hours = [
        dt for dt, occ in detections
        if occ and WORK_START <= dt.time() <= WORK_END
    ]
    tingkat_kehadiran = len(occupied_in_work_hours) > 0
    ketepatan_datang = "N/A"
    if occupied_in_work_hours:
        first = min(occupied_in_work_hours)
        ketepatan_datang = first.strftime("%H:%M")

    all_times = [dt for dt, _ in detections if WORK_START <= dt.time() <= WORK_END]
    lama_bekerja = 0
    if all_times:
        first_seen_dt = min(all_times)
        last_seen_dt = max(all_times)
        diff_minutes = int((last_seen_dt - first_seen_dt).total_seconds() / 60)
        lama_bekerja = min(diff_minutes, 600)

    productive_minutes = 0
    unproductive_minutes = 0
    for dt, occ in detections:
        if not occ:
            continue
        t = dt.time()
        if WORK_START <= t <= WORK_END:
            productive_minutes += DETECTION_INTERVAL_MINUTES
        else:
            unproductive_minutes += DETECTION_INTERVAL_MINUTES

    return StatsResponse(
        person=desk,
        date=target_date.isoformat(),
        work_hours=work_hours,
        indicators={
            "tingkat_kehadiran": Indicator(label="Tingkat Kehadiran", value=tingkat_kehadiran, description="Hadir jika muncul >= 1x selama jam kerja"),
            "ketepatan_datang": Indicator(label="Ketepatan Waktu Datang", value=ketepatan_datang, description="Waktu pertama muncul dalam jam kerja (HH:MM)"),
            "lama_bekerja": Indicator(label="Lama Berada di Kantor", value=lama_bekerja, unit="minutes", description="Durasi dari first_seen sampai last_seen dalam jam kerja"),
            "waktu_produktif": Indicator(label="Waktu Produktif", value=productive_minutes, unit="minutes", description="Menit occupied selama jam kerja"),
            "waktu_tidak_produktif": Indicator(label="Waktu Tidak Produktif", value=unproductive_minutes, unit="minutes", description="Menit occupied di luar jam kerja"),
        }
    )
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/stats_computer.py
git commit -m "feat(detect-api): add stats computer - compute 5 management indicators from MySQL"
```

---

### Task 9: FastAPI app — all endpoints including stats

**Files:**
- Create: `detect-api/main.py`

**Interfaces:**
- Consumes: all other modules
- Produces: FastAPI app with `/health`, `/detect`, `/detect/file`, `/stats/{person}`, `/stats/desk/{desk}`

---

- [ ] **Step 1: Write `main.py`**

```python
"""FastAPI application for desk occupancy detection and management indicators."""
import logging
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware

from config import get_config, get_settings
from schemas import DetectRequest, DetectResponse, DeskResult, HealthResponse
from stats_schemas import StatsResponse, DeskStatsResponse
from model_loader import ModelLoader
from pytorch_inference import run_inference
from angle_detector import detect_angle
from stats_computer import compute_stats, compute_desk_stats

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


# ============================================================
# INFERENCE ENDPOINTS
# ============================================================

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
        try:
            detected = detect_angle(req.image_base64)
            if detected is None:
                raise HTTPException(
                    status_code=422,
                    detail="Could not auto-detect angle. Please specify `angle` explicitly.",
                )
            angle = detected
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

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


# ============================================================
# STATS ENDPOINTS — 5 Management Indicators
# ============================================================

@app.get("/stats/{person}", response_model=StatsResponse)
def get_stats_by_person(
    person: str,
    date: date = Query(default=None, description="Date in YYYY-MM-DD format, defaults to today"),
):
    """
    Get 5 management indicators for a person on a given date.

    Indicators:
    1. Tingkat Kehadiran     - Hadir jika >= 1 detection occupied di jam kerja
    2. Ketepatan Datang     - Waktu pertama occupied dalam jam kerja (HH:MM)
    3. Lama Bekerja         - Durasi first_seen to last_seen (menit)
    4. Waktu Produktif      - Menit occupied di dalam jam kerja
    5. Waktu Tidak Produktif - Menit occupied di luar jam kerja
    """
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
```

- [ ] **Step 2: Commit**

```bash
git add detect-api/main.py
git commit -m "feat(detect-api): add FastAPI app with inference + stats endpoints"
```

---

### Task 10: README — usage documentation (updated with stats)

**Files:**
- Create: `detect-api/README.md`

---

- [ ] **Step 1: Write `README.md`**

````markdown
# Desk Occupancy Detection API

FastAPI backend for desk occupancy detection and 5 management indicators using trained PyTorch MobileNetV2 classifiers.

## Requirements

- Python 3.10+
- Trained model files: `models/best_model_angle_1.pth`, `models/best_model_angle_2.pth`
- `zones.json` copied to the project root
- MySQL database on shared hosting

## Quick Start

### 1. Install dependencies

```bash
cd detect-api
pip install -r requirements.txt
```

### 2. Copy model files and config

```bash
cp -r models detect-api/
cp zones.json detect-api/
```

### 3. Run MySQL schema on shared hosting

Upload and run `sql/schema.sql` via phpMyAdmin or MySQL CLI.

### 4. Configure database

Edit `.env`:
```
DB_HOST=<your-shared-hosting-mysql-ip>
DB_PORT=3306
DB_NAME=upitas_mon
DB_USER=<db_user>
DB_PASSWORD=<db_password>
```

### 5. Run the server

```bash
# Development
python main.py

# Production (VPS)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server runs on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## API Endpoints

### Inference

#### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "device": "cpu", "models_loaded": ["angle_1", "angle_2"]}
```

#### `POST /detect`
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'"$(base64 -w 0 image.jpg)"'", "angle": "angle_1"}'
```
```json
{
  "angle": "angle_1",
  "desks": {
    "desk_a": {"occupied": true, "confidence": 0.87, "person": "Asep"},
    "desk_b": {"occupied": false, "confidence": 0.12, "person": "Budi"},
    "desk_c": {"occupied": true, "confidence": 0.91, "person": "Saep"},
    "desk_d": {"occupied": false, "confidence": 0.05, "person": "naisya"}
  }
}
```

#### `POST /detect/file`
```bash
curl -X POST http://localhost:8000/detect/file \
  -F "file=@image.jpg" \
  -F "angle=angle_1"
```

### Statistics — 5 Management Indicators

#### `GET /stats/{person}?date=2026-08-29`
```bash
curl http://localhost:8000/stats/Asep?date=2026-08-29
```
```json
{
  "person": "Asep",
  "date": "2026-08-29",
  "work_hours": {"start": "07:00", "end": "17:00", "total_minutes": 600},
  "indicators": {
    "tingkat_kehadiran": {"label": "Tingkat Kehadiran", "value": true, "description": "Hadir jika muncul >= 1x selama jam kerja"},
    "ketepatan_datang": {"label": "Ketepatan Waktu Datang", "value": "08:15", "description": "Waktu pertama muncul dalam jam kerja (HH:MM)"},
    "lama_bekerja": {"label": "Lama Berada di Kantor", "value": 420, "unit": "minutes", "description": "Durasi dari first_seen sampai last_seen dalam jam kerja"},
    "waktu_produktif": {"label": "Waktu Produktif", "value": 390, "unit": "minutes", "description": "Menit occupied selama jam kerja"},
    "waktu_tidak_produktif": {"label": "Waktu Tidak Produktif", "value": 30, "unit": "minutes", "description": "Menit occupied di luar jam kerja"}
  }
}
```

#### `GET /stats/desk/{desk}?date=2026-08-29`
Same as above but queries by desk name instead of person name.

## Configuration

Edit `.env`:
```
THRESHOLD=0.5
DEVICE=cpu
MODEL_DIR=models
ZONES_PATH=zones.json

DB_HOST=localhost
DB_PORT=3306
DB_NAME=upitas_mon
DB_USER=
DB_PASSWORD=
```

## PHP Integration

### 1. Inference call — in existing PHP files after image upload

```php
// Call FastAPI /detect
$image_base64 = base64_encode(file_get_contents($uploaded_image_path));
$ch = curl_init('http://<vps-ip>:8000/detect');
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
    'image_base64' => $image_base64,
    'angle' => 'angle_1'
]));
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);
$results = json_decode($response, true);

// Insert into MySQL
$conn = new mysqli($db_host, $db_user, $db_pass, $db_name);
$image_ts = date('Y-m-d H:i:s');

foreach ($results['desks'] as $desk => $data) {
    $stmt = $conn->prepare(
        "INSERT INTO detection_logs (image_timestamp, person, desk, occupied, confidence, angle) VALUES (?, ?, ?, ?, ?, ?)"
    );
    $occupied = $data['occupied'] ? 1 : 0;
    $stmt->bind_param("sssids",
        $image_ts,
        $data['person'],
        $desk,
        $occupied,
        $data['confidence'],
        $results['angle']
    );
    $stmt->execute();
}
$conn->close();
```

### 2. Display 5 indicators — query MySQL directly in PHP

```php
// Get stats for today
$person = 'Asep';
$today = date('Y-m-d');

$result = $conn->query("
    SELECT * FROM detection_logs
    WHERE person = '$person' AND DATE(image_timestamp) = '$today'
    ORDER BY image_timestamp ASC
");

// Compute indicators in PHP (or call FastAPI /stats/{person})
// ... then display in dashboard
```

## Deployment

### VPS (Ubuntu/Debian)

```bash
# SSH to your VPS
git clone <your-repo> /opt/detect-api
cd /opt/detect-api
pip install -r requirements.txt
cp -r /path/to/models .
cp zones.json .

# Run
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 > api.log 2>&1 &
```

## Architecture

```
ESP32-CAM → PHP (upload) → POST /detect → FastAPI → PyTorch inference
                            ↓
                      PHP writes detection_logs → MySQL
                            ↓
PHP Dashboard → GET /stats/{person} → FastAPI → MySQL → 5 indicators
```

## Project Structure

```
detect-api/
├── main.py              # FastAPI app (inference + stats endpoints)
├── model_loader.py      # Load & cache PyTorch models per angle
├── pytorch_inference.py # Crop → preprocess → inference
├── angle_detector.py    # SSIM-based angle auto-detection
├── schemas.py           # Inference Pydantic schemas
├── stats_schemas.py     # Stats Pydantic schemas
├── stats_computer.py    # Compute 5 indicators from MySQL
├── config.py            # zones.json + env + DB config
├── .env                 # Configuration
├── requirements.txt     # Dependencies
├── sql/schema.sql       # MySQL table definitions
└── README.md
```
````

- [ ] **Step 2: Commit**

```bash
git add detect-api/README.md
git commit -m "docs(detect-api): add README with inference + stats usage"
```

---

## Self-Review Checklist

- [x] Spec coverage: all components covered (config, schema, inference, stats)
- [x] Placeholder scan: no TBD/TODO found
- [x] Type consistency: `Zone` dataclass fields match `zones.json` keys; stats response fields match spec
- [x] No ONNX references — uses PyTorch `.pth` directly
- [x] SSIM angle detection logic matches `detect.py` implementation
- [x] Model architecture matches `detect.py` MobileNetV2Classifier exactly
- [x] 5 indicators implemented: Tingkat Kehadiran, Ketepatan Datang, Lama Bekerja, Waktu Produktif, Waktu Tidak Produktif
- [x] Work hours: 07:00–17:00 hardcoded in `stats_computer.py`
- [x] DB connection via `mysql.connector` with config from `config.py`
- [x] Option A (PHP writes to MySQL, FastAPI stateless for inference) — both routes supported

---

**Plan saved to:** `docs/superpowers/plans/2026-08-29-detect-api-plan.md`
