# FastAPI Inference API — Design Spec

**Date:** 2026-08-29
**Status:** Approved

## 1. Overview

Build a FastAPI backend service that exposes desk occupancy detection via HTTP endpoints. The service loads trained PyTorch MobileNetV2 classifiers directly (no ONNX conversion needed) and handles image preprocessing, angle auto-detection, and inference. PHP frontend on shared hosting calls this API via HTTP.

Detection logs are stored in MySQL by PHP (Option A — shared hosting MySQL connection stays in PHP, FastAPI remains stateless for inference). FastAPI also exposes statistics endpoints that compute 5 management indicators from stored detection logs.

## 2. Architecture

```
ESP32-CAM
    │
    ▼
PHP Frontend (iovt.my.id/upitas_mon/)
    ├── Upload gambar (existing)
    ├── POST /detect ──────────────► FastAPI Backend (VPS)
    │   response: { "angle": "...", "desks": {...} }
    │
    ├── INSERT detection_logs ──► MySQL (shared hosting)
    │
    └── Tampilkan hasil + 5 indikator (existing + new)
         │
         └── GET /stats/{person} ──► FastAPI
             response: 5 management indicators
```

## 3. File Structure

```
detect-api/
├── main.py                  # FastAPI app + endpoints
├── model_loader.py          # Load & cache PyTorch models per angle
├── pytorch_inference.py     # Crop → preprocess → inference
├── angle_detector.py        # SSIM-based angle auto-detection
├── schemas.py               # Pydantic request/response models
├── stats_schemas.py         # Pydantic schemas for stats responses
├── stats_computer.py        # Compute 5 management indicators from logs
├── config.py                # Load zones.json + env config
├── .env                     # THRESHOLD, DEVICE, MODEL_DIR, DB_CONFIG
├── requirements.txt          # Dependencies
├── sql/schema.sql           # MySQL table definitions
└── README.md                # Usage documentation
```

## 4. Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `torch` + `torchvision` | Model loading & inference |
| `python-multipart` | File upload endpoint |
| `python-dotenv` | Env config |
| `opencv-python` | Image processing, SSIM angle detection |
| `scikit-image` | SSIM metric |
| `pillow` | Image handling |
| `pydantic>=2.0.0` | Request/response validation |
| `pydantic-settings>=2.0.0` | Settings management |
| `mysql-connector-python` | MySQL connection for stats endpoint |

## 5. Endpoints

### `GET /health`
Returns service health and loaded model status.
```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": ["angle_1", "angle_2"]
}
```

### `POST /detect`
Detect occupancy from base64-encoded image.
```json
// Request
{
  "image_base64": "<base64 JPEG>",
  "angle": "angle_1",   // optional, auto-detect if omitted
  "threshold": 0.5      // optional, default from .env
}

// Response
{
  "angle": "angle_1",
  "desks": {
    "desk_a": {"occupied": true,  "confidence": 0.87},
    "desk_b": {"occupied": false, "confidence": 0.12},
    "desk_c": {"occupied": true,  "confidence": 0.91},
    "desk_d": {"occupied": false, "confidence": 0.05}
  }
}
```

### `POST /detect/file`
Detect from multipart file upload (alternative to base64).
- Form field: `file` (image), `angle` (optional)

### `GET /stats/{person}`
Returns 5 management indicators for a person on a specific date.

Query params: `date` (YYYY-MM-DD, default: today)
```json
// Response
{
  "person": "Asep",
  "date": "2026-08-29",
  "work_hours": {
    "start": "07:00",
    "end": "17:00",
    "total_minutes": 600
  },
  "indicators": {
    "tingkat_kehadiran": {
      "label": "Tingkat Kehadiran",
      "value": true,
      "description": "Hadir jika muncul >= 1x selama jam kerja"
    },
    "ketepatan_datang": {
      "label": "Ketepatan Waktu Datang",
      "value": "08:15",
      "description": "Waktu pertama muncul dalam jam kerja (HH:MM)"
    },
    "lama_bekerja": {
      "label": "Lama Berada di Kantor",
      "value": 420,
      "unit": "minutes",
      "description": "Durasi dari first_seen sampai last_seen dalam jam kerja"
    },
    "waktu_produktif": {
      "label": "Waktu Produktif",
      "value": 390,
      "unit": "minutes",
      "description": "Menit occupied selama jam kerja"
    },
    "waktu_tidak_produktif": {
      "label": "Waktu Tidak Produktif",
      "value": 30,
      "unit": "minutes",
      "description": "Menit occupied di luar jam kerja"
    }
  }
}
```

### `GET /stats/desk/{desk}`
Returns 5 indicators for a desk (by desk name instead of person name).
Query params: `date` (YYYY-MM-DD, default: today)

## 6. Inference Pipeline

```
1. Receive image (base64 or file)
2. Decode → PIL Image / numpy array
3. If angle not specified:
   a. Run SSIM angle detection against all angle zones
   b. Pick angle with highest SSIM score
4. Load cached PyTorch model for that angle
5. Crop each desk zone from image (zones from zones.json)
6. For each crop:
   a. Resize to 96x96
   b. Convert to tensor
   c. Normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   d. Run model inference
   e. Apply sigmoid, compare to threshold
7. Return structured results
```

## 7. Model Loading

- Models loaded once at startup, cached in memory
- One model per angle: `models/best_model_angle_1.pth`, `models/best_model_angle_2.pth`
- Model architecture: MobileNetV2 + custom classifier (matches `detect.py`)
- `torch.load(map_location=device, weights_only=False)`

## 8. Angle Auto-Detection

Reuse existing SSIM logic from `detect.py`:
- Build average grayscale reference crops for each angle from `dataset/sorted/{angle}/*`
- For incoming image, compare SSIM of each zone crop to reference
- Score = average SSIM across all desks for that angle
- Pick angle with highest score

## 9. Configuration

- `zones.json` — zone definitions (copied from repo root, read at startup)
- `.env`:
  - `THRESHOLD=0.5`
  - `DEVICE=cpu` (set to `cuda` on VPS with GPU)
  - `MODEL_DIR=models` (relative to project root)
  - `DB_HOST=` (MySQL host — IP or hostname of shared hosting)
  - `DB_PORT=3306`
  - `DB_NAME=` (database name)
  - `DB_USER=` (database user)
  - `DB_PASSWORD=` (database password)

## 10. Error Handling

| Scenario | Response |
|----------|----------|
| Invalid base64 | 400 Bad Request |
| Image decode failed | 400 Bad Request |
| No angle detected | 422 Unprocessable Entity |
| Model load failed | 503 Service Unavailable |
| Inference error | 500 Internal Server Error |
| DB connection failed | 503 Service Unavailable |
| Person not found | 404 Not Found |

## 11. Deployment Options

**Local (development/demo):**
```bash
cd detect-api
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000
```

**VPS (production):**
```bash
scp -r detect-api user@vps:/opt/detect-api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 12. MySQL Schema

Executed once on the shared hosting MySQL (via phpMyAdmin or CLI):

```sql
CREATE TABLE detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_timestamp DATETIME NOT NULL,
    person VARCHAR(50) NOT NULL,
    desk VARCHAR(20) NOT NULL,
    occupied BOOLEAN NOT NULL,
    confidence FLOAT NOT NULL,
    angle VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_person_date (person, image_timestamp),
    INDEX idx_desk_date (desk, image_timestamp)
);

CREATE TABLE presence_summary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    first_seen TIME,
    last_seen TIME,
    total_presence_minutes INT DEFAULT 0,
    work_hours_minutes INT DEFAULT 600,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_person_date (person, date)
);
```

## 13. PHP Integration

**Inference call** — in existing PHP files (`save_detection.php`, `check_new.php`):

```php
// 1. Call FastAPI /detect
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

// 2. Insert into MySQL
$conn = new mysqli($db_host, $db_user, $db_pass, $db_name);
$image_ts = date('Y-m-d H:i:s'); // from ESP32 filename or upload time

foreach ($results['desks'] as $desk => $data) {
    $person = $data['person'];
    $occupied = $data['occupied'] ? 1 : 0;
    $confidence = $data['confidence'];

    $stmt = $conn->prepare(
        "INSERT INTO detection_logs (image_timestamp, person, desk, occupied, confidence, angle) VALUES (?, ?, ?, ?, ?, ?)"
    );
    $stmt->bind_param("sssids", $image_ts, $person, $desk, $occupied, $confidence, $results['angle']);
    $stmt->execute();
}
$conn->close();
```

**Stats display** — query MySQL directly for 5 indicators (PHP computes locally), or call FastAPI `/stats/{person}`.

## 14. Stats Computation Logic

Work hours: **07:00 – 17:00** (600 minutes/day).

For a given person + date, query `detection_logs` for that date:

| Indicator | Logic |
|-----------|-------|
| **Tingkat Kehadiran** | `true` if any detection with `occupied=true` exists in work hours; `false` otherwise |
| **Ketepatan Datang** | Earliest `image_timestamp` where `occupied=true` and time is within 07:00–17:00. Format: "HH:MM". Returns "N/A" if not present. |
| **Lama Bekerja** | `last_seen - first_seen` (in minutes), clamped to work hours. |
| **Waktu Produktif** | Count of detections where `occupied=true` within work hours × 5 min (detection interval). |
| **Waktu Tidak Produktif** | Detection interval × number of occupied detections OUTSIDE work hours. |

## 15. Components & Responsibilities

| Component | Responsibility | Dependencies |
|-----------|----------------|--------------|
| `main.py` | FastAPI app, routing, all endpoints | all modules |
| `model_loader.py` | Load + cache PyTorch models | torch, torchvision |
| `pytorch_inference.py` | Crop, preprocess, inference | torch, torchvision, PIL, cv2 |
| `angle_detector.py` | SSIM-based angle detection | cv2, skimage |
| `schemas.py` | Inference request/response Pydantic models | pydantic |
| `stats_schemas.py` | Stats request/response Pydantic models | pydantic |
| `stats_computer.py` | Compute 5 indicators from detection logs | mysql-connector |
| `config.py` | Load zones.json, env vars, DB config | json, python-dotenv |
| `sql/schema.sql` | MySQL table definitions | — |
