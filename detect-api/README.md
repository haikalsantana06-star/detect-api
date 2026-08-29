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
