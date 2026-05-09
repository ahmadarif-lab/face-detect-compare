---
title: Face Detect Compare
emoji: 👤
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Face Detection & Compare

Aplikasi web sederhana untuk **face detection** dan **face comparison** dengan ArcFace embedding (512-dim).

- Backend: Flask (Python)
- Frontend: HTML + vanilla JS
- Dua backend inference yang bisa dipilih lewat env var (lihat di bawah).

## Setup

```bash
# 1. buat virtualenv (Python 3.9+ direkomendasikan)
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

> **Catatan:** Saat pertama kali menjalankan endpoint, library akan men-download
> bobot model (RetinaFace ~100MB + ArcFace ~130MB untuk backend `deepface`,
> atau buffalo_l ~280MB untuk backend `onnx`). Pastikan koneksi internet tersedia.

## Pilih Backend Inference

App mendukung dua backend yang dipilih lewat env var `FACE_BACKEND`:

| Backend | Detector | Embedding | Kelebihan |
|---------|----------|-----------|-----------|
| `deepface` (default) | RetinaFace (TF) | ArcFace (TF/Keras) | Akurasi RetinaFace bagus untuk wajah kecil/multi-face |
| `onnx`               | SCRFD (ONNX)   | ArcFace (ONNX)    | Lebih cepat, footprint kecil, cocok untuk CPU & deployment |

Kedua backend sudah terinstall dari `requirements.txt` — tinggal pilih saat runtime.

## Menjalankan

### Backend `deepface` (default — RetinaFace + DeepFace)

```bash
python app.py
# atau pakai port lain:
PORT=8000 python app.py
```

### Backend `onnx` (InsightFace SCRFD + ArcFace ONNX)

```bash
FACE_BACKEND=onnx python app.py

# kombinasi env var:
FACE_BACKEND=onnx PORT=8000 python app.py
```

### Production (gunicorn)

```bash
# default backend
gunicorn --bind 0.0.0.0:7860 --workers 1 --timeout 300 app:app

# pakai onnx
FACE_BACKEND=onnx gunicorn --bind 0.0.0.0:7860 --workers 1 --timeout 300 app:app
```

> **Penting**: pakai `--workers 1`. Tiap worker me-load model ~300MB ke RAM,
> dan model di-load di import-time (warmup). Multiple workers = multiple loads.

### Dev mode (auto-reload)

```bash
FLASK_DEBUG=1 python app.py
```

Buka http://localhost:5001

> Catatan macOS: port 5000 sering bentrok dengan **AirPlay Receiver**.
> Default app sudah pakai 5001. Bisa juga matikan AirPlay Receiver di
> System Settings → General → AirDrop & Handoff.

## Environment Variables

| Var | Default | Keterangan |
|-----|---------|-----------|
| `FACE_BACKEND` | `deepface` | `deepface` atau `onnx` |
| `PORT`         | `5001`     | port HTTP |
| `FLASK_DEBUG`  | `0`        | set `1` untuk auto-reload + debugger |

## Endpoints

### `POST /api/detect`
Form-data: `image` (file)

Response:
```json
{
  "face_count": 2,
  "faces": [
    {"score": 0.99, "facial_area": [x1, y1, x2, y2], "crop": "data:image/jpeg;base64,..."}
  ],
  "image": "data:image/jpeg;base64,...",
  "image_w": 1280,
  "image_h": 960,
  "backend": "deepface",
  "processing_ms": 312
}
```

### `POST /api/compare`
Form-data: `image1` (file), `image2` (file)

Response (ringkas):
```json
{
  "backend": "deepface",
  "processing_ms": 845,
  "face_count_1": 1,
  "face_count_2": 1,
  "best_match_index_2": 0,
  "image1": "data:image/jpeg;base64,...",
  "image2": "data:image/jpeg;base64,...",
  "image1_w": 1280, "image1_h": 960,
  "image2_w": 1280, "image2_h": 960,
  "boxes_1": [[x1, y1, x2, y2]],
  "boxes_2": [[x1, y1, x2, y2]],
  "crops_1": ["data:image/jpeg;base64,..."],
  "crops_2": ["data:image/jpeg;base64,..."],
  "matches_2to1": [
    {
      "match_index": 0,
      "similarity_percent": 67.81,
      "verified": true,
      "metrics": [
        {"key": "cosine", "label": "Cosine", "distance": 0.3219, "threshold": 0.68, "verified": true, "similarity_percent": 67.81},
        {"key": "euclidean", "label": "Euclidean", "distance": 8.1234, "threshold": 4.15, "verified": false}
      ]
    }
  ],
  "matches_1to2": [0]
}
```

- `matches_2to1[i]` = best match di Foto 1 untuk wajah ke-`i` di Foto 2 (lengkap dengan multi-metric).
- `matches_1to2[i]` = index wajah di Foto 2 yang paling mirip dengan wajah ke-`i` di Foto 1 (reverse lookup untuk navigasi UI).
- `best_match_index_2` = index wajah di Foto 2 yang punya pasangan terdekat secara global.
- Verifikasi MATCH/NO MATCH pakai cosine threshold `0.68` (default DeepFace untuk ArcFace).

## Struktur

```
.
├── app.py                # backend Flask
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── uploads/              # auto-created, file dihapus setelah inference
```
