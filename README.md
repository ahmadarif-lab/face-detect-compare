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

Aplikasi web sederhana untuk **face detection** (RetinaFace) dan **face comparison** (ArcFace).

- Library: [`retina-face`](https://github.com/serengil/retinaface) + [`deepface`](https://github.com/serengil/deepface) (keduanya by serengil)
- Backend: Flask (Python)
- Frontend: HTML + Bootstrap 5 + vanilla JS

## Setup

```bash
# 1. buat virtualenv (Python 3.9+ direkomendasikan)
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

> **Catatan:** Saat pertama kali menjalankan endpoint, library akan men-download
> bobot model RetinaFace (~100MB) dan ArcFace (~130MB) dari GitHub Release ke
> `~/.deepface/weights/`. Pastikan koneksi internet tersedia.

## Menjalankan

```bash
python app.py
# atau pakai port lain:
PORT=8000 python app.py
```

Buka http://localhost:5001

> Catatan macOS: port 5000 sering bentrok dengan **AirPlay Receiver**.
> Default app sudah pakai 5001. Bisa juga matikan AirPlay Receiver di
> System Settings → General → AirDrop & Handoff.

## Endpoints

### `POST /api/detect`
Form-data: `image` (file)

Response:
```json
{
  "face_count": 2,
  "faces": [
    {"id": "face_1", "score": 0.99, "facial_area": [x1,y1,x2,y2], "landmarks": {...}}
  ],
  "image": "data:image/jpeg;base64,..."
}
```

### `POST /api/compare`
Form-data: `image1` (file), `image2` (file)

Response:
```json
{
  "verified": true,
  "distance": 0.32,
  "threshold": 0.68,
  "similarity_percent": 67.81,
  "model": "ArcFace",
  "detector_backend": "retinaface",
  "similarity_metric": "cosine"
}
```

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
