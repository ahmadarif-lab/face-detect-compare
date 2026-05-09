import base64
import os
import tempfile
import time
import uuid
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

# FACE_BACKEND=onnx (default) atau FACE_BACKEND=deepface
FACE_BACKEND = os.environ.get("FACE_BACKEND", "onnx").lower()

DETECTION_SIZE = (640, 640)
WARMUP_IMG_SHAPE = (112, 112, 3)
WARMUP_PIXEL_VALUE = 128
JPEG_QUALITY_FULL = 92
JPEG_QUALITY_CROP = 90
ONNX_PADDING_RATIO = 0.5  # 50% per side fallback for close-up faces
NORM_EPS = 1e-12

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Per-metric thresholds — deepface values are officially published by serengil/deepface.
# InsightFace does not publish thresholds; 0.50 is the widely-cited community value
# (stricter than deepface's 0.68 because InsightFace embeddings sit in a tighter cluster).
_ARCFACE_THRESHOLDS_DEEPFACE = {
    "cosine": 0.68,
    "euclidean": 4.15,
    "euclidean_l2": 1.13,
    "manhattan": None,
    "chebyshev": None,
}
_ARCFACE_THRESHOLDS_ONNX = {
    "cosine": 0.50,  # InsightFace community best-practice
    "euclidean": None,
    "euclidean_l2": None,
    "manhattan": None,
    "chebyshev": None,
}

ARCFACE_THRESHOLDS = (
    _ARCFACE_THRESHOLDS_DEEPFACE if FACE_BACKEND == "deepface"
    else _ARCFACE_THRESHOLDS_ONNX
)
ARCFACE_COSINE_THRESHOLD = ARCFACE_THRESHOLDS["cosine"]

METRIC_LABELS = {
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "euclidean_l2": "Euclidean L2",
    "manhattan": "Manhattan",
    "chebyshev": "Chebyshev",
}

# ---------------------------------------------------------------------------
# Backend init
# ---------------------------------------------------------------------------

if FACE_BACKEND == "onnx":
    from insightface.app import FaceAnalysis as _FaceAnalysis
    # buffalo_l bundles 5 models (detection, recognition, 2 landmark, genderage);
    # we only use detection + recognition. Skipping the rest cuts per-face work
    # from 5 inferences to 2 and reduces RAM by ~150 MB.
    _insight_app = _FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    _insight_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
else:
    from retinaface import RetinaFace
    from deepface import DeepFace

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


class FaceNotFoundError(Exception):
    pass


def _warmup_models():
    """Load model weights at startup so the first request is not slow."""
    print(f"[startup] backend={FACE_BACKEND} — loading models...", flush=True)
    if FACE_BACKEND == "onnx":
        # InsightFace already initialized at module level; one dummy inference
        # forces ONNX graph fully into memory.
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _insight_app.get(dummy)
    else:
        dummy = np.full(WARMUP_IMG_SHAPE, WARMUP_PIXEL_VALUE, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            dummy_path = os.path.join(td, "warmup.jpg")
            Image.fromarray(dummy).save(dummy_path, format="JPEG")
            try:
                RetinaFace.detect_faces(dummy_path)
                DeepFace.represent(img_path=dummy, model_name="ArcFace",
                                   enforce_detection=False, detector_backend="skip")
            except Exception as e:
                print(f"[startup] warmup non-fatal error: {e}", flush=True)
    print("[startup] models ready.", flush=True)


_warmup_models()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_upload(file_storage) -> str:
    # UUID stem prevents path traversal regardless of input filename;
    # only the extension needs validation.
    name = file_storage.filename or ""
    ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported extension: {ext!r}")
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file_storage.save(path)
    return path


def image_to_jpeg_b64(img_path: str) -> tuple[str, int, int]:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY_FULL)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), w, h


def crop_b64(img_path: str, bbox: list) -> str:
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2, y2))
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=JPEG_QUALITY_CROP)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Distance / metric helpers
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + NORM_EPS
    return float(1.0 - np.dot(a, b) / denom)


def compute_all_distances(a: np.ndarray, b: np.ndarray) -> dict:
    a_n = a / (np.linalg.norm(a) + NORM_EPS)
    b_n = b / (np.linalg.norm(b) + NORM_EPS)
    diff = a - b
    diff_n = a_n - b_n
    return {
        "cosine": cosine_distance(a, b),
        "euclidean": float(np.linalg.norm(diff)),
        "euclidean_l2": float(np.linalg.norm(diff_n)),
        "manhattan": float(np.sum(np.abs(diff))),
        "chebyshev": float(np.max(np.abs(diff))),
    }


def build_metric_rows(distances: dict) -> list:
    rows = []
    for key in ("cosine", "euclidean", "euclidean_l2", "manhattan", "chebyshev"):
        d = distances[key]
        thr = ARCFACE_THRESHOLDS.get(key)
        row = {
            "key": key,
            "label": METRIC_LABELS[key],
            "distance": round(d, 4),
            "threshold": round(thr, 4) if thr is not None else None,
            "verified": bool(d <= thr) if thr is not None else None,
        }
        if key == "cosine":
            row["similarity_percent"] = round(max(0.0, 1.0 - d) * 100.0, 2)
        elif key == "euclidean_l2":
            # L2-normalized euclidean range is [0, 2]; map to [0, 100]%.
            row["similarity_percent"] = round(max(0.0, 1.0 - d / 2.0) * 100.0, 2)
        rows.append(row)
    return rows


def pairwise_cosine_distances(embs1: list, embs2: list) -> np.ndarray:
    """Full (len(embs1) x len(embs2)) cosine distance matrix."""
    A = np.stack(embs1).astype(np.float32)
    B = np.stack(embs2).astype(np.float32)
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + NORM_EPS)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + NORM_EPS)
    return 1.0 - A_n @ B_n.T


def best_match_from_matrix(dist_matrix: np.ndarray):
    """(row, col, distance) of the closest pair."""
    i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    return int(i), int(j), float(dist_matrix[i, j])


def per_face_best_from_matrix(dist_matrix: np.ndarray) -> list:
    """For each row, return (best_col_index, distance)."""
    if dist_matrix.size == 0:
        return []
    j = np.argmin(dist_matrix, axis=1)
    d = dist_matrix[np.arange(dist_matrix.shape[0]), j]
    return [(int(j[i]), float(d[i])) for i in range(len(j))]


# ---------------------------------------------------------------------------
# Detection / embedding
# ---------------------------------------------------------------------------

def _onnx_get_faces(img_path: str):
    """Run InsightFace with automatic padding fallback for close-up images.

    SCRFD fails when a face fills most of the frame. Adding 50% padding on each
    side shrinks the face-to-frame ratio enough for detection to succeed.
    Bboxes/kps are shifted back to original image coordinates before returning.
    """
    img = cv2.imread(img_path)
    faces = _insight_app.get(img)
    if faces:
        return faces, img

    h, w = img.shape[:2]
    ph, pw = int(h * ONNX_PADDING_RATIO), int(w * ONNX_PADDING_RATIO)
    canvas = np.zeros((h + ph * 2, w + pw * 2, 3), dtype=np.uint8)
    canvas[ph:ph + h, pw:pw + w] = img
    faces = _insight_app.get(canvas)
    for f in faces:
        # Slice indexing returns a view (in-place writable). Fancy indexing
        # would return a copy and the np.clip(..., out=...) below would no-op.
        f.bbox[0::2] -= pw  # x1, x2
        f.bbox[1::2] -= ph  # y1, y2
        np.clip(f.bbox[0::2], 0, w - 1, out=f.bbox[0::2])
        np.clip(f.bbox[1::2], 0, h - 1, out=f.bbox[1::2])
        if f.kps is not None:
            f.kps[:, 0] -= pw
            f.kps[:, 1] -= ph
    return faces, img


def detect_faces_raw(img_path: str) -> list:
    """Detect all faces. Returns list of {facial_area, score}."""
    if FACE_BACKEND == "onnx":
        faces, _ = _onnx_get_faces(img_path)
        return [
            {
                "facial_area": [int(v) for v in f.bbox],
                "score": float(f.det_score),
            }
            for f in faces
        ]

    detections = RetinaFace.detect_faces(img_path)
    if not isinstance(detections, dict) or not detections:
        return []
    return [
        {
            "facial_area": [int(v) for v in face["facial_area"]],
            "score": float(face["score"]),
        }
        for face in detections.values()
    ]


def get_faces_with_embeddings(img_path: str):
    """Detect every face and compute its ArcFace embedding.

    Returns (embeddings, bboxes). Empty lists if no face detected.
    """
    if FACE_BACKEND == "onnx":
        faces, _ = _onnx_get_faces(img_path)
        if not faces:
            return [], []
        embeddings = [np.array(f.embedding, dtype=np.float32) for f in faces]
        bboxes = [[int(v) for v in f.bbox] for f in faces]
        return embeddings, bboxes

    detections = RetinaFace.detect_faces(img_path)
    if not isinstance(detections, dict) or not detections:
        return [], []

    aligned = RetinaFace.extract_faces(img_path=img_path, align=True)

    embeddings, bboxes = [], []
    for i, face in enumerate(detections.values()):
        if i >= len(aligned):
            break
        # RetinaFace returns RGB; DeepFace.represent expects BGR for numpy input.
        face_bgr = cv2.cvtColor(aligned[i], cv2.COLOR_RGB2BGR)
        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )
        embeddings.append(np.array(rep[0]["embedding"], dtype=np.float32))
        bboxes.append([int(v) for v in face["facial_area"]])
    return embeddings, bboxes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    detector_name = "SCRFD" if FACE_BACKEND == "onnx" else "RetinaFace"
    return render_template(
        "index.html",
        backend=FACE_BACKEND,
        detector_name=detector_name,
    )


@app.route("/api/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Tidak ada gambar yang diunggah"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Tipe file tidak didukung"}), 400

    path = None
    try:
        path = save_upload(file)
        t0 = time.perf_counter()
        faces_raw = detect_faces_raw(path)
        processing_ms = round((time.perf_counter() - t0) * 1000)
        img_b64, w, h = image_to_jpeg_b64(path)
        if not faces_raw:
            return jsonify({
                "face_count": 0,
                "faces": [],
                "image": f"data:image/jpeg;base64,{img_b64}",
                "image_w": w,
                "image_h": h,
                "backend": FACE_BACKEND,
                "processing_ms": processing_ms,
            })

        face_list = []
        for face in faces_raw:
            bbox = face["facial_area"]
            face_list.append({
                "score": round(face["score"], 4),
                "facial_area": bbox,
                "crop": "data:image/jpeg;base64," + crop_b64(path, bbox),
            })
        return jsonify({
            "face_count": len(face_list),
            "faces": face_list,
            "image": f"data:image/jpeg;base64,{img_b64}",
            "image_w": w,
            "image_h": h,
            "backend": FACE_BACKEND,
            "processing_ms": processing_ms,
        })
    except Exception as e:
        return jsonify({"error": f"Deteksi gagal: {e}"}), 500
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.route("/api/compare", methods=["POST"])
def compare():
    f1 = request.files.get("image1")
    f2 = request.files.get("image2")
    if not f1 or not f2 or f1.filename == "" or f2.filename == "":
        return jsonify({"error": "Dua gambar diperlukan"}), 400
    if not (allowed_file(f1.filename) and allowed_file(f2.filename)):
        return jsonify({"error": "Tipe file tidak didukung"}), 400

    t0 = time.perf_counter()
    p1 = p2 = None
    try:
        p1 = save_upload(f1)
        p2 = save_upload(f2)

        embs1, boxes1 = get_faces_with_embeddings(p1)
        embs2, boxes2 = get_faces_with_embeddings(p2)
        if not embs1:
            raise FaceNotFoundError("Tidak ada wajah terdeteksi di foto 1")
        if not embs2:
            raise FaceNotFoundError("Tidak ada wajah terdeteksi di foto 2")

        # Compute pairwise distances once; reuse for best-match and per-face matches.
        D_12 = pairwise_cosine_distances(embs1, embs2)  # rows=img1, cols=img2
        D_21 = D_12.T

        _, best_idx_2, _ = best_match_from_matrix(D_12)
        threshold = ARCFACE_COSINE_THRESHOLD

        crops_1 = ["data:image/jpeg;base64," + crop_b64(p1, b) for b in boxes1]
        crops_2 = ["data:image/jpeg;base64," + crop_b64(p2, b) for b in boxes2]
        img1_b64, w1, h1 = image_to_jpeg_b64(p1)
        img2_b64, w2, h2 = image_to_jpeg_b64(p2)

        matches_2to1 = []
        for idx, (mj, md) in enumerate(per_face_best_from_matrix(D_21)):
            distances = compute_all_distances(embs2[idx], embs1[mj])
            matches_2to1.append({
                "match_index": mj,
                "similarity_percent": round(max(0.0, 1.0 - md) * 100.0, 2),
                "verified": bool(md <= threshold),
                "metrics": build_metric_rows(distances),
            })

        # Reverse lookup only — UI uses this to navigate from img1 click to img2 face.
        matches_1to2 = [mj for mj, _md in per_face_best_from_matrix(D_12)]

        processing_ms = round((time.perf_counter() - t0) * 1000)
        return jsonify({
            "backend": FACE_BACKEND,
            "processing_ms": processing_ms,
            "face_count_1": len(boxes1),
            "face_count_2": len(boxes2),
            "best_match_index_2": best_idx_2,
            "image1": "data:image/jpeg;base64," + img1_b64,
            "image2": "data:image/jpeg;base64," + img2_b64,
            "image1_w": w1, "image1_h": h1,
            "image2_w": w2, "image2_h": h2,
            "boxes_1": boxes1,
            "boxes_2": boxes2,
            "crops_1": crops_1,
            "crops_2": crops_2,
            "matches_2to1": matches_2to1,
            "matches_1to2": matches_1to2,
        })
    except FaceNotFoundError as e:
        processing_ms = round((time.perf_counter() - t0) * 1000)
        return jsonify({"error": str(e), "processing_ms": processing_ms, "backend": FACE_BACKEND}), 400
    except Exception as e:
        return jsonify({"error": f"Perbandingan gagal: {e}"}), 500
    finally:
        for p in (p1, p2):
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
