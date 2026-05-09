import base64
import os
import uuid
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw

from retinaface import RetinaFace
from deepface import DeepFace

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ArcFace + cosine threshold (default deepface).
ARCFACE_COSINE_THRESHOLD = 0.68

# Per-metric thresholds for ArcFace (deepface defaults; None = no canonical threshold).
ARCFACE_THRESHOLDS = {
    "cosine": 0.68,
    "euclidean": 4.15,
    "euclidean_l2": 1.13,
    "manhattan": None,
    "chebyshev": None,
}

METRIC_LABELS = {
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "euclidean_l2": "Euclidean L2",
    "manhattan": "Manhattan",
    "chebyshev": "Chebyshev",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


class FaceNotFoundError(Exception):
    pass


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_upload(file_storage) -> str:
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file_storage.save(path)
    return path


def image_to_jpeg_b64(img_path: str) -> tuple[str, int, int]:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), w, h


def crop_b64(img_path: str, bbox: list) -> str:
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2, y2))
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_all_distances(a: np.ndarray, b: np.ndarray) -> dict:
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    diff = a - b
    diff_n = a_n - b_n
    return {
        "cosine": float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
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


def get_faces_with_embeddings(img_path: str):
    """Detect every face and compute its ArcFace embedding.

    Returns (embeddings, bboxes). Empty lists if no face detected.
    """
    detections = RetinaFace.detect_faces(img_path)
    if not isinstance(detections, dict) or not detections:
        return [], []

    aligned = RetinaFace.extract_faces(img_path=img_path, align=True)

    embeddings, bboxes = [], []
    for i, face in enumerate(detections.values()):
        if i >= len(aligned):
            break
        face_rgb = aligned[i]
        # RetinaFace returns RGB; DeepFace.represent expects BGR for numpy input.
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )
        embeddings.append(np.array(rep[0]["embedding"], dtype=np.float32))
        bboxes.append([int(v) for v in face["facial_area"]])
    return embeddings, bboxes


def best_match(embs1, embs2):
    """Return (i, j, distance) for the closest pair across both lists."""
    best = None
    for i, e1 in enumerate(embs1):
        for j, e2 in enumerate(embs2):
            d = cosine_distance(e1, e2)
            if best is None or d < best[2]:
                best = (i, j, d)
    return best


def per_face_best(source_embs, target_embs):
    """For each embedding in source, find its closest in target.

    Returns list of (target_index, distance) parallel to source_embs.
    """
    results = []
    for e_src in source_embs:
        best_j, best_d = None, None
        for j, e_tgt in enumerate(target_embs):
            d = cosine_distance(e_src, e_tgt)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        results.append((best_j, best_d))
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    path = save_upload(file)
    try:
        faces = RetinaFace.detect_faces(path)
        img_b64, w, h = image_to_jpeg_b64(path)
        if not isinstance(faces, dict) or not faces:
            return jsonify({
                "face_count": 0,
                "faces": [],
                "image": f"data:image/jpeg;base64,{img_b64}",
                "image_w": w,
                "image_h": h,
            })

        face_list = []
        for key, face in faces.items():
            bbox = [int(v) for v in face["facial_area"]]
            face_list.append({
                "id": key,
                "score": round(float(face["score"]), 4),
                "facial_area": bbox,
                "crop": "data:image/jpeg;base64," + crop_b64(path, bbox),
                "landmarks": {
                    name: [float(p) for p in pt]
                    for name, pt in face["landmarks"].items()
                },
            })
        return jsonify({
            "face_count": len(face_list),
            "faces": face_list,
            "image": f"data:image/jpeg;base64,{img_b64}",
            "image_w": w,
            "image_h": h,
        })
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/api/compare", methods=["POST"])
def compare():
    f1 = request.files.get("image1")
    f2 = request.files.get("image2")
    if not f1 or not f2 or f1.filename == "" or f2.filename == "":
        return jsonify({"error": "Two images are required"}), 400
    if not (allowed_file(f1.filename) and allowed_file(f2.filename)):
        return jsonify({"error": "Unsupported file type"}), 400

    p1 = save_upload(f1)
    p2 = save_upload(f2)
    try:
        embs1, boxes1 = get_faces_with_embeddings(p1)
        embs2, boxes2 = get_faces_with_embeddings(p2)
        if not embs1:
            raise FaceNotFoundError("Tidak ada wajah terdeteksi di foto 1")
        if not embs2:
            raise FaceNotFoundError("Tidak ada wajah terdeteksi di foto 2")

        i, j, distance = best_match(embs1, embs2)
        threshold = ARCFACE_COSINE_THRESHOLD
        similarity = max(0.0, 1.0 - distance) * 100.0

        crops_1 = ["data:image/jpeg;base64," + crop_b64(p1, b) for b in boxes1]
        crops_2 = ["data:image/jpeg;base64," + crop_b64(p2, b) for b in boxes2]
        img1_b64, w1, h1 = image_to_jpeg_b64(p1)
        img2_b64, w2, h2 = image_to_jpeg_b64(p2)

        matches_2to1 = []
        for idx, (mj, md) in enumerate(per_face_best(embs2, embs1)):
            distances = compute_all_distances(embs2[idx], embs1[mj])
            matches_2to1.append({
                "face_index": idx,
                "match_index": mj,
                "distance": round(md, 4),
                "similarity_percent": round(max(0.0, 1.0 - md) * 100.0, 2),
                "verified": bool(md <= threshold),
                "metrics": build_metric_rows(distances),
            })

        matches_1to2 = [
            {
                "face_index": idx,
                "match_index": mj,
                "distance": round(md, 4),
                "similarity_percent": round(max(0.0, 1.0 - md) * 100.0, 2),
                "verified": bool(md <= threshold),
            }
            for idx, (mj, md) in enumerate(per_face_best(embs1, embs2))
        ]

        return jsonify({
            "verified": bool(distance <= threshold),
            "distance": round(distance, 4),
            "threshold": threshold,
            "similarity_percent": round(similarity, 2),
            "model": "ArcFace",
            "detector_backend": "retinaface",
            "similarity_metric": "cosine",
            "face_count_1": len(boxes1),
            "face_count_2": len(boxes2),
            "best_match_index_1": i,
            "best_match_index_2": j,
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
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Comparison failed: {e}"}), 500
    finally:
        for p in (p1, p2):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
