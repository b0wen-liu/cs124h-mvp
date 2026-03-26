"""
app.py — Flask + SocketIO API server for the CV fitness features.

Endpoints:
    GET  /api/health             — liveness check
    POST /api/body-analyze       — body measurements + BF% from 1-8 photos
    POST /api/body-compare       — before/after comparison

WebSocket namespace: /api/pushup-stream
    connect  → tracker session created
    frame    → { image_base64: str }  →  emits "result"
    reset    → resets rep count for session
    disconnect → session cleaned up

Run:
    python app.py
    # or in production:
    # gunicorn -k eventlet -w 1 app:app
"""

import base64
import os
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit

from body_analyzer import BodyAnalyzer
from pushup_tracker import PushupTracker

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cv-fitness-dev-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_analyzer = BodyAnalyzer(marker_size_cm=5.0)

# sid → PushupTracker
_trackers: dict[str, PushupTracker] = {}


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _save_upload(file) -> str:
    """Save a FileStorage to a temp file and return its path."""
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()
    return tmp.name


def _parse_phone_dims() -> tuple | None:
    """Parse phone_height and phone_width from form data, return (h, w) or None."""
    try:
        ph = float(request.form.get("phone_height", 0))
        pw = float(request.form.get("phone_width",  0))
        return (ph, pw) if ph > 0 and pw > 0 else None
    except (TypeError, ValueError):
        return None


@app.route("/api/body-analyze", methods=["POST"])
def body_analyze():
    """
    POST /api/body-analyze
    multipart/form-data fields:
        images        (files, 1-8)       — photos from any angles (auto-classified)
        height_cm     (float, required)  — user's height in cm
        sex           ("male"/"female")  — for Navy BF% formula
        phone_height  (float, optional)  — phone height in inches (fallback scale)
        phone_width   (float, optional)  — phone width in inches
        visualize     ("true"/"false")

    Returns circumferences (cm), body_fat_pct, skeletal_ratios, calibration info.
    Attach a 5 cm ArUco marker (DICT_4X4_50, ID 0) in each photo for best accuracy.
    """
    files = request.files.getlist("images")
    # also accept legacy single-file field names
    for fname in ("front_image", "side_image"):
        if fname in request.files:
            files.append(request.files[fname])
    if not files:
        return jsonify({"error": "Provide at least one image under 'images'"}), 400

    try:
        height_cm = float(request.form.get("height_cm", 0))
    except (TypeError, ValueError):
        height_cm = 0
    if height_cm <= 0:
        return jsonify({"error": "Provide 'height_cm' (positive float, e.g. 177.8)"}), 400

    sex = request.form.get("sex", "male").lower()
    if sex not in ("male", "female"):
        return jsonify({"error": "'sex' must be 'male' or 'female'"}), 400

    paths = []
    try:
        for f in files:
            paths.append(_save_upload(f))

        phone_dims = _parse_phone_dims()
        visualize  = request.form.get("visualize", "false").lower() == "true"
        result     = _analyzer.analyze(paths, height_cm=height_cm, sex=sex,
                                        phone_dims=phone_dims, visualize=visualize)
    finally:
        for p in paths:
            if os.path.exists(p):
                os.unlink(p)

    status = 200 if result.get("success") else 422
    return jsonify(result), status


@app.route("/api/body-compare", methods=["POST"])
def body_compare():
    """
    POST /api/body-compare
    multipart/form-data fields:
        before_images (files, 1-8)  — photos from the "before" session
        after_images  (files, 1-8)  — photos from the "after" session
        height_cm     (float, required)
        sex           ("male"/"female")
        phone_height, phone_width  (floats, optional)
    """
    before_files = request.files.getlist("before_images")
    after_files  = request.files.getlist("after_images")
    # legacy field names
    for fname in ("before_front", "before_side"):
        if fname in request.files:
            before_files.append(request.files[fname])
    for fname in ("after_front", "after_side"):
        if fname in request.files:
            after_files.append(request.files[fname])

    if not before_files or not after_files:
        return jsonify({"error": "Provide 'before_images' and 'after_images' files"}), 400

    try:
        height_cm = float(request.form.get("height_cm", 0))
    except (TypeError, ValueError):
        height_cm = 0
    if height_cm <= 0:
        return jsonify({"error": "Provide 'height_cm' (positive float)"}), 400

    sex = request.form.get("sex", "male").lower()
    if sex not in ("male", "female"):
        return jsonify({"error": "'sex' must be 'male' or 'female'"}), 400

    before_paths, after_paths = [], []
    try:
        for f in before_files:
            before_paths.append(_save_upload(f))
        for f in after_files:
            after_paths.append(_save_upload(f))

        phone_dims = _parse_phone_dims()
        result = _analyzer.compare(before_paths, after_paths,
                                    height_cm=height_cm, sex=sex,
                                    phone_dims=phone_dims)
    finally:
        for p in before_paths + after_paths:
            if os.path.exists(p):
                os.unlink(p)

    status = 200 if result.get("success") else 422
    return jsonify(result), status


# ── WebSocket: /api/pushup-stream ──────────────────────────────────────────────

@socketio.on("connect", namespace="/api/pushup-stream")
def on_connect():
    sid = request.sid
    _trackers[sid] = PushupTracker()
    emit("connected", {"message": "Pushup tracker ready", "session_id": sid})


@socketio.on("disconnect", namespace="/api/pushup-stream")
def on_disconnect():
    sid = request.sid
    tracker = _trackers.pop(sid, None)
    if tracker:
        tracker.close()


@socketio.on("frame", namespace="/api/pushup-stream")
def on_frame(data):
    """
    Client sends:  { "image_base64": "<base64 JPEG>" }
    Server emits:  "result" event with:
        {
            "rep_count":            int,
            "state":                "up" | "down",
            "elbow_angle":          float | null,
            "form_score":           float | null,
            "last_rep_form_score":  float | null,
            "feedback":             list[str],
            "pose_detected":        bool,
            "annotated_frame_base64": "<base64 JPEG>"
        }
    """
    sid     = request.sid
    tracker = _trackers.get(sid)
    if tracker is None:
        emit("error", {"message": "Session not found — reconnect first"})
        return

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        nparr  = np.frombuffer(img_bytes, np.uint8)
        frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            emit("error", {"message": "Could not decode frame"})
            return

        result = tracker.process_frame(frame)

        annotated = result.pop("annotated_frame", None)
        if annotated is not None:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result["annotated_frame_base64"] = base64.b64encode(buf).decode("utf-8")

        emit("result", result)

    except Exception as exc:
        emit("error", {"message": str(exc)})


@socketio.on("reset", namespace="/api/pushup-stream")
def on_reset():
    sid = request.sid
    if sid in _trackers:
        _trackers[sid].reset()
    emit("reset_ok", {"message": "Tracker reset — rep count cleared"})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CV API server starting on http://localhost:5000")
    print("  POST /api/body-analyze")
    print("  POST /api/body-compare")
    print("  WS   /api/pushup-stream")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
