"""
app.py — Flask + SocketIO API server for the CV fitness features.

Endpoints:
    GET  /api/health             — liveness check
    POST /api/body-analyze       — body ratio analysis from a photo
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

from body_measure import analyze_body, compare_bodies
from pushup_tracker import PushupTracker

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cv-fitness-dev-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# sid → PushupTracker
_trackers: dict[str, PushupTracker] = {}


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/body-analyze", methods=["POST"])
def body_analyze():
    """
    Accepts:
        multipart/form-data  with field "image" (file) and optional "visualize" ("true"/"false")
        OR application/json  with field "image_base64" (base64 string) and optional "visualize"

    Returns 200 on success, 422 on pose/confidence failure, 400 on bad request.

    Success response:
        {
            "success": true,
            "timestamp": "2024-01-01T00:00:00Z",
            "confidence": 0.91,
            "ratios": {
                "shoulder_hip_ratio":    1.32,
                "torso_to_shoulder":     1.05,
                "arm_span_to_shoulder":  2.61,
                "leg_to_torso":          1.74,
                "left_arm_to_shoulder":  1.28,
                "right_arm_to_shoulder": 1.27
            }
        }
    """
    visualize = False

    if request.is_json:
        data = request.get_json()
        if "image_base64" not in data:
            return jsonify({"error": "JSON body must include 'image_base64'"}), 400
        visualize = str(data.get("visualize", "false")).lower() == "true"
        img_bytes = base64.b64decode(data["image_base64"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image from base64"}), 400
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
        tmp.close()

    elif "image" in request.files:
        file = request.files["image"]
        visualize = request.form.get("visualize", "false").lower() == "true"
        suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(tmp.name)
        tmp_path = tmp.name
        tmp.close()

    else:
        return jsonify({"error": "Provide 'image' (multipart) or 'image_base64' (JSON)"}), 400

    try:
        result = analyze_body(tmp_path, visualize=visualize)
    finally:
        os.unlink(tmp_path)

    status = 200 if result.get("success") else 422
    return jsonify(result), status


@app.route("/api/body-compare", methods=["POST"])
def body_compare():
    """
    Accepts multipart/form-data with files "before" and "after".

    Returns delta report:
        {
            "success": true,
            "before": { ...ratios... },
            "after":  { ...ratios... },
            "deltas": {
                "shoulder_hip_ratio": { "before": 1.28, "after": 1.34, "change_pct": 4.69 },
                ...
            }
        }
    """
    if "before" not in request.files or "after" not in request.files:
        return jsonify({"error": "Provide both 'before' and 'after' image files"}), 400

    paths = {}
    try:
        for key in ("before", "after"):
            f = request.files[key]
            suffix = os.path.splitext(f.filename or "")[1] or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.save(tmp.name)
            paths[key] = tmp.name
            tmp.close()

        result = compare_bodies(paths["before"], paths["after"])
    finally:
        for p in paths.values():
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
