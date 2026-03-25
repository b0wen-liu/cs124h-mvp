"""
body_measure.py — Body proportion analysis via MediaPipe Pose Landmarker (Tasks API).

Usage:
    from body_measure import analyze_body, compare_bodies

    result = analyze_body("photo.jpg")
    # { "success": True, "ratios": { "shoulder_hip_ratio": 1.32, ... }, ... }

    delta = compare_bodies("before.jpg", "after.jpg")
    # { "deltas": { "shoulder_hip_ratio": { "before": 1.28, "after": 1.34, "change_pct": 4.69 } }, ... }
"""

import base64
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from model_utils import get_model_path

# ── Pose connections for manual skeleton drawing ───────────────────────────────
# These are the standard MediaPipe BlazePose 33-landmark connections.
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

_MIN_VISIBILITY = 0.5
_KEY_LANDMARKS  = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Lazy-initialised detector (IMAGE mode — stateless, safe to reuse).
_detector: vision.PoseLandmarker | None = None


def _get_detector() -> vision.PoseLandmarker:
    global _detector
    if _detector is None:
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path()),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=_MIN_VISIBILITY,
            min_pose_presence_confidence=_MIN_VISIBILITY,
        )
        _detector = vision.PoseLandmarker.create_from_options(options)
    return _detector


def _pt(lm, idx: int, w: int, h: int) -> np.ndarray:
    return np.array([lm[idx].x * w, lm[idx].y * h])


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _safe_ratio(num: float, den: float):
    return round(num / den, 4) if den > 1e-6 else None


def _draw_skeleton(image: np.ndarray, landmarks, w: int, h: int) -> None:
    """Draw pose skeleton on image in-place."""
    for a, b in _POSE_CONNECTIONS:
        if landmarks[a].visibility > _MIN_VISIBILITY and landmarks[b].visibility > _MIN_VISIBILITY:
            p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(image, p1, p2, (180, 180, 180), 2)
    for lm in landmarks:
        if lm.visibility > _MIN_VISIBILITY:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (0, 230, 230), -1)


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_body(image_path: str, visualize: bool = False) -> dict:
    """
    Analyze body proportions from a front-facing image.

    Args:
        image_path: Path to a JPEG or PNG image.
        visualize:  If True, include a base64-encoded annotated JPEG under "visualization".

    Returns on success:
        {
            "success": True,
            "timestamp": "<ISO-8601>",
            "confidence": 0.91,
            "ratios": {
                "shoulder_hip_ratio":     float,  # > 1 = shoulders wider than hips
                "torso_to_shoulder":      float,
                "arm_span_to_shoulder":   float,
                "leg_to_torso":           float,
                "left_arm_to_shoulder":   float,
                "right_arm_to_shoulder":  float,
            },
            "visualization": "<base64 JPEG>"   # only when visualize=True
        }

    Returns on failure:
        { "success": False, "error": "<reason>" }
    """
    # Load image via OpenCV (for pixel dimensions + optional annotation)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"success": False, "error": f"Could not load image: {image_path}"}

    h, w = img_bgr.shape[:2]

    # MediaPipe Tasks API requires an mp.Image in SRGB format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = _get_detector().detect(mp_image)

    if not result.pose_landmarks:
        return {
            "success": False,
            "error": "No pose detected. Ensure full body is visible with good lighting.",
        }

    lm = result.pose_landmarks[0]  # first (and only expected) person

    # Average visibility of key landmarks as a confidence proxy
    avg_conf = float(np.mean([lm[i].visibility for i in _KEY_LANDMARKS]))
    if avg_conf < _MIN_VISIBILITY:
        return {
            "success": False,
            "error": (
                f"Pose confidence too low ({avg_conf:.2f}). "
                "Wear fitted clothing and ensure full body is in frame."
            ),
        }

    # Key points in pixel space
    l_shoulder = _pt(lm, 11, w, h)
    r_shoulder = _pt(lm, 12, w, h)
    l_hip      = _pt(lm, 23, w, h)
    r_hip      = _pt(lm, 24, w, h)
    l_wrist    = _pt(lm, 15, w, h)
    r_wrist    = _pt(lm, 16, w, h)
    l_ankle    = _pt(lm, 27, w, h)
    r_ankle    = _pt(lm, 28, w, h)

    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip      = (l_hip      + r_hip)      / 2
    mid_ankle    = (l_ankle    + r_ankle)    / 2

    shoulder_w = _dist(l_shoulder, r_shoulder)
    hip_w      = _dist(l_hip,      r_hip)
    torso_len  = _dist(mid_shoulder, mid_hip)
    arm_span   = _dist(l_wrist,    r_wrist)
    l_arm      = _dist(l_shoulder, l_wrist)
    r_arm      = _dist(r_shoulder, r_wrist)
    leg_len    = _dist(mid_hip,    mid_ankle)

    ref = shoulder_w  # normaliser — all ratios are relative to shoulder width

    output = {
        "success":    True,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "confidence": round(avg_conf, 3),
        "ratios": {
            "shoulder_hip_ratio":    _safe_ratio(shoulder_w, hip_w),
            "torso_to_shoulder":     _safe_ratio(torso_len,  ref),
            "arm_span_to_shoulder":  _safe_ratio(arm_span,   ref),
            "leg_to_torso":          _safe_ratio(leg_len,    torso_len),
            "left_arm_to_shoulder":  _safe_ratio(l_arm,      ref),
            "right_arm_to_shoulder": _safe_ratio(r_arm,      ref),
        },
    }

    if visualize:
        annotated = img_bgr.copy()
        _draw_skeleton(annotated, lm, w, h)

        cv2.line(annotated, tuple(l_shoulder.astype(int)), tuple(r_shoulder.astype(int)), (0, 255, 0), 2)
        cv2.line(annotated, tuple(l_hip.astype(int)),      tuple(r_hip.astype(int)),      (0, 255, 0), 2)
        cv2.line(annotated, tuple(mid_shoulder.astype(int)), tuple(mid_hip.astype(int)),  (255, 100, 0), 2)

        ratios = output["ratios"]
        cv2.putText(annotated, f"S/H: {ratios['shoulder_hip_ratio']}", tuple((mid_shoulder + [6, -8]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        output["visualization"] = base64.b64encode(buf).decode("utf-8")

    return output


def compare_bodies(image_path_before: str, image_path_after: str) -> dict:
    """
    Compare body measurements between two images and return percentage deltas.

    Returns:
        {
            "success": True,
            "timestamp": "<ISO-8601>",
            "before": { ...ratios... },
            "after":  { ...ratios... },
            "deltas": {
                "shoulder_hip_ratio": { "before": 1.28, "after": 1.34, "change_pct": 4.69 },
                ...
            }
        }
    """
    r1 = analyze_body(image_path_before)
    r2 = analyze_body(image_path_after)

    if not r1.get("success"):
        return {"success": False, "error": f"Before image: {r1.get('error')}"}
    if not r2.get("success"):
        return {"success": False, "error": f"After image: {r2.get('error')}"}

    rb, ra = r1["ratios"], r2["ratios"]
    deltas = {
        k: {"before": v1, "after": ra[k], "change_pct": round((ra[k] - v1) / v1 * 100, 2)}
        for k, v1 in rb.items()
        if v1 is not None and ra.get(k) is not None and abs(v1) > 1e-6
    }

    return {
        "success":   True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "before":    rb,
        "after":     ra,
        "deltas":    deltas,
    }
