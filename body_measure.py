"""
body_measure.py — Body measurement analysis via MediaPipe + phone-based scale calibration.

How it works
------------
1. Detect the user's phone in the image using EfficientDet object detection.
2. Compare the phone's pixel bounding box to its known real-world dimensions
   to compute a pixels-per-inch scale factor.
3. Run MediaPipe Pose to extract body landmarks.
4. Convert landmark distances from pixels to inches using the scale factor.
5. (Optional) If a side-view image is also provided, run person segmentation
   on the side view to measure body depth at each landmark level, then compute
   ellipse-approximation circumferences for waist, hips, and chest.

Limitations
-----------
- Bicep circumference requires a dedicated flexed-arm photo (not a standard
  standing pose) because MediaPipe only gives shoulder / elbow / wrist, not
  the bicep peak. This is noted in the output.
- Circumferences are ellipse approximations (±10–20%). Accuracy improves with
  a true side-on pose rather than a 3/4 angle.
- If no phone is detected, the function falls back to scale-free ratios.

Usage
-----
    from body_measure import analyze_body, compare_bodies

    # Front only (widths in inches if phone detected, ratios otherwise)
    result = analyze_body(["front.jpg"], phone_dims=(6.12, 2.88))

    # Front + side (adds circumference estimates)
    result = analyze_body(["front.jpg", "side.jpg"], phone_dims=(6.12, 2.88))

    # Progress comparison
    delta = compare_bodies(["before_front.jpg"], ["after_front.jpg"], phone_dims=(6.12, 2.88))
"""

import base64
import math
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from model_utils import get_model_path

# ── Pose skeleton connections ──────────────────────────────────────────────────
_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

_MIN_VIS = 0.5

# Lazy-init detectors
_pose_detector    = None
_phone_detector   = None
_segmenter        = None


# ── Detector init ──────────────────────────────────────────────────────────────

def _get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("pose")),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=_MIN_VIS,
            min_pose_presence_confidence=_MIN_VIS,
        )
        _pose_detector = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_detector


def _get_phone_detector():
    global _phone_detector
    if _phone_detector is None:
        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("detector")),
            running_mode=mp_vision.RunningMode.IMAGE,
            score_threshold=0.25,
            category_allowlist=["cell phone"],
        )
        _phone_detector = mp_vision.ObjectDetector.create_from_options(options)
    return _phone_detector


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("segmenter")),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        _segmenter = mp_vision.ImageSegmenter.create_from_options(options)
    return _segmenter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mp_image(bgr: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _pt(lm, idx: int, w: int, h: int) -> np.ndarray:
    return np.array([lm[idx].x * w, lm[idx].y * h])


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ellipse_circumference(width_in: float, depth_in: float) -> float:
    """Ramanujan approximation for ellipse circumference given full width and depth."""
    a = width_in / 2
    b = depth_in / 2
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def _draw_skeleton(image: np.ndarray, landmarks, w: int, h: int) -> None:
    for a, b in _POSE_CONNECTIONS:
        if landmarks[a].visibility > _MIN_VIS and landmarks[b].visibility > _MIN_VIS:
            p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(image, p1, p2, (180, 180, 180), 2)
    for lm in landmarks:
        if lm.visibility > _MIN_VIS:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 230, 230), -1)


# ── Phone detection / scale ────────────────────────────────────────────────────

def _detect_phone_bbox(bgr: np.ndarray):
    """Return (x, y, w, h) pixel bounding box of the most-confident phone, or None."""
    result = _get_phone_detector().detect(_mp_image(bgr))
    best = None
    best_score = 0.0
    for det in result.detections:
        score = det.categories[0].score if det.categories else 0
        if score > best_score:
            best_score = score
            bb = det.bounding_box
            best = (bb.origin_x, bb.origin_y, bb.width, bb.height)
    return best


def _pixels_per_inch(bgr: np.ndarray, phone_h_in: float, phone_w_in: float):
    """
    Detect phone in image and compute a pixels-per-inch scale factor.
    Returns (ppi, bbox) or (None, None) if phone not found.
    """
    bbox = _detect_phone_bbox(bgr)
    if bbox is None:
        return None, None
    _, _, bw, bh = bbox
    ppi_h = bh / phone_h_in
    ppi_w = bw / phone_w_in
    return (ppi_h + ppi_w) / 2, bbox


# ── Side-view depth estimation ─────────────────────────────────────────────────

def _body_depth_at_y(side_bgr: np.ndarray, y_norm: float, ppi: float) -> float | None:
    """
    Segment the person in a side-view image and measure the body depth (pixels)
    at a given normalised y-position (0 = top, 1 = bottom).
    Returns depth in inches, or None.
    """
    h, w = side_bgr.shape[:2]
    result = _get_segmenter().segment(_mp_image(side_bgr))
    if not result.confidence_masks:
        return None

    mask = result.confidence_masks[0].numpy_view()  # float32 0–1, shape (H, W)
    row  = int(np.clip(y_norm * h, 0, h - 1))

    # Find leftmost and rightmost body pixels in that row
    body_cols = np.where(mask[row] > 0.5)[0]
    if len(body_cols) < 5:
        return None

    depth_px = float(body_cols[-1] - body_cols[0])
    return depth_px / ppi


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_body(
    images: list[str],
    phone_dims: tuple[float, float] | None = None,
    visualize: bool = False,
) -> dict:
    """
    Analyze body measurements from one or two photos.

    Args:
        images:      [front_path] or [front_path, side_path].
                     Front view required; side view enables circumference estimates.
        phone_dims:  (height_inches, width_inches) of the phone visible in the photo.
                     If None or phone not detected, falls back to ratio-only output.
        visualize:   If True, include a base64-encoded annotated JPEG under "visualization".

    Returns on success:
        {
            "success": True,
            "scale_calibrated": bool,
            "pixels_per_inch": float | None,
            "has_side_view": bool,
            "measurements": {
                # Always present (inches if calibrated, else None):
                "shoulder_width":       {"value": 17.2, "unit": "in"},
                "hip_width":            {"value": 14.8, "unit": "in"},
                "waist_width_est":      {"value": 13.1, "unit": "in",
                                         "note": "estimated at midpoint between shoulder and hip"},
                "torso_length":         {"value": 21.3, "unit": "in"},
                "left_arm_length":      {"value": 24.1, "unit": "in"},
                "right_arm_length":     {"value": 23.9, "unit": "in"},
                "leg_length":           {"value": 38.2, "unit": "in"},

                # Only if side view provided:
                "chest_circumference":  {"value": 38.4, "unit": "in", "note": "ellipse estimate"},
                "waist_circumference":  {"value": 32.1, "unit": "in", "note": "ellipse estimate"},
                "hip_circumference":    {"value": 39.7, "unit": "in", "note": "ellipse estimate"},
            },
            "notes": ["Bicep circumference requires a dedicated flexed-arm photo.", ...],
            "ratios": { ... },   # scale-free ratios, always present for comparison
            "confidence": 0.91,
            "timestamp": "...",
            "visualization": "<base64 JPEG>"  # only if visualize=True
        }
    """
    if not images:
        return {"success": False, "error": "No images provided."}

    front_bgr = cv2.imread(images[0])
    if front_bgr is None:
        return {"success": False, "error": f"Could not load image: {images[0]}"}

    side_bgr = None
    if len(images) > 1:
        side_bgr = cv2.imread(images[1])

    h, w = front_bgr.shape[:2]

    # ── Pose detection (front) ────────────────────────────────────────
    pose_result = _get_pose_detector().detect(_mp_image(front_bgr))
    if not pose_result.pose_landmarks:
        return {"success": False, "error": "No pose detected. Ensure full body is visible with good lighting."}

    lm = pose_result.pose_landmarks[0]
    key_idx = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    avg_conf = float(np.mean([lm[i].visibility for i in key_idx]))
    if avg_conf < _MIN_VIS:
        return {"success": False, "error": f"Pose confidence too low ({avg_conf:.2f}). Wear fitted clothing, ensure good lighting."}

    def pt(idx): return _pt(lm, idx, w, h)

    l_shoulder  = pt(11);  r_shoulder = pt(12)
    l_elbow     = pt(13);  r_elbow    = pt(14)
    l_wrist     = pt(15);  r_wrist    = pt(16)
    l_hip       = pt(23);  r_hip      = pt(24)
    l_knee      = pt(25);  r_knee     = pt(26)
    l_ankle     = pt(27);  r_ankle    = pt(28)

    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip      = (l_hip      + r_hip)      / 2
    mid_ankle    = (l_ankle    + r_ankle)    / 2
    mid_waist    = (mid_shoulder + mid_hip)  / 2   # estimated waist = midpoint

    # Pixel distances
    px = {
        "shoulder_width":  _dist(l_shoulder, r_shoulder),
        "hip_width":       _dist(l_hip,      r_hip),
        "torso_length":    _dist(mid_shoulder, mid_hip),
        "left_arm_length": _dist(l_shoulder,  l_wrist),
        "right_arm_length":_dist(r_shoulder,  r_wrist),
        "leg_length":      _dist(mid_hip,     mid_ankle),
    }

    # Scale-free ratios (always computed)
    ref = px["shoulder_width"]
    ratios = {k: round(v / ref, 4) if ref > 1e-6 else None for k, v in px.items()}
    ratios["shoulder_hip_ratio"] = round(px["shoulder_width"] / px["hip_width"], 4) if px["hip_width"] > 1e-6 else None

    # Estimated waist width pixel distance — use proportion of shoulder+hip widths
    # (rough estimate: waist ≈ 75–80% of hip width for average frame)
    px["waist_width_est"] = _dist(l_hip, r_hip) * 0.78  # conservative estimate

    # ── Scale calibration ──────────────────────────────────────────────
    ppi      = None
    phone_bb = None
    notes    = []

    if phone_dims:
        ppi, phone_bb = _pixels_per_inch(front_bgr, phone_dims[0], phone_dims[1])
        if ppi is None:
            notes.append("Phone not detected in front photo — showing ratios only. Ensure the phone is fully visible.")

    def to_in(px_val):
        if ppi and px_val is not None:
            return round(px_val / ppi, 2)
        return None

    unit = "in" if ppi else "ratio"

    measurements = {}
    labels = {
        "shoulder_width":   ("Shoulder width",  None),
        "hip_width":        ("Hip width",        None),
        "waist_width_est":  ("Waist width",      "estimated at mid-torso — use a tape measure for precision"),
        "torso_length":     ("Torso length",     None),
        "left_arm_length":  ("Left arm length",  "shoulder to wrist"),
        "right_arm_length": ("Right arm length", "shoulder to wrist"),
        "leg_length":       ("Leg length",       "hip to ankle"),
    }

    for key, (label, note) in labels.items():
        val = to_in(px[key]) if ppi else round(px[key] / ref, 4)
        entry = {"label": label, "value": val, "unit": unit}
        if note:
            entry["note"] = note
        measurements[key] = entry

    # ── Side view: circumferences ──────────────────────────────────────
    has_side = side_bgr is not None
    if has_side and ppi:
        sh, sw = side_bgr.shape[:2]

        # y-normalised positions to sample depth at
        # Use FRONT landmarks' y-norm values (same body height levels)
        y_chest  = float((lm[11].y + lm[12].y) / 2)          # shoulder/chest level
        y_waist  = float((mid_waist[1]) / h)                   # estimated waist
        y_hip    = float((lm[23].y + lm[24].y) / 2)           # hip level

        for circ_key, y_norm, width_key, label in [
            ("chest_circumference",  y_chest, "shoulder_width",  "Chest circumference"),
            ("waist_circumference",  y_waist, "waist_width_est", "Waist circumference"),
            ("hip_circumference",    y_hip,   "hip_width",       "Hip circumference"),
        ]:
            depth_in = _body_depth_at_y(side_bgr, y_norm, ppi)
            width_in = to_in(px[width_key])
            if depth_in and width_in:
                circ = round(_ellipse_circumference(width_in, depth_in), 2)
                measurements[circ_key] = {
                    "label": label,
                    "value": circ,
                    "unit":  "in",
                    "note":  "ellipse estimate from front width + side depth",
                }
            else:
                measurements[circ_key] = {
                    "label": label,
                    "value": None,
                    "unit":  "in",
                    "note":  "Could not measure depth from side view",
                }

    notes.append(
        "Bicep circumference requires a dedicated flexed-arm photo (arm raised, bicep visible from front). "
        "Standard standing pose only gives shoulder-to-wrist length."
    )

    # ── Visualize ─────────────────────────────────────────────────────
    output = {
        "success":         True,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "confidence":      round(avg_conf, 3),
        "scale_calibrated": ppi is not None,
        "pixels_per_inch": round(ppi, 2) if ppi else None,
        "has_side_view":   has_side,
        "measurements":    measurements,
        "ratios":          ratios,
        "notes":           notes,
    }

    if visualize:
        annotated = front_bgr.copy()
        _draw_skeleton(annotated, lm, w, h)

        # Measurement lines
        for p1, p2, color in [
            (l_shoulder, r_shoulder, (0, 255, 0)),
            (l_hip,      r_hip,      (0, 255, 0)),
            (mid_shoulder, mid_hip,  (255, 130, 0)),
            (l_shoulder, l_wrist,    (0, 200, 255)),
            (r_shoulder, r_wrist,    (0, 200, 255)),
        ]:
            cv2.line(annotated, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 2)

        # Phone bbox
        if phone_bb:
            x, y, bw, bh = phone_bb
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
            cv2.putText(annotated, "scale ref", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Measurement labels
        def label_at(text, pt, color=(255,255,255)):
            cv2.putText(annotated, text, (int(pt[0])+6, int(pt[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        sw_val = measurements["shoulder_width"]["value"]
        hw_val = measurements["hip_width"]["value"]
        unit_s = "in" if ppi else ""
        if sw_val: label_at(f"{sw_val}{unit_s}", (mid_shoulder + np.array([0,-12])), (0,255,0))
        if hw_val: label_at(f"{hw_val}{unit_s}", (mid_hip + np.array([0,14])),       (0,255,0))

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        output["visualization"] = base64.b64encode(buf).decode("utf-8")

    return output


_M2IN = 39.3701  # metres → inches


def estimate_view_angle(bgr: np.ndarray) -> float | None:
    """
    Estimate the horizontal body-rotation angle from a single photo.

    Returns angle in [0, 360) degrees:
        0°   = person facing camera (front)
        90°  = person's left side toward camera (right profile visible)
        180° = person's back toward camera
        270° = person's right side toward camera (left profile visible)

    Method: the shoulder vector (right_shoulder − left_shoulder) in
    MediaPipe's *world* coordinate system directly encodes the viewing
    angle via atan2(dz, dx), because world landmarks are in a
    camera-relative frame (z = depth from camera).
    """
    result = _get_pose_detector().detect(_mp_image(bgr))
    if not result.pose_world_landmarks:
        return None

    wl = result.pose_world_landmarks[0]
    dx = wl[12].x - wl[11].x   # right_shoulder.x − left_shoulder.x
    dz = wl[12].z - wl[11].z   # right_shoulder.z − left_shoulder.z (depth axis)

    return math.degrees(math.atan2(dz, dx)) % 360


def analyze_body_multiview(
    image_paths: list[str],
    phone_dims: tuple[float, float] | None = None,
    visualize: bool = False,
) -> dict:
    """
    Process N photos of the same person from arbitrary angles.

    1. Runs pose detection on every image.
    2. Estimates the horizontal viewing angle from the world-landmark
       shoulder vector (atan2 of dz / dx).
    3. Sorts photos by angle — no user input needed.
    4. Rotates each view's world landmarks back to a canonical
       front-facing frame (−θ rotation around the Y axis), then
       averages all views weighted by per-landmark visibility.
       This reduces noise and fills in landmarks that are occluded
       in any single view.
    5. Reports measurements directly in inches (world landmarks are
       in metres — no phone needed for widths/lengths).
    6. If phone_dims provided, also estimates circumferences via
       multi-view segmentation and ellipse fitting.

    Args:
        image_paths: 2–8 photos from various angles around the person.
        phone_dims:  (height_in, width_in) — only needed for circumferences.
        visualize:   Include annotated image of the nearest-front view.

    Returns same structure as analyze_body(), plus:
        "n_views":     int   — number of usable photos
        "view_angles": list  — detected angle for each usable photo
        "source":      "multi-view fusion"
    """
    per_view = []

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        result = _get_pose_detector().detect(_mp_image(bgr))
        if not result.pose_landmarks or not result.pose_world_landmarks:
            continue

        lm_2d    = result.pose_landmarks[0]
        lm_world = result.pose_world_landmarks[0]

        # Require core landmarks to be reasonably visible
        key_vis = float(np.mean([lm_2d[i].visibility for i in [11, 12, 23, 24]]))
        if key_vis < _MIN_VIS:
            continue

        angle = estimate_view_angle(bgr) or 0.0

        per_view.append({
            "path":     path,
            "bgr":      bgr,
            "angle":    angle,
            "lm_2d":    lm_2d,
            "lm_world": lm_world,
            "key_vis":  key_vis,
        })

    if not per_view:
        return {"success": False, "error": "No valid poses detected in any image."}

    # Sort by detected angle
    per_view.sort(key=lambda v: v["angle"])

    # ── Fuse world landmarks ───────────────────────────────────────────
    # Each view's world landmarks are in that camera's reference frame.
    # Rotating by +θ around Y brings them into the canonical front frame.
    accum   = np.zeros((33, 3))
    weights = np.zeros(33)

    for view in per_view:
        θ       = math.radians(view["angle"])
        cos_θ   = math.cos(θ)
        sin_θ   = math.sin(θ)

        for i in range(33):
            wl  = view["lm_world"][i]
            vis = max(0.0, float(wl.visibility))
            # Rotate around Y by +θ  (inverse of camera rotation θ)
            x_r =  wl.x * cos_θ + wl.z * sin_θ
            y_r =  wl.y
            z_r = -wl.x * sin_θ + wl.z * cos_θ
            accum[i]   += np.array([x_r, y_r, z_r]) * vis
            weights[i] += vis

    fused = np.where(weights[:, None] > 0, accum / weights[:, None], 0.0)

    # ── Measurements from fused 3D landmarks (metres → inches) ────────
    def p(i): return fused[i]
    def d(i, j): return round(float(np.linalg.norm(fused[i] - fused[j])) * _M2IN, 2)

    mid_sh    = (p(11) + p(12)) / 2
    mid_hip   = (p(23) + p(24)) / 2
    mid_ankle = (p(27) + p(28)) / 2
    mid_waist = (mid_sh + mid_hip) / 2

    def vec_in(a, b): return round(float(np.linalg.norm(a - b)) * _M2IN, 2)

    measurements = {
        "shoulder_width":   {"label": "Shoulder width",    "value": d(11, 12),                  "unit": "in"},
        "hip_width":        {"label": "Hip width",         "value": d(23, 24),                  "unit": "in"},
        "torso_length":     {"label": "Torso length",      "value": vec_in(mid_sh, mid_hip),    "unit": "in"},
        "left_arm_length":  {"label": "Left arm length",   "value": d(11, 15),                  "unit": "in", "note": "shoulder to wrist"},
        "right_arm_length": {"label": "Right arm length",  "value": d(12, 16),                  "unit": "in", "note": "shoulder to wrist"},
        "leg_length":       {"label": "Leg length",        "value": vec_in(mid_hip, mid_ankle), "unit": "in", "note": "hip to ankle"},
    }

    # ── Circumferences via multi-view segmentation ─────────────────────
    ppi   = None
    notes = []

    if phone_dims:
        for view in per_view:
            p_val, _ = _pixels_per_inch(view["bgr"], phone_dims[0], phone_dims[1])
            if p_val:
                ppi = p_val
                break
        if ppi is None:
            notes.append("Phone not detected in any photo — circumferences unavailable.")

    if ppi and len(per_view) >= 2:
        # For each body level, collect the visible body width from every view.
        # Each view is at angle θ — the width it sees is the chord length of
        # the body cross-section at that angle.
        # For an ellipse with semi-axes a (front/back) and b (left/right):
        #   w(θ) = 2 * sqrt((b·cos θ)² + (a·sin θ)²)
        # We use max(widths) ≈ 2·max(a,b) and min(widths) ≈ 2·min(a,b)
        # to estimate circumference via Ramanujan.

        for circ_key, y_fn, label in [
            ("chest_circumference",
             lambda v: (v["lm_2d"][11].y + v["lm_2d"][12].y) / 2,
             "Chest circumference"),
            ("waist_circumference",
             lambda v: (v["lm_2d"][11].y + v["lm_2d"][12].y +
                        v["lm_2d"][23].y + v["lm_2d"][24].y) / 4,
             "Waist circumference"),
            ("hip_circumference",
             lambda v: (v["lm_2d"][23].y + v["lm_2d"][24].y) / 2,
             "Hip circumference"),
        ]:
            widths = []
            for view in per_view:
                w_in = _body_depth_at_y(view["bgr"], y_fn(view), ppi)
                if w_in:
                    widths.append(w_in)

            if len(widths) >= 2:
                circ = round(_ellipse_circumference(max(widths), min(widths)), 2)
                measurements[circ_key] = {
                    "label": label, "value": circ, "unit": "in",
                    "note":  f"ellipse fit from {len(widths)} views",
                }
            elif len(widths) == 1:
                measurements[circ_key] = {
                    "label": label,
                    "value": round(math.pi * widths[0], 2),
                    "unit":  "in",
                    "note":  "single-view estimate (assumed circular)",
                }

    notes.append(
        "Bicep circumference requires a dedicated flexed-arm photo. "
        "Standard standing pose only gives shoulder-to-wrist length."
    )

    # ── Ratios (scale-free, always useful for comparison) ──────────────
    sw = d(11, 12)
    hw = d(23, 24)
    tl = vec_in(mid_sh, mid_hip)
    ratios = {
        "shoulder_hip_ratio":   round(sw / hw,  3) if hw  > 0 else None,
        "torso_shoulder_ratio": round(tl / sw,  3) if sw  > 0 else None,
    }

    # Find view closest to front for visualization
    front_view = min(per_view, key=lambda v: min(v["angle"], 360 - v["angle"]))

    output = {
        "success":          True,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "source":           "multi-view fusion",
        "n_views":          len(per_view),
        "view_angles":      [round(v["angle"], 1) for v in per_view],
        "scale_calibrated": ppi is not None,
        "measurements":     measurements,
        "ratios":           ratios,
        "notes":            notes,
    }

    if visualize:
        annotated = front_view["bgr"].copy()
        h_img, w_img = annotated.shape[:2]
        _draw_skeleton(annotated, front_view["lm_2d"], w_img, h_img)
        for i, view in enumerate(per_view):
            cv2.putText(annotated, f"View {i+1}: {view['angle']:.0f}\u00b0",
                        (10, 24 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        output["visualization"] = base64.b64encode(buf).decode("utf-8")

    return output


def compare_bodies(
    images_before: list[str],
    images_after:  list[str],
    phone_dims: tuple[float, float] | None = None,
) -> dict:
    """
    Compare measurements between two sessions.

    Returns delta report with before/after values and percentage changes.
    """
    r1 = analyze_body(images_before, phone_dims=phone_dims)
    r2 = analyze_body(images_after,  phone_dims=phone_dims)

    if not r1.get("success"):
        return {"success": False, "error": f"Before: {r1.get('error')}"}
    if not r2.get("success"):
        return {"success": False, "error": f"After: {r2.get('error')}"}

    m1, m2 = r1["measurements"], r2["measurements"]
    deltas = {}
    for key in m1:
        v1 = m1[key].get("value")
        v2 = m2.get(key, {}).get("value")
        if v1 and v2 and abs(v1) > 1e-6:
            deltas[key] = {
                "label":      m1[key].get("label", key),
                "before":     v1,
                "after":      v2,
                "unit":       m1[key].get("unit", ""),
                "change":     round(v2 - v1, 2),
                "change_pct": round((v2 - v1) / v1 * 100, 1),
            }

    return {
        "success":   True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "before":    m1,
        "after":     m2,
        "deltas":    deltas,
    }
