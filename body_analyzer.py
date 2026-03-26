"""
body_analyzer.py — BodyAnalyzer class.

Pipeline per the plan:
  1. ArUco marker (or phone fallback) → pixels_per_cm
  2. MediaPipe Pose → 33 landmarks
  3. MediaPipe Segmenter → person mask
  4. Segmentation mask at landmark-defined Y rows → widths in cm
  5. Ellipse model → circumferences
  6. U.S. Navy formula → body fat %
  7. Landmark distances → skeletal ratios
"""

import math
import base64
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from model_utils import get_model_path

# Typical body depth/width ratios from anthropometric literature
_DEPTH_RATIOS = {
    "neck":     0.85,
    "chest":    0.70,
    "waist":    0.75,
    "hip":      0.70,
    "bicep":    0.90,
    "forearm":  0.85,
    "thigh":    0.80,
    "calf":     0.85,
}

_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

_MIN_VIS = 0.5


class BodyAnalyzer:
    """
    Usage:
        analyzer = BodyAnalyzer(marker_size_cm=5.0)
        result = analyzer.analyze(["front.jpg", "side.jpg"], height_cm=175, sex="male")
    """

    def __init__(self, marker_size_cm: float = 5.0):
        self.marker_size_cm = marker_size_cm
        self._pose_det  = None
        self._segmenter = None
        self._phone_det = None
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self._aruco  = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_paths: list[str],
        height_cm: float,
        sex: str,
        phone_dims: tuple[float, float] | None = None,
        visualize: bool = False,
    ) -> dict:
        """
        Args:
            image_paths: 1–8 photos (any mix of front/side/flexed — auto-classified).
            height_cm:   User height in cm (needed for Navy BF%).
            sex:         "male" or "female".
            phone_dims:  (height_cm, width_cm) of phone — used if no ArUco found.
            visualize:   Include annotated image in result.
        """
        views   = []
        warnings = []

        for path in image_paths:
            bgr = cv2.imread(path)
            if bgr is None:
                warnings.append(f"Could not load: {path}")
                continue

            pose_result = self._pose(bgr)
            if not pose_result.pose_landmarks:
                warnings.append(f"No pose detected in: {path}")
                continue

            lm  = pose_result.pose_landmarks[0]
            seg = self._segment(bgr)
            ppc, cal_method, aruco_corners = self._calibrate(bgr, phone_dims)
            pose_type = self._classify(lm)

            views.append({
                "path": path, "bgr": bgr, "lm": lm,
                "seg": seg, "ppc": ppc, "cal_method": cal_method,
                "aruco_corners": aruco_corners, "pose_type": pose_type,
            })

        if not views:
            return {"success": False, "error": "No valid poses detected in any image."}

        # Use the best pixels_per_cm across all views
        ppc = next((v["ppc"] for v in views if v["ppc"]), None)
        cal_method = next((v["cal_method"] for v in views if v["ppc"]), "none")

        if ppc is None:
            warnings.append(
                "No ArUco marker or phone detected — circumferences unavailable. "
                "Print a 5cm ArUco marker (DICT_4X4_50, ID 0) and hold it at your side."
            )

        # Group by pose type
        front_relaxed = [v for v in views if v["pose_type"] == "front_relaxed"]
        front_flexed  = [v for v in views if v["pose_type"] == "front_flexed"]
        side_views    = [v for v in views if v["pose_type"] == "side"]

        # Extract widths from each group, average for stability
        front_widths = self._avg_widths(front_relaxed, ppc, "front_relaxed")
        flexed_widths = self._avg_widths(front_flexed, ppc, "front_flexed")
        side_widths   = self._avg_widths(side_views,   ppc, "side")

        # Merge: relaxed widths are primary; flexed bicep/forearm override if present
        widths = {**front_widths}
        for k in ("bicep_l", "bicep_r", "forearm_l", "forearm_r"):
            if k in flexed_widths:
                widths[k] = flexed_widths[k]

        # Build circumferences
        circumferences = {}
        if ppc:
            depth_src = side_widths  # side view gives actual depth

            circ_map = [
                ("neck",     "neck",    "neck"),
                ("chest",    "chest",   "chest"),
                ("waist",    "waist",   "waist"),
                ("hip",      "hip",     "hip"),
                ("bicep_l",  "bicep_l", "bicep"),
                ("bicep_r",  "bicep_r", "bicep"),
                ("forearm_l","forearm_l","forearm"),
                ("forearm_r","forearm_r","forearm"),
                ("thigh_l",  "thigh_l", "thigh"),
                ("thigh_r",  "thigh_r", "thigh"),
                ("calf_l",   "calf_l",  "calf"),
                ("calf_r",   "calf_r",  "calf"),
            ]

            for circ_key, width_key, ratio_key in circ_map:
                w = widths.get(width_key)
                if w is None:
                    continue
                # Use measured side depth if available, else fixed ratio
                d_key = width_key.rstrip("_lr").rstrip("_")
                depth_w = depth_src.get(width_key) or depth_src.get(d_key)
                if depth_w:
                    circ = self._ellipse_circ(w, depth_w)
                    note = "ellipse from front+side"
                else:
                    circ = self._ellipse_circ_ratio(w, _DEPTH_RATIOS[ratio_key])
                    note = f"front only (±3–5cm)"

                is_flexed = width_key in ("bicep_l", "bicep_r") and width_key in flexed_widths
                circumferences[circ_key] = {
                    "value": round(circ, 1),
                    "unit":  "cm",
                    "note":  ("flexed " if is_flexed else "") + note,
                }

        # Navy BF%
        bf_pct = None
        if "waist" in circumferences and "neck" in circumferences:
            w_cm = circumferences["waist"]["value"]
            n_cm = circumferences["neck"]["value"]
            h_cm = circumferences["hip"]["value"] if "hip" in circumferences else None
            bf_pct = self._navy_bf(w_cm, n_cm, height_cm, sex, h_cm)

        # Skeletal ratios (from best front view)
        ratios = {}
        best_front = (front_relaxed or front_flexed or views)[0]
        if ppc:
            ratios = self._skeletal_ratios(
                best_front["lm"], best_front["bgr"].shape, ppc,
                widths.get("waist"), widths.get("hip"),
            )

        # Photo type summary
        photo_types = {v["path"]: v["pose_type"] for v in views}

        result = {
            "success":          True,
            "timestamp":        datetime.utcnow().isoformat() + "Z",
            "n_photos_used":    len(views),
            "photo_types":      photo_types,
            "calibration":      {"method": cal_method, "pixels_per_cm": round(ppc, 2) if ppc else None},
            "circumferences":   circumferences,
            "body_fat_pct":     bf_pct,
            "skeletal_ratios":  ratios,
            "warnings":         warnings,
        }

        if visualize:
            v = best_front
            ann = v["bgr"].copy()
            h, w = ann.shape[:2]
            lm = v["lm"]
            for a, b in _POSE_CONNECTIONS:
                if lm[a].visibility > _MIN_VIS and lm[b].visibility > _MIN_VIS:
                    p1 = (int(lm[a].x * w), int(lm[a].y * h))
                    p2 = (int(lm[b].x * w), int(lm[b].y * h))
                    cv2.line(ann, p1, p2, (180, 180, 180), 2)
            if v["aruco_corners"] is not None:
                cv2.aruco.drawDetectedMarkers(ann, v["aruco_corners"])
            # Overlay key measurements
            mid_sh_y = int((lm[11].y + lm[12].y) / 2 * h)
            mid_sh_x = int((lm[11].x + lm[12].x) / 2 * w)
            if "waist" in circumferences:
                cv2.putText(ann, f"Waist: {circumferences['waist']['value']}cm",
                            (10, mid_sh_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result["visualization"] = base64.b64encode(buf).decode("utf-8")

        return result

    def compare(
        self,
        images_before: list[str],
        images_after:  list[str],
        height_cm: float,
        sex: str,
        phone_dims: tuple | None = None,
    ) -> dict:
        r1 = self.analyze(images_before, height_cm, sex, phone_dims)
        r2 = self.analyze(images_after,  height_cm, sex, phone_dims)

        if not r1.get("success"):
            return {"success": False, "error": f"Before: {r1.get('error')}"}
        if not r2.get("success"):
            return {"success": False, "error": f"After: {r2.get('error')}"}

        c1, c2 = r1["circumferences"], r2["circumferences"]
        deltas = {}
        for k in c1:
            v1 = c1[k]["value"]
            v2 = c2.get(k, {}).get("value")
            if v1 and v2:
                deltas[k] = {
                    "before": v1, "after": v2, "unit": "cm",
                    "change": round(v2 - v1, 1),
                    "change_pct": round((v2 - v1) / v1 * 100, 1),
                }

        bf1 = r1.get("body_fat_pct")
        bf2 = r2.get("body_fat_pct")

        return {
            "success":      True,
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "circumferences": deltas,
            "body_fat_pct": {"before": bf1, "after": bf2,
                             "change": round(bf2 - bf1, 1) if bf1 and bf2 else None},
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pose(self, bgr):
        if self._pose_det is None:
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("pose")),
                running_mode=mp_vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._pose_det = mp_vision.PoseLandmarker.create_from_options(opts)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self._pose_det.detect(img)

    def _segment(self, bgr) -> np.ndarray | None:
        if self._segmenter is None:
            opts = mp_vision.ImageSegmenterOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("segmenter")),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_confidence_masks=True,
            )
            self._segmenter = mp_vision.ImageSegmenter.create_from_options(opts)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = self._segmenter.segment(img)
        if not result.confidence_masks:
            return None
        mask = result.confidence_masks[0].numpy_view().copy()
        # Resize to match input if needed
        h, w = bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        # Morphological closing to clean edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = (mask > 0.5).astype(np.uint8) * 255
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return (closed > 127).astype(np.float32)

    def _calibrate(self, bgr, phone_dims):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            c = corners[0][0]
            edge_px = np.mean([
                np.linalg.norm(c[0] - c[1]), np.linalg.norm(c[1] - c[2]),
                np.linalg.norm(c[2] - c[3]), np.linalg.norm(c[3] - c[0]),
            ])
            return edge_px / self.marker_size_cm, "aruco", corners

        if phone_dims:
            if self._phone_det is None:
                opts = mp_vision.ObjectDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("detector")),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    score_threshold=0.25,
                    category_allowlist=["cell phone"],
                )
                self._phone_det = mp_vision.ObjectDetector.create_from_options(opts)
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            det = self._phone_det.detect(img)
            best = max(det.detections, key=lambda d: d.categories[0].score, default=None)
            if best:
                bb = best.bounding_box
                ph_cm, pw_cm = phone_dims
                ppc = ((bb.height / ph_cm) + (bb.width / pw_cm)) / 2
                return ppc, "phone", None

        return None, "none", None

    def _classify(self, lm) -> str:
        hip_x_spread   = abs(lm[23].x - lm[24].x)
        sh_x_spread    = abs(lm[11].x - lm[12].x)
        vis_asymmetry  = abs(lm[11].visibility - lm[12].visibility)

        if hip_x_spread < 0.08 or sh_x_spread < 0.08 or vis_asymmetry > 0.4:
            return "side"

        def ang(a, b, c):
            pa = np.array([lm[a].x, lm[a].y])
            pb = np.array([lm[b].x, lm[b].y])
            pc = np.array([lm[c].x, lm[c].y])
            ba, bc = pa - pb, pc - pb
            return math.degrees(math.acos(
                np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9), -1, 1)
            ))

        l_elbow = ang(11, 13, 15)
        r_elbow = ang(12, 14, 16)
        wrist_up = abs(lm[15].y - lm[11].y) < 0.15 or abs(lm[16].y - lm[12].y) < 0.15

        if min(l_elbow, r_elbow) < 120 and wrist_up:
            return "front_flexed"

        return "front_relaxed"

    def _mask_width(self, mask, y_row, x_center=None) -> int:
        if mask is None:
            return 0
        row = mask[int(np.clip(y_row, 0, mask.shape[0] - 1))]
        body = np.where(row > 0.5)[0]
        if len(body) == 0:
            return 0
        if x_center is None:
            return int(body[-1] - body[0])
        # Contiguous region around x_center (for limb isolation)
        xc = int(np.clip(x_center, 0, len(row) - 1))
        if row[xc] <= 0.5:
            found = next((xc + d for d in range(-30, 31) if 0 <= xc+d < len(row) and row[xc+d] > 0.5), None)
            if found is None:
                return 0
            xc = found
        lo, hi = xc, xc
        while lo > 0 and row[lo - 1] > 0.5:
            lo -= 1
        while hi < len(row) - 1 and row[hi + 1] > 0.5:
            hi += 1
        return hi - lo

    def _scan(self, mask, y_center, half=10, take_min=False, x_center=None) -> int:
        h = mask.shape[0] if mask is not None else 1
        vals = [self._mask_width(mask, int(np.clip(y_center + dy, 0, h-1)), x_center)
                for dy in range(-half, half + 1)]
        vals = [v for v in vals if v > 0]
        if not vals:
            return 0
        return min(vals) if take_min else max(vals)

    def _extract_widths(self, bgr, lm, seg, ppc, pose_type) -> dict:
        H, W = bgr.shape[:2]
        def py(i): return int(lm[i].y * H)
        def px(i): return int(lm[i].x * W)
        def midy(a, b): return int((lm[a].y + lm[b].y) / 2 * H)
        def midx(a, b): return int((lm[a].x + lm[b].x) / 2 * W)

        raw = {}
        def store(k, px_w):
            if px_w > 0 and ppc:
                raw[k] = round(px_w / ppc, 2)

        store("neck",    self._mask_width(seg, int(py(0)*0.3 + midy(11,12)*0.7)))
        store("chest",   self._mask_width(seg, midy(11,12) + int((midy(23,24) - midy(11,12)) * 0.30)))
        store("waist",   self._scan(seg, midy(11,12) + int((midy(23,24)-midy(11,12))*0.55), 15, take_min=True))
        store("hip",     self._scan(seg, midy(23,24), 10, take_min=False))

        for side, sh, el, wr in [("l", 11, 13, 15), ("r", 12, 14, 16)]:
            by = midy(sh, el);  bx = midx(sh, el)
            fy = midy(el, wr);  fx = midx(el, wr)
            ty = midy(23 if side=="l" else 24, 25 if side=="l" else 26)
            tx = midx(23 if side=="l" else 24, 25 if side=="l" else 26)
            ky = midy(25 if side=="l" else 26, 27 if side=="l" else 28)
            kx = midx(25 if side=="l" else 26, 27 if side=="l" else 28)

            if pose_type == "front_flexed":
                store(f"bicep_{side}", self._scan(seg, by, 10, take_min=False, x_center=bx))
            else:
                store(f"bicep_{side}", self._mask_width(seg, by, x_center=bx))
            store(f"forearm_{side}", self._mask_width(seg, fy, x_center=fx))
            store(f"thigh_{side}",   self._scan(seg, ty, 8,  take_min=False, x_center=tx))
            store(f"calf_{side}",    self._mask_width(seg, ky, x_center=kx))

        return raw

    def _avg_widths(self, views, ppc, pose_type) -> dict:
        if not views:
            return {}
        all_w = [self._extract_widths(v["bgr"], v["lm"], v["seg"], ppc, pose_type)
                 for v in views]
        keys = set(k for w in all_w for k in w)
        return {k: round(np.mean([w[k] for w in all_w if k in w]), 2) for k in keys}

    @staticmethod
    def _ellipse_circ(width_cm, depth_cm) -> float:
        a, b = width_cm / 2, depth_cm / 2
        h = ((a - b) / (a + b)) ** 2
        return math.pi * (a + b) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))

    @staticmethod
    def _ellipse_circ_ratio(width_cm, depth_ratio) -> float:
        return BodyAnalyzer._ellipse_circ(width_cm, width_cm * depth_ratio)

    @staticmethod
    def _navy_bf(waist_cm, neck_cm, height_cm, sex, hip_cm=None):
        try:
            if sex.lower() == "male":
                return round(86.010 * math.log10(waist_cm - neck_cm)
                             - 70.041 * math.log10(height_cm) + 36.76, 1)
            if hip_cm:
                return round(163.205 * math.log10(waist_cm + hip_cm - neck_cm)
                             - 97.684 * math.log10(height_cm) - 78.387, 1)
        except (ValueError, ZeroDivisionError):
            pass
        return None

    def _skeletal_ratios(self, lm, shape, ppc, waist_w_cm, hip_w_cm) -> dict:
        H, W = shape[:2]
        def d(a, b):
            return math.sqrt((lm[a].x*W - lm[b].x*W)**2 + (lm[a].y*H - lm[b].y*H)**2) / ppc

        shoulder_cm = d(11, 12)
        torso_cm    = (d(11, 23) + d(12, 24)) / 2
        leg_cm      = (d(23, 25) + d(25, 27) + d(24, 26) + d(26, 28)) / 2
        arm_span_cm = d(15, 16)
        height_est  = (d(0, 27) + d(0, 28)) / 2

        ratios = {
            "torso_to_leg":  round(torso_cm / leg_cm, 3)    if leg_cm > 0    else None,
            "ape_index":     round(arm_span_cm / height_est, 3) if height_est > 0 else None,
        }
        if waist_w_cm and shoulder_cm > 0:
            ratios["shoulder_to_waist"] = round(shoulder_cm / waist_w_cm, 3)
        if waist_w_cm and hip_w_cm and hip_w_cm > 0:
            ratios["waist_to_hip"] = round(waist_w_cm / hip_w_cm, 3)

        return ratios
