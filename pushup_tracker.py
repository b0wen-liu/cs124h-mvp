"""
pushup_tracker.py — Real-time pushup rep counting and form analysis.
Uses MediaPipe Pose Landmarker Tasks API (RunningMode.VIDEO).

Usage:
    import cv2
    from pushup_tracker import PushupTracker

    with PushupTracker() as tracker:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = tracker.process_frame(frame)
            cv2.imshow("Pushup Tracker", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()

Camera tip: Side view works best. Place camera at shoulder height, ~3–6 ft away.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from model_utils import get_model_path

# ── Pose skeleton connections (BlazePose 33 landmarks) ────────────────────────
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

_MIN_VIS = 0.5


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b, formed by segments b→a and b→c."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _draw_skeleton(image: np.ndarray, landmarks, w: int, h: int) -> None:
    for a, b in _POSE_CONNECTIONS:
        if landmarks[a].visibility > _MIN_VIS and landmarks[b].visibility > _MIN_VIS:
            p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(image, p1, p2, (180, 180, 180), 2)
    for lm in landmarks:
        if lm.visibility > _MIN_VIS:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 82, 255), -1)


class PushupTracker:
    """
    Stateful pushup tracker. Feed frames via process_frame().

    Thresholds
    ----------
    DOWN_ANGLE : elbow angle (°) marking the bottom of a rep  (~90°)
    UP_ANGLE   : elbow angle (°) marking the top position     (~160°)

    Form score (0–100) per frame, averaged over each completed rep:
        hip alignment  — up to 30 pts deducted for sag / pike
        depth          — up to 20 pts deducted for not going deep enough
        elbow flare    — up to 15 pts deducted for excessive flare
    """

    DOWN_ANGLE = 90.0
    UP_ANGLE   = 160.0

    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=get_model_path("pose")),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=_MIN_VIS,
            min_pose_presence_confidence=_MIN_VIS,
            min_tracking_confidence=_MIN_VIS,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._start_ns = time.monotonic_ns()

        self.rep_count            = 0
        self.state                = "up"   # "up" | "down"
        self.last_rep_form_score  = None
        self._rep_scores: list[float] = []
        self.running              = False  # must call start() before reps are counted
        self._arms_confirmed_up   = False  # must see straight arms before first rep

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one BGR video frame.

        Returns
        -------
        dict with keys:
            rep_count            : int
            state                : "up" | "down"
            elbow_angle          : float | None
            form_score           : float | None   (0–100, current frame)
            last_rep_form_score  : float | None   (averaged over last rep)
            feedback             : list[str]
            pose_detected        : bool
            annotated_frame      : np.ndarray (BGR with HUD overlay)
        """
        h, w = frame.shape[:2]

        # VIDEO mode requires a monotonically increasing timestamp in milliseconds.
        timestamp_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000

        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = self._detector.detect_for_video(mp_image, timestamp_ms)

        output = {
            "rep_count":           self.rep_count,
            "state":               self.state,
            "running":             self.running,
            "elbow_angle":         None,
            "form_score":          None,
            "last_rep_form_score": self.last_rep_form_score,
            "feedback":            [],
            "pose_detected":       False,
            "annotated_frame":     frame.copy(),
        }

        if not result.pose_landmarks:
            output["feedback"].append("No pose detected - position yourself in frame")
            self._draw_hud(output["annotated_frame"], output)
            return output

        output["pose_detected"] = True
        lm = result.pose_landmarks[0]

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Choose more-visible side for elbow angle
        left_vis  = min(lm[11].visibility, lm[13].visibility, lm[15].visibility)
        right_vis = min(lm[12].visibility, lm[14].visibility, lm[16].visibility)

        if left_vis >= right_vis:
            elbow_angle = _angle(pt(11), pt(13), pt(15))
        else:
            elbow_angle = _angle(pt(12), pt(14), pt(16))

        output["elbow_angle"] = round(elbow_angle, 1)

        mid_shoulder_raw = (pt(11) + pt(12)) / 2
        mid_ankle_raw    = (pt(27) + pt(28)) / 2

        # ── Rep state machine ──────────────────────────────────────────────────
        # Require arms confirmed straight first — prevents counting a rep just
        # by starting with elbows already bent (works for floor & wall pushups).
        if self.running:
            if elbow_angle > self.UP_ANGLE:
                self._arms_confirmed_up = True
            if self._arms_confirmed_up:
                if self.state == "up" and elbow_angle < self.DOWN_ANGLE:
                    self.state = "down"
                elif self.state == "down" and elbow_angle > self.UP_ANGLE:
                    self.state = "up"
                    self.rep_count += 1
                    if self._rep_scores:
                        self.last_rep_form_score = round(float(np.mean(self._rep_scores)), 1)
                        self._rep_scores = []

        output["state"]               = self.state
        output["rep_count"]           = self.rep_count
        output["last_rep_form_score"] = self.last_rep_form_score

        # ── Form analysis ──────────────────────────────────────────────────────
        form_score = 100.0
        feedback   = []

        mid_shoulder = mid_shoulder_raw
        mid_hip      = (pt(23) + pt(24)) / 2
        mid_ankle    = mid_ankle_raw


        # 1. Hip alignment: shoulder → hip → ankle should be colinear
        body_vec = mid_ankle - mid_shoulder
        body_len = float(np.linalg.norm(body_vec))
        if body_len > 1e-6:
            t         = np.dot(mid_hip - mid_shoulder, body_vec) / (body_len ** 2)
            projected = mid_shoulder + t * body_vec
            dev_pct   = float(np.linalg.norm(mid_hip - projected)) / body_len * 100
            if dev_pct > 8:
                form_score -= min(30, dev_pct * 1.5)
                if mid_hip[1] > projected[1] + 5:
                    feedback.append("Fix your hips - keep body straight")
                else:
                    feedback.append("Don't pike - lower your hips")

        # 2. Depth: penalise if "down" state but not deep enough
        if self.state == "down" and elbow_angle > self.DOWN_ANGLE + 20:
            form_score -= 20
            feedback.append("Go lower - aim for 90 deg elbow bend")

        # 3. Elbow flare (both arms visible)
        if left_vis > 0.4 and right_vis > 0.4:
            shoulder_w = abs(lm[11].x - lm[12].x)
            elbow_w    = abs(lm[13].x - lm[14].x)
            if shoulder_w > 1e-6 and elbow_w > shoulder_w * 1.4:
                form_score -= 15
                feedback.append("Tuck your elbows - they're flaring out")

        form_score = max(0.0, form_score)
        output["form_score"] = round(form_score, 1)
        output["feedback"]   = feedback
        self._rep_scores.append(form_score)

        # ── Draw skeleton + HUD ────────────────────────────────────────────────
        annotated = frame.copy()
        _draw_skeleton(annotated, lm, w, h)
        self._draw_hud(annotated, output)
        output["annotated_frame"] = annotated
        return output

    def start(self) -> None:
        """Begin counting reps."""
        self.reset()
        self.running = True

    def stop(self) -> None:
        """Pause counting without closing the detector."""
        self.running = False

    def reset(self) -> None:
        """Reset rep count and state (call between sets)."""
        self.rep_count            = 0
        self.state                = "up"
        self.last_rep_form_score  = None
        self._rep_scores          = []
        self._arms_confirmed_up   = False

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(frame: np.ndarray, data: dict) -> None:
        h, w   = frame.shape[:2]
        font   = cv2.FONT_HERSHEY_SIMPLEX
        orange = (0, 82, 255)    # Illini Orange #FF5200
        navy   = (75, 41, 19)    # Illini Blue  #13294B
        green  = orange          # use orange as the "good" color
        amber  = orange          # feedback text in orange
        red    = (39, 74, 232)   # deeper orange-red for bad form
        white  = (255, 255, 255)

        status_text = "[TRACKING]" if data.get("running") else "|| PAUSED - press S to start"
        status_color = green if data.get("running") else amber
        cv2.putText(frame, status_text, (12, 30), font, 0.65, status_color, 2)
        cv2.putText(frame, f"Reps: {data['rep_count']}", (12, 68), font, 1.3, green, 3)

        if data["elbow_angle"] is not None:
            cv2.putText(frame, f"Elbow: {data['elbow_angle']:.0f}", (12, 84), font, 0.9, white, 2)

        if data["form_score"] is not None:
            score = data["form_score"]
            color = white if score >= 80 else (orange if score >= 55 else red)
            cv2.putText(frame, f"Form: {score:.0f}/100", (12, 120), font, 0.9, color, 2)

        if data["last_rep_form_score"] is not None:
            cv2.putText(frame, f"Last rep: {data['last_rep_form_score']:.0f}", (12, 152), font, 0.7, white, 2)

        state_color = white if data.get("state") == "up" else orange
        cv2.putText(frame, (data.get("state") or "up").upper(), (w - 110, 44), font, 1.3, state_color, 3)

        for i, msg in enumerate(data.get("feedback", [])):
            cv2.putText(frame, msg, (12, h - 16 - i * 30), font, 0.58, amber, 2)
