"""
model_utils.py — Downloads the MediaPipe Pose Landmarker model on first use.

The .task file is stored in cs124h-mvp/models/ and reused on subsequent runs.
"""

import os
import urllib.request

_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_FILE = "pose_landmarker_full.task"
MODEL_PATH = os.path.join(_DIR, _MODEL_FILE)

# Full model: best accuracy. Swap for _lite_ if CPU is too slow.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def get_model_path() -> str:
    """Return path to the model file, downloading it first if absent."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(_DIR, exist_ok=True)
        print(f"[model_utils] Downloading pose model → {MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        print("[model_utils] Download complete.")
    return MODEL_PATH
