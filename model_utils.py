"""
model_utils.py — Downloads MediaPipe task models on first use.

Models are stored in cs124h-mvp/models/ and reused on subsequent runs.
"""

import os
import urllib.request

_DIR = os.path.join(os.path.dirname(__file__), "models")

_MODELS = {
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "detector": (
        "efficientdet_lite0.tflite",
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
    ),
    "segmenter": (
        "selfie_segmenter.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    ),
}


def get_model_path(key: str) -> str:
    """Return local path to a model file, downloading it first if absent."""
    filename, url = _MODELS[key]
    path = os.path.join(_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(_DIR, exist_ok=True)
        print(f"[model_utils] Downloading {key} model → {path}")
        urllib.request.urlretrieve(url, path)
        print(f"[model_utils] {key} model ready.")
    return path
