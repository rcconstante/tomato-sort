"""Loads the trained best.pt. Never substitutes a different model or fake detections."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import METADATA_PATH, MODEL_PATH

MISSING_HELP = (
    "No trained model found at backend/models/best.pt. Run notebooks/TomatoSort_AI_Training.ipynb in Google Colab, "
    "download best.pt (and model_metadata.json) from the export cell, copy them into backend/models/, "
    "then press 'Reload model' or restart the server."
)


class ModelNotInstalled(RuntimeError):
    pass


class ModelManager:
    def __init__(self, path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.path = path
        self.metadata_path = metadata_path
        self._lock = threading.Lock()
        self.names: dict[int, str] = {}
        self.metadata: dict | None = None
        self.device: str | int = "cpu"
        self.error: str | None = None
        self.loaded = False

    def load(self) -> None:
        with self._lock:
            self.loaded, self.names, self.metadata, self.error = False, {}, None, None
            if not self.path.exists():
                self.error = MISSING_HELP
                return
            try:
                import torch
                from ultralytics import YOLO

                self.device = 0 if torch.cuda.is_available() else "cpu"
                model = YOLO(str(self.path))
                if model.task != "detect":
                    raise ValueError(f"best.pt is a '{model.task}' model; an object-detection model is required")
                self.names = {int(k): str(v) for k, v in model.names.items()}
            except Exception as e:  # corrupt file, wrong format, missing deps
                self.error = f"Failed to load {self.path.name}: {e}"
                return
            if self.metadata_path.exists():
                try:
                    self.metadata = json.loads(self.metadata_path.read_text())
                except json.JSONDecodeError:
                    self.metadata = None
            self.loaded = True

    def new_instance(self):
        """Fresh YOLO object per session, so ByteTrack state never leaks between runs."""
        if not self.loaded:
            raise ModelNotInstalled(self.error or MISSING_HELP)
        from ultralytics import YOLO
        return YOLO(str(self.path))

    def status(self) -> dict:
        return {
            "installed": self.loaded,
            "path": str(self.path),
            "error": self.error,
            "classes": self.names,
            "device": "cuda:0" if self.device == 0 else "cpu",
            "metadata": self.metadata,
        }
