"""Paths and runtime-editable inspection settings."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("TOMATOSORT_MODEL", BACKEND_DIR / "models" / "best.pt"))
METADATA_PATH = MODEL_PATH.with_name("model_metadata.json")
DATA_DIR = Path(os.environ.get("TOMATOSORT_DATA", BACKEND_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "inspections.db"
SETTINGS_PATH = DATA_DIR / "settings.json"

Decision = Literal["ACCEPT", "REJECT", "REVIEW"]


class InspectionLine(BaseModel):
    """Line in normalised frame coordinates (0..1)."""
    x1: float = Field(0.05, ge=0, le=1)
    y1: float = Field(0.62, ge=0, le=1)
    x2: float = Field(0.95, ge=0, le=1)
    y2: float = Field(0.62, ge=0, le=1)
    # "forward": count when crossing to the positive side of the line (for a left→right
    # horizontal line that is top → bottom of the frame); "reverse": the opposite; "both": either.
    direction: Literal["forward", "reverse", "both"] = "forward"


class Settings(BaseModel):
    detect_conf: float = Field(0.15, ge=0.01, le=0.99, description="Minimum detector confidence passed to the tracker")
    certain_conf: float = Field(0.50, ge=0.01, le=0.99, description="Below this a detection is drawn grey and its vote is down-weighted")
    min_observations: int = Field(2, ge=1, le=60, description="Frames a track must be seen before it can be graded")
    grading_mode: Literal["binary", "strict"] = "binary"
    count_trigger: Literal["detection", "line"] = "detection"
    min_vote_share: float = Field(0.65, ge=0.5, le=1.0, description="Winning class share of weighted votes needed to grade")
    imgsz: int = Field(512, ge=320, le=1280)
    line: InspectionLine = InspectionLine()
    show_boxes: bool = True
    outline_mode: Literal["contour", "box"] = "contour"
    show_line: bool = True
    loop_video: bool = True
    realtime_playback: bool = True
    sorting: dict[str, Decision] = {"ripe": "ACCEPT", "unripe": "REJECT", "uncertain": "REVIEW"}


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._settings = Settings()
        if path.exists():
            try:
                self._settings = Settings.model_validate_json(path.read_text())
            except Exception:
                pass  # corrupt or outdated file: fall back to defaults

    def get(self) -> Settings:
        with self._lock:
            return self._settings

    def update(self, patch: dict) -> Settings:
        with self._lock:
            merged = self._settings.model_dump()
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            self._settings = Settings.model_validate(merged)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._settings.model_dump(), indent=2))
            return self._settings
