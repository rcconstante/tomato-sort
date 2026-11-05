"""Video source opening and validation."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2

from .config import UPLOAD_DIR

SourceType = Literal["upload", "file", "webcam", "rtsp"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class SourceError(ValueError):
    pass


@dataclass
class SourceSpec:
    type: SourceType
    value: str

    @property
    def is_live(self) -> bool:
        return self.type in ("webcam", "rtsp")

    def describe(self) -> str:
        if self.type == "upload":
            return Path(self.value).name.split("_", 1)[-1]
        if self.type == "webcam":
            return f"Camera {self.value}"
        return self.value


def resolve(spec: SourceSpec) -> str | int:
    if spec.type == "upload":
        path = (UPLOAD_DIR / Path(spec.value).name).resolve()
        if not path.exists():
            raise SourceError("Uploaded video not found — upload it again.")
        return str(path)
    if spec.type == "file":
        path = Path(spec.value.strip().strip('"')).expanduser()
        if not path.is_file():
            raise SourceError(f"File not found: {path}")
        if path.suffix.lower() not in VIDEO_EXT:
            raise SourceError(f"Unsupported video type '{path.suffix}'. Use one of {sorted(VIDEO_EXT)}.")
        return str(path)
    if spec.type == "webcam":
        try:
            idx = int(spec.value)
        except ValueError as e:
            raise SourceError("Webcam index must be a number (0 = default camera).") from e
        if not 0 <= idx <= 16:
            raise SourceError("Webcam index must be between 0 and 16.")
        return idx
    if spec.type == "rtsp":
        url = spec.value.strip()
        if not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise SourceError("Stream URL must start with rtsp://, rtsps://, http:// or https://")
        return url
    raise SourceError(f"Unknown source type {spec.type}")


def open_capture(spec: SourceSpec) -> cv2.VideoCapture:
    target = resolve(spec)
    if spec.type == "webcam" and sys.platform == "win32":
        cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
    elif spec.type == "rtsp":
        cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        what = {"webcam": "camera", "rtsp": "stream"}.get(spec.type, "video file")
        raise SourceError(f"Could not open {what}: {spec.describe()}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise SourceError(f"Opened {spec.describe()} but could not read a frame (invalid or empty video?)")
    if not spec.is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def validate_video_file(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SourceError("Not a readable video file.")
        ok, frame = cap.read()
        if not ok:
            raise SourceError("Video contains no readable frames.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"width": frame.shape[1], "height": frame.shape[0], "fps": round(fps, 2),
                "frames": frames, "duration_s": round(frames / fps, 2) if fps else None}
    finally:
        cap.release()
