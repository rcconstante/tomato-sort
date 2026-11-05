"""Inspection pipeline: read frames → YOLO + ByteTrack → line-crossing grading → sorting → annotated stream."""
from __future__ import annotations

import base64
import logging
import threading
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Settings, SettingsStore
from .db import Database, now_iso
from .grading import UNCERTAIN, Grade, TrackVotes, category, crossed, side_of_line, within_segment
from .model_manager import ModelManager
from .outlines import tomato_contour
from .ripeness import color_ripeness
from .sources import SourceSpec, open_capture

log = logging.getLogger("tomatosort.pipeline")

# BGR colours
GREEN = (70, 200, 60)
ORANGE = (0, 140, 255)
GRAY = (160, 160, 160)
LINE_COLOR = (40, 215, 255)
DECISION_COLOR = {"ACCEPT": GREEN, "REJECT": (60, 60, 230), "REVIEW": (0, 200, 255)}

TRACK_TTL_FRAMES = 150      # forget tracks not seen for this many frames
MAX_STREAM_WIDTH = 1280
RECONNECT_ATTEMPTS = 5
TRACKER_CONFIG = str(Path(__file__).with_name("bytetrack.yaml"))


def color_for(label: str, conf: float, certain_conf: float) -> tuple[int, int, int]:
    if conf < certain_conf:
        return GRAY
    cat = category(label)
    return GREEN if cat == "ripe" else ORANGE if cat == "unripe" else GRAY


@dataclass
class TrackState:
    votes: TrackVotes = field(default_factory=TrackVotes)
    prev_side: float | None = None
    last_seen: int = 0
    counted: bool = False
    decision: str | None = None
    final_label: str | None = None
    final_prediction: str | None = None
    final_confidence: float = 0.0
    outline_frame: int = -100
    outline_points: np.ndarray | None = None
    color_assisted: bool = False


class Pipeline:
    def __init__(self, models: ModelManager, db: Database, settings: SettingsStore):
        self.models = models
        self.db = db
        self.settings = settings
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._reset_requested = threading.Event()
        self.state = "idle"          # idle | running | paused | finished | stopped | error
        self.error: str | None = None
        self.source: SourceSpec | None = None
        self.source_info: dict = {}
        self.session_id: int | None = None
        self._jpeg: bytes | None = None
        self._frame_seq = 0
        self._events: deque[dict] = deque(maxlen=300)
        self._event_seq = 0
        self._clear_counters()

    # ------------------------------------------------------------------ control
    def start(self, spec: SourceSpec) -> None:
        self.stop()
        cap = open_capture(spec)            # raises SourceError with a readable message
        try:
            model = self.models.new_instance()  # raises ModelNotInstalled
        except Exception:
            cap.release()
            raise
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        with self._lock:
            self._clear_counters()
            self._events.clear()
            self.source = spec
            self.source_info = {
                "type": spec.type, "name": spec.describe(), "live": spec.is_live,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 2) if fps else None,
                "frames": None if spec.is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            }
            self.session_id = self.db.start_session(spec.type, spec.describe(), self.models.path.name)
            self.state, self.error = "running", None
            self._stop.clear()
            self._paused.clear()
            self._reset_requested.clear()
            self._thread = threading.Thread(target=self._run, args=(cap, spec, model), daemon=True, name="pipeline")
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self.state == "running":
                self._paused.set()
                self.state = "paused"

    def resume(self) -> None:
        with self._lock:
            if self.state == "paused":
                self._paused.clear()
                self.state = "running"

    def stop(self) -> None:
        thread = self._thread
        if thread and thread.is_alive():
            self._stop.set()
            self._paused.clear()
            thread.join(timeout=5)
        with self._lock:
            self._thread = None
            if self.session_id is not None:
                self.db.end_session(self.session_id)
            if self.state in ("running", "paused"):
                self.state = "stopped"

    def reset(self) -> None:
        """Clear counts and tracking. A running file source restarts from its first frame."""
        thread = self._thread
        if thread and thread.is_alive():
            self._reset_requested.set()
        else:
            with self._lock:
                self._clear_counters()
                self._events.clear()
                self._jpeg = None
                self.state, self.error, self.source, self.source_info = "idle", None, None, {}

    # ------------------------------------------------------------------ outputs
    def latest_frame(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frame_seq, self._jpeg

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    def status(self) -> dict:
        with self._lock:
            now = time.time()
            per_min = sum(1 for t in self._inspection_times if now - t <= 60)
            return {
                "state": self.state,
                "error": self.error,
                "session_id": self.session_id,
                "source": self.source_info,
                "frame_index": self.frame_index,
                "loops": self.loops,
                "fps": round(self.fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "active_tracks": self.active_tracks,
                "throughput_per_min": per_min,
                "counts": {"total": self.total, "labels": dict(self.label_counts), "decisions": dict(self.decision_counts)},
                "last_event_seq": self._event_seq,
            }

    # ------------------------------------------------------------------ internals
    def _clear_counters(self) -> None:
        self.total = 0
        self.label_counts: dict[str, int] = {}
        self.decision_counts = {"ACCEPT": 0, "REJECT": 0, "REVIEW": 0}
        self.frame_index = 0
        self.loops = 0
        self.fps = 0.0
        self.inference_ms = 0.0
        self.active_tracks = 0
        self._tracks: dict[int, TrackState] = {}
        self._inspection_times: deque[float] = deque(maxlen=2000)

    def _run(self, cap: cv2.VideoCapture, spec: SourceSpec, model) -> None:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = 1.0 / src_fps if 1 <= src_fps <= 240 else 1 / 25
        next_due = time.perf_counter()
        last_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                if self._reset_requested.is_set():
                    self._reset_requested.clear()
                    with self._lock:
                        self._clear_counters()
                        self._events.clear()
                    model = self.models.new_instance()
                    if not spec.is_live:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                if self._paused.is_set():
                    if spec.is_live:
                        cap.grab()          # keep the camera buffer fresh while paused
                    time.sleep(0.03)
                    next_due = time.perf_counter()
                    continue

                ok, frame = cap.read()
                if not ok:
                    if spec.is_live:
                        cap = self._reconnect(cap, spec)
                        if cap is None:
                            return
                        continue
                    if self.settings.get().loop_video:
                        # Replay: tracker IDs restart, so tracking state must restart too.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        model = self.models.new_instance()
                        with self._lock:
                            self._tracks.clear()
                            self.loops += 1
                        continue
                    with self._lock:
                        self.state = "finished"
                    return

                self._process(frame, model)

                now = time.perf_counter()
                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt

                if not spec.is_live and self.settings.get().realtime_playback:
                    next_due += frame_interval
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -1:
                        next_due = time.perf_counter()  # inference slower than real time: don't try to catch up
        except Exception as e:
            log.exception("pipeline crashed")
            with self._lock:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
        finally:
            cap.release()
            if self.session_id is not None:
                self.db.end_session(self.session_id)

    def _reconnect(self, cap: cv2.VideoCapture, spec: SourceSpec):
        cap.release()
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            with self._lock:
                self.error = f"Source disconnected — reconnecting ({attempt}/{RECONNECT_ATTEMPTS})"
            if self._stop.wait(min(2 ** attempt, 10)):
                return None
            try:
                new_cap = open_capture(spec)
                with self._lock:
                    self.error = None
                return new_cap
            except Exception:
                continue
        with self._lock:
            self.state, self.error = "error", f"Lost connection to {spec.describe()}"
        return None

    def _process(self, frame: np.ndarray, model) -> None:
        s = self.settings.get()
        h, w = frame.shape[:2]
        lx1, ly1, lx2, ly2 = s.line.x1 * w, s.line.y1 * h, s.line.x2 * w, s.line.y2 * h

        t0 = time.perf_counter()
        result = model.track(frame, persist=True, tracker=TRACKER_CONFIG, conf=s.detect_conf,
                             imgsz=s.imgsz, device=self.models.device, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        names = self.models.names

        draw: list[tuple] = []
        boxes = result.boxes
        with self._lock:
            self.frame_index += 1
            fi = self.frame_index
            self.inference_ms = infer_ms if self.inference_ms == 0 else 0.9 * self.inference_ms + 0.1 * infer_ms
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)
                for box, conf, cls, tid in zip(xyxy, confs, clss, ids):
                    label = names.get(int(cls), str(cls))
                    conf = float(conf)
                    hint = color_ripeness(frame, box) if s.grading_mode == "binary" else None
                    assisted = hint is not None and hint != category(label)
                    if assisted:
                        label = hint
                    if tid is None:
                        draw.append((box, None, label, conf, None))
                        continue
                    tr = self._tracks.setdefault(int(tid), TrackState())
                    tr.color_assisted = tr.color_assisted or assisted
                    tr.votes.add(label, conf, s.certain_conf)
                    tr.last_seen = fi
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    side = side_of_line(cx, cy, lx1, ly1, lx2, ly2)
                    crossed_line = (tr.prev_side is not None
                                    and within_segment(cx, cy, lx1, ly1, lx2, ly2)
                                    and crossed(tr.prev_side, side, s.line.direction))
                    confirmed = tr.votes.observations >= s.min_observations
                    if confirmed and s.grading_mode == "binary":
                        # Give an exactly tied prediction a few more observations
                        # before logging it as uncertain.
                        candidate = tr.votes.finalize(s.min_observations, s.min_vote_share,
                                                      s.certain_conf, require_certainty=False)
                        confirmed = candidate.label != UNCERTAIN or tr.votes.observations >= max(6, s.min_observations)
                    should_count = confirmed if s.count_trigger == "detection" else crossed_line
                    if not tr.counted and should_count:
                        self._inspect(int(tid), tr, frame, box, s, fi)
                    if side != 0:
                        tr.prev_side = side
                    draw.append((box, int(tid), label, conf, tr))
            for tid in [t for t, tr in self._tracks.items() if fi - tr.last_seen > TRACK_TTL_FRAMES]:
                del self._tracks[tid]
            self.active_tracks = sum(1 for tr in self._tracks.values() if tr.last_seen == fi)

        jpeg = self._render(frame, draw, s, (lx1, ly1, lx2, ly2))
        with self._lock:
            self._jpeg = jpeg
            self._frame_seq += 1

    def _inspect(self, tid: int, tr: TrackState, frame: np.ndarray, box, s: Settings, fi: int) -> None:
        grade: Grade = tr.votes.finalize(s.min_observations, s.min_vote_share, s.certain_conf,
                                        require_certainty=s.grading_mode == "strict")
        cat = UNCERTAIN if grade.label == UNCERTAIN else category(grade.label)
        decision = s.sorting.get(cat, "REVIEW")
        tr.counted, tr.decision, tr.final_label = True, decision, grade.label
        tr.final_prediction, tr.final_confidence = grade.predicted_label, grade.confidence

        row = {
            "session_id": self.session_id, "track_id": tid, "label": grade.label,
            "predicted_label": grade.predicted_label,
            "confidence": grade.confidence, "vote_share": grade.vote_share, "observations": grade.observations,
            "decision": decision,
            "reason": grade.reason or ("Color-assisted classification" if tr.color_assisted else ""),
            "frame_index": fi, "created_at": now_iso(),
        }
        row["id"] = self.db.add_inspection(row)
        self.total += 1
        self.label_counts[grade.label] = self.label_counts.get(grade.label, 0) + 1
        self.decision_counts[decision] = self.decision_counts.get(decision, 0) + 1
        self._inspection_times.append(time.time())
        self._event_seq += 1
        self._events.append({**row, "seq": self._event_seq, "number": self.total, "category": cat,
                             "votes": grade.votes, "crop": self._crop(frame, box)})

    @staticmethod
    def _crop(frame: np.ndarray, box) -> str | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = 0.15 * max(x2 - x1, y2 - y1)
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        scale = 220 / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode() if ok else None

    def _render(self, frame: np.ndarray, draw: list[tuple], s: Settings, line: tuple) -> bytes:
        out = frame.copy()
        h, w = out.shape[:2]
        thick = max(2, round(w / 640))
        font_scale = max(0.45, w / 1800)

        if s.show_boxes:
            for box, tid, label, conf, tr in draw:
                x1, y1, x2, y2 = map(int, box)
                if tr is not None and tr.votes.weights and s.grading_mode == "binary":
                    label = max(tr.votes.weights, key=tr.votes.weights.get)
                col = color_for(label, conf, s.certain_conf if s.grading_mode == "strict" else 0)
                if tr is not None and tr.counted:
                    label, conf = tr.final_prediction or tr.final_label, tr.final_confidence
                    col = DECISION_COLOR[tr.decision]
                contour = self._outline(frame, box, tr) if s.outline_mode == "contour" else None
                if contour is not None:
                    rx, ry, rw, rh = cv2.boundingRect(contour)
                    roi = out[ry:ry + rh, rx:rx + rw]
                    overlay = roi.copy()
                    cv2.fillPoly(overlay, [contour - (rx, ry)], col)
                    cv2.addWeighted(overlay, 0.12, roi, 0.88, 0, dst=roi)
                    cv2.polylines(out, [contour], True, col, thick, cv2.LINE_AA)
                else:
                    cv2.rectangle(out, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
                text = f"#{tid} {label}" if tid is not None else str(label)
                if tr is not None and tr.counted:
                    text += f" | {tr.decision}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                ty = max(th + 6, y1)
                cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
                cv2.putText(out, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)

        if s.show_line and s.count_trigger == "line":
            lx1, ly1, lx2, ly2 = map(int, line)
            cv2.line(out, (lx1, ly1), (lx2, ly2), LINE_COLOR, thick + 1, cv2.LINE_AA)

        if w > MAX_STREAM_WIDTH:
            out = cv2.resize(out, (MAX_STREAM_WIDTH, int(h * MAX_STREAM_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def _outline(self, frame: np.ndarray, box, track: TrackState | None) -> np.ndarray | None:
        origin = np.asarray(box[:2], dtype=np.float32)
        size = np.maximum(np.asarray(box[2:], dtype=np.float32) - origin, 1)
        # Refit every third processed frame; map cached relative points to the
        # current tracked box between fits to avoid repeated GrabCut overhead.
        if track is None or self.frame_index - track.outline_frame >= 3:
            points = tomato_contour(frame, box)
            if track is None:
                return points
            track.outline_frame = self.frame_index
            track.outline_points = (points.astype(np.float32) - origin) / size if points is not None else None
        if track.outline_points is None:
            return None
        points = np.rint(track.outline_points * size + origin).astype(np.int32)
        points[:, :, 0] = np.clip(points[:, :, 0], 0, frame.shape[1] - 1)
        points[:, :, 1] = np.clip(points[:, :, 1], 0, frame.shape[0] - 1)
        return points
