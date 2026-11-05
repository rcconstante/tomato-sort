"""Drives Pipeline._process with a scripted tracker to check count-once, grading, sorting and storage."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from app.config import SettingsStore
from app.db import Database
from app.pipeline import Pipeline


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, dets):
        self.xyxy = _Arr([d[0] for d in dets])
        self.conf = _Arr([d[1] for d in dets])
        self.cls = _Arr([d[2] for d in dets])
        self.id = _Arr([d[3] for d in dets])

    def __len__(self):
        return len(self.conf.numpy())


class _Result:
    def __init__(self, dets):
        self.boxes = _Boxes(dets)


class ScriptedTracker:
    """Track 1: ripe, moves down through the line. Track 2: unripe, moves down.
    Track 3: flickers between classes (should be uncertain). Track 4: moves up (wrong direction)."""

    def __init__(self):
        self.t = 0

    def track(self, frame, **_):
        self.t += 1
        y = 20 + self.t * 6          # line at y = 100 (0.5 of 200 px); crosses around t = 13-14
        dets = [
            ([10, y - 10, 30, y + 10], 0.9, 0, 1),
            ([50, y - 10, 70, y + 10], 0.85, 1, 2),
            ([90, y - 10, 110, y + 10], 0.8, self.t % 2, 3),
            ([130, 180 - self.t * 6 - 10, 150, 180 - self.t * 6 + 10], 0.9, 0, 4),
        ]
        return [_Result(dets)]


class _Models:
    names = {0: "ripe", 1: "unripe"}
    device = "cpu"


@pytest.fixture
def pipeline(tmp_path):
    settings = SettingsStore(tmp_path / "settings.json")
    settings.update({"line": {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5, "direction": "forward"}})
    p = Pipeline(_Models(), Database(tmp_path / "t.db"), settings)
    p.session_id = p.db.start_session("file", "test.mp4", "stub")
    return p


def test_each_tomato_counted_once_and_graded(pipeline):
    model = ScriptedTracker()
    frame = np.zeros((200, 200, 3), np.uint8)
    for _ in range(30):                         # tracks keep moving well past the line
        pipeline._process(frame, model)

    status = pipeline.status()
    assert status["counts"]["total"] == 3       # track 4 moves the wrong way
    assert status["counts"]["labels"] == {"ripe": 1, "unripe": 1, "uncertain": 1}
    assert status["counts"]["decisions"] == {"ACCEPT": 1, "REJECT": 1, "REVIEW": 1}

    rows = pipeline.db.inspections(pipeline.session_id)
    assert sorted(r["track_id"] for r in rows) == [1, 2, 3]
    by_track = {r["track_id"]: r for r in rows}
    assert by_track[1]["decision"] == "ACCEPT"
    assert by_track[2]["decision"] == "REJECT"
    assert by_track[3]["label"] == "uncertain" and "disagree" in by_track[3]["reason"]

    events = pipeline.events_since(0)
    assert len(events) == 3 and all(e["crop"] for e in events)
    assert pipeline.latest_frame()[1][:2] == b"\xff\xd8"   # JPEG


def test_reset_clears_counts(pipeline):
    model = ScriptedTracker()
    frame = np.zeros((200, 200, 3), np.uint8)
    for _ in range(30):
        pipeline._process(frame, model)
    pipeline.reset()        # no thread running → immediate reset to idle
    s = pipeline.status()
    assert s["counts"]["total"] == 0 and s["state"] == "idle" and pipeline.events_since(0) == []
