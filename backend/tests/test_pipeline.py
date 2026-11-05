"""Drives Pipeline._process with a scripted tracker to check count-once, grading, sorting and storage."""
import numpy as np
import pytest
import cv2

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
    settings.update({"count_trigger": "line", "grading_mode": "strict", "line": {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5, "direction": "forward"}})
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
    assert by_track[1]["predicted_label"] == "ripe"
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


def test_contour_renderer_on_foreground_object(pipeline):
    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(frame, (100, 100), 45, (0, 0, 200), -1)
    jpeg = pipeline._render(frame, [([50, 50, 150, 150], 7, "ripe", 0.9, None)],
                            pipeline.settings.get(), (0, 180, 200, 180))
    rendered = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert rendered.shape == frame.shape
    # Top-left box corner remains background because a contour was selected.
    assert rendered[149, 50].max() < 30


def test_existing_database_upgrade_retains_old_records(tmp_path):
    import sqlite3
    from app.db import SCHEMA
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.execute("INSERT INTO inspections(session_id,track_id,label,confidence,vote_share,observations,decision,frame_index,created_at) VALUES (1,1,'uncertain',0.42,1,8,'REVIEW',10,'test')")
    db = Database(path)
    assert db.inspections()[0]["label"] == "uncertain"
    assert db.inspections()[0]["predicted_label"] is None


def test_low_confidence_agreement_does_not_force_accept(pipeline):
    from app.pipeline import TrackState
    track = TrackState()
    for _ in range(8):
        track.votes.add("ripe", 0.42, 0.5)
    pipeline._inspect(9, track, np.zeros((100, 100, 3), np.uint8),
                      (10, 10, 60, 60), pipeline.settings.get(), 10)
    row = pipeline.db.inspections()[0]
    assert row["label"] == "uncertain" and row["decision"] == "REVIEW"
    assert row["predicted_label"] == "ripe" and row["vote_share"] == 1
    assert "42% / 50% required" in row["reason"]
    assert track.final_confidence == 0.42


def test_contour_cache_moves_with_track_without_refitting(pipeline, monkeypatch):
    from app.pipeline import TrackState
    calls = []
    def estimate(frame, box):
        calls.append(1)
        return np.array([[[10, 10]], [[30, 10]], [[20, 30]]], dtype=np.int32)
    monkeypatch.setattr("app.pipeline.tomato_contour", estimate)
    track = TrackState()
    frame = np.zeros((100, 100, 3), np.uint8)
    first = pipeline._outline(frame, (0, 0, 40, 40), track)
    pipeline.frame_index += 1
    second = pipeline._outline(frame, (10, 0, 50, 40), track)
    assert len(calls) == 1
    np.testing.assert_array_equal(second, first + (10, 0))
    pipeline.frame_index += 3
    pipeline._outline(frame, (10, 0, 50, 40), track)
    assert len(calls) == 2


def test_simple_mode_counts_unripe_without_confidence_rejection(pipeline):
    from app.pipeline import TrackState
    pipeline.settings.update({"grading_mode": "binary"})
    track = TrackState()
    for _ in range(4):
        track.votes.add("unripe", 0.35, 0.5)
    pipeline._inspect(12, track, np.zeros((100, 100, 3), np.uint8),
                      (10, 10, 60, 60), pipeline.settings.get(), 12)
    row = pipeline.db.inspections()[0]
    assert row["label"] == "unripe" and row["decision"] == "REJECT"
    assert pipeline.label_counts == {"unripe": 1}


def test_inspection_line_has_no_direction_arrow(pipeline):
    pipeline.settings.update({"show_boxes": False})
    jpeg = pipeline._render(np.zeros((200, 200, 3), np.uint8), [],
                            pipeline.settings.get(), (0, 100, 200, 100))
    rendered = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert rendered[100, 100].max() > 150
    assert rendered[115, 100].max() < 30


def test_simple_mode_color_assistance_is_recorded(pipeline):
    pipeline.settings.update({"grading_mode": "binary", "show_boxes": False})
    frame = np.full((200, 200, 3), (0, 180, 0), np.uint8)
    tracker = ScriptedTracker()
    for _ in range(30):
        pipeline._process(frame, tracker)
    rows = pipeline.db.inspections()
    assert len(rows) == 3
    assert all(row["label"] == "unripe" and row["decision"] == "REJECT" for row in rows)
    by_track = {row["track_id"]: row for row in rows}
    assert by_track[1]["reason"] == "Color-assisted classification"
    assert by_track[3]["reason"] == "Color-assisted classification"
    assert by_track[2]["reason"] == ""


def test_late_unripe_detection_logs_once_without_line_crossing(pipeline):
    pipeline.settings.update({"count_trigger": "detection", "grading_mode": "binary", "show_boxes": False})
    class LateTracker:
        def track(self, frame, **kwargs):
            return [_Result([([40, 130, 80, 180], 0.35, 1, 202)])]
    frame = np.zeros((200, 200, 3), np.uint8)
    tracker = LateTracker()
    pipeline._process(frame, tracker)
    assert pipeline.status()["counts"]["total"] == 0
    for _ in range(12):
        pipeline._process(frame, tracker)
    rows = pipeline.db.inspections()
    assert len(rows) == 1
    assert rows[0]["track_id"] == 202
    assert rows[0]["label"] == "unripe" and rows[0]["decision"] == "REJECT"
    assert pipeline.status()["counts"]["labels"] == {"unripe": 1}


def test_line_mode_does_not_count_stationary_downstream_detection(pipeline):
    class LateTracker:
        def track(self, frame, **kwargs):
            return [_Result([([40, 130, 80, 180], 0.8, 1, 202)])]
    for _ in range(8):
        pipeline._process(np.zeros((200, 200, 3), np.uint8), LateTracker())
    assert pipeline.db.inspections() == []
