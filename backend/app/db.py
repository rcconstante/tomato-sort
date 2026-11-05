"""SQLite storage for sessions and inspection events."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source TEXT NOT NULL,
    model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    track_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    vote_share REAL NOT NULL,
    observations INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    frame_index INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inspections_session ON inspections(session_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(inspections)")}
            if "predicted_label" not in columns:
                self._conn.execute("ALTER TABLE inspections ADD COLUMN predicted_label TEXT")
            self._conn.commit()

    def start_session(self, source_type: str, source: str, model: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (source_type, source, model, started_at) VALUES (?, ?, ?, ?)",
                (source_type, source, model, now_iso()))
            self._conn.commit()
            return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                               (now_iso(), session_id))
            self._conn.commit()

    def add_inspection(self, row: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO inspections (session_id, track_id, label, confidence, vote_share, observations,
                   decision, reason, frame_index, created_at, predicted_label)
                   VALUES (:session_id, :track_id, :label, :confidence, :vote_share, :observations,
                   :decision, :reason, :frame_index, :created_at, :predicted_label)""",
                {**row, "predicted_label": row.get("predicted_label")})
            self._conn.commit()
            return cur.lastrowid

    def inspections(self, session_id: int | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM inspections"
        args: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            args = (session_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, (*args, limit))]

    def sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.*, COUNT(i.id) AS total,
                          SUM(i.decision = 'ACCEPT') AS accepted,
                          SUM(i.decision = 'REJECT') AS rejected,
                          SUM(i.decision = 'REVIEW') AS review
                   FROM sessions s LEFT JOIN inspections i ON i.session_id = s.id
                   GROUP BY s.id ORDER BY s.id DESC LIMIT ?""", (limit,))
            return [dict(r) for r in rows]
