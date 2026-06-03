"""SQLite implementation of the analytics warehouse.

The schema is a small star: ``sessions`` is the dimension, ``frame_analytics``
and ``crossings`` are the fact tables, and ``session_summary`` is a
pre-aggregated roll-up for fast dashboard loads. SQLite keeps the demo
zero-config; the same DDL/queries port directly to Postgres or a cloud DW.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..domain.geometry import CrossingDirection
from ..domain.interfaces import AnalyticsRepository
from ..domain.models import (
    CrossingEvent,
    FrameAnalytics,
    ObjectClass,
    SessionSummary,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    started_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_analytics (
    session_id     TEXT NOT NULL,
    frame_index    INTEGER NOT NULL,
    ts             TEXT NOT NULL,
    person_count   INTEGER NOT NULL,
    vehicle_count  INTEGER NOT NULL,
    total          INTEGER NOT NULL,
    crossings_in   INTEGER NOT NULL,
    crossings_out  INTEGER NOT NULL,
    is_anomaly     INTEGER NOT NULL,
    anomaly_score  REAL NOT NULL,
    PRIMARY KEY (session_id, frame_index),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS crossings (
    session_id    TEXT NOT NULL,
    track_id      INTEGER NOT NULL,
    object_class  TEXT NOT NULL,
    direction     TEXT NOT NULL,
    line_name     TEXT NOT NULL,
    frame_index   INTEGER NOT NULL,
    ts            TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS session_summary (
    session_id        TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    frames_processed  INTEGER NOT NULL,
    duration_seconds  REAL NOT NULL,
    people_in         INTEGER NOT NULL,
    people_out        INTEGER NOT NULL,
    vehicles_in       INTEGER NOT NULL,
    vehicles_out      INTEGER NOT NULL,
    peak_crowd        INTEGER NOT NULL,
    peak_crowd_frame  INTEGER NOT NULL,
    anomalies         INTEGER NOT NULL,
    avg_confidence    REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_frame_session ON frame_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_cross_session ON crossings(session_id);
"""


class SQLiteAnalyticsRepository(AnalyticsRepository):
    def __init__(self, db_path: str | Path = "data/assbi.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the Streamlit dashboard caches this repo and
        # reads it from a different thread than the one that created it. Access
        # is serialised (one pipeline writer; read-only dashboard), so sharing
        # the connection across threads is safe here.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writes ------------------------------------------------------------
    def start_session(self, session_id: str, source: str) -> None:
        # A session id identifies one analysis run, so starting it is
        # idempotent: wipe any facts from a previous run before recording new
        # ones. (frame_analytics is keyed and would be overwritten, but the
        # append-only crossings table would otherwise double-count on re-runs.)
        self._conn.execute("DELETE FROM crossings WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM frame_analytics WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM session_summary WHERE session_id=?", (session_id,))
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions(session_id, source, started_at) VALUES (?,?,?)",
            (session_id, source, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def save_frame(self, session_id: str, frame: FrameAnalytics) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO frame_analytics
               (session_id, frame_index, ts, person_count, vehicle_count, total,
                crossings_in, crossings_out, is_anomaly, anomaly_score)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, frame.frame_index, frame.timestamp.isoformat(),
                frame.person_count, frame.vehicle_count, frame.total_detections,
                frame.crossings_in, frame.crossings_out,
                int(frame.is_anomaly), frame.anomaly_score,
            ),
        )

    def save_crossing(self, session_id: str, event: CrossingEvent) -> None:
        self._conn.execute(
            """INSERT INTO crossings
               (session_id, track_id, object_class, direction, line_name, frame_index, ts)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id, event.track_id, event.object_class.value,
                event.direction.value, event.line_name, event.frame_index,
                event.timestamp.isoformat(),
            ),
        )

    def save_summary(self, s: SessionSummary) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO session_summary
               (session_id, source, frames_processed, duration_seconds,
                people_in, people_out, vehicles_in, vehicles_out,
                peak_crowd, peak_crowd_frame, anomalies, avg_confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                s.session_id, s.source, s.frames_processed, s.duration_seconds,
                s.people_in, s.people_out, s.vehicles_in, s.vehicles_out,
                s.peak_crowd, s.peak_crowd_frame, s.anomalies, s.avg_confidence,
            ),
        )
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    # -- reads -------------------------------------------------------------
    def frame_series(self, session_id: str) -> list[FrameAnalytics]:
        rows = self._conn.execute(
            "SELECT * FROM frame_analytics WHERE session_id=? ORDER BY frame_index",
            (session_id,),
        ).fetchall()
        return [
            FrameAnalytics(
                frame_index=r["frame_index"],
                timestamp=datetime.fromisoformat(r["ts"]),
                person_count=r["person_count"],
                vehicle_count=r["vehicle_count"],
                total_detections=r["total"],
                crossings_in=r["crossings_in"],
                crossings_out=r["crossings_out"],
                is_anomaly=bool(r["is_anomaly"]),
                anomaly_score=r["anomaly_score"],
            )
            for r in rows
        ]

    def crossings(self, session_id: str) -> list[CrossingEvent]:
        rows = self._conn.execute(
            "SELECT * FROM crossings WHERE session_id=? ORDER BY frame_index",
            (session_id,),
        ).fetchall()
        return [
            CrossingEvent(
                track_id=r["track_id"],
                object_class=ObjectClass(r["object_class"]),
                direction=CrossingDirection(r["direction"]),
                line_name=r["line_name"],
                frame_index=r["frame_index"],
                timestamp=datetime.fromisoformat(r["ts"]),
            )
            for r in rows
        ]

    def summary(self, session_id: str) -> SessionSummary | None:
        r = self._conn.execute(
            "SELECT * FROM session_summary WHERE session_id=?", (session_id,)
        ).fetchone()
        return self._row_to_summary(r) if r else None

    def list_sessions(self) -> list[SessionSummary]:
        rows = self._conn.execute(
            "SELECT * FROM session_summary ORDER BY session_id DESC"
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    @staticmethod
    def _row_to_summary(r: sqlite3.Row) -> SessionSummary:
        return SessionSummary(
            session_id=r["session_id"], source=r["source"],
            frames_processed=r["frames_processed"], duration_seconds=r["duration_seconds"],
            people_in=r["people_in"], people_out=r["people_out"],
            vehicles_in=r["vehicles_in"], vehicles_out=r["vehicles_out"],
            peak_crowd=r["peak_crowd"], peak_crowd_frame=r["peak_crowd_frame"],
            anomalies=r["anomalies"], avg_confidence=r["avg_confidence"],
        )

    def close(self) -> None:
        self._conn.close()
