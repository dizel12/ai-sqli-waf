from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, method TEXT, path TEXT, param TEXT, value TEXT,
  scores TEXT, active_model TEXT, score REAL, threshold REAL,
  blocked INTEGER, secure_mode INTEGER
);
"""


class EventStore:
    def __init__(self, db_path: str):
        self._path = db_path
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL;")
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, *, method, path, param, value, scores, active_model,
               score, threshold, blocked, secure_mode) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(ts,method,path,param,value,scores,"
                "active_model,score,threshold,blocked,secure_mode) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), method, path, param,
                 str(value)[:200], json.dumps(scores), active_model,
                 float(score), float(threshold), int(blocked), int(secure_mode)))
            return int(cur.lastrowid)

    def _rows(self, sql: str, args: tuple) -> list[dict]:
        with self._conn() as c:
            out = []
            for r in c.execute(sql, args):
                d = dict(r)
                d["scores"] = json.loads(d["scores"]) if d["scores"] else {}
                out.append(d)
            return out

    def recent(self, limit: int = 100) -> list[dict]:
        return self._rows(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def after(self, last_id: int, limit: int = 100) -> list[dict]:
        return self._rows(
            "SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (last_id, limit))
