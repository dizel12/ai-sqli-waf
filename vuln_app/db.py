# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import sqlite3


def connect(db_path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
