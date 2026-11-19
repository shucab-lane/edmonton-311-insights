"""Load: write cleaned frames into a SQLite warehouse."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import ROOT

SCHEMA = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def replace_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Atomically replace a table's contents (delete + bulk insert in one transaction)."""
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
        elif pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype(int)
    frame = frame.astype(object).where(frame.notna(), None)

    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    with conn:
        conn.execute(f"DELETE FROM {table}")
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            frame.itertuples(index=False, name=None),
        )
        conn.execute(
            "INSERT INTO pipeline_runs (run_at, table_name, rows_loaded) VALUES (?, ?, ?)",
            (datetime.now(UTC).isoformat(timespec="seconds"), table, len(frame)),
        )
    return len(frame)
