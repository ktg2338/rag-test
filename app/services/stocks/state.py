import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from app.core.config import settings


class SeenStore:
    """이미 처리한 DART 공시(rcept_no)를 추적하는 SQLite 저장소."""

    def __init__(self, path: str | None = None):
        path = path or os.path.join(settings.DART_STATE_DIR, "dart_seen.sqlite")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_disclosures (
                rcept_no TEXT PRIMARY KEY,
                corp_code TEXT,
                ingested_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def has(self, rcept_no: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM seen_disclosures WHERE rcept_no = ? LIMIT 1",
            (rcept_no,),
        )
        return cur.fetchone() is not None

    def mark(self, rcept_no: str, corp_code: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO seen_disclosures (rcept_no, corp_code, ingested_at) VALUES (?, ?, ?)",
            (rcept_no, corp_code, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def mark_many(self, items: Iterable[tuple[str, str]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO seen_disclosures (rcept_no, corp_code, ingested_at) VALUES (?, ?, ?)",
            [(r, c, now) for r, c in items],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
