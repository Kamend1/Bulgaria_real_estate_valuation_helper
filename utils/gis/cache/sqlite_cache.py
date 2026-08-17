"""
File-based SQLite cache for public GIS/API responses.

AGKK, data.egov.bg and municipal ArcGIS servers are shared public
infrastructure with no SLA — repeatedly re-fetching the same parcel/zone
(e.g. on every page load of the same appraisal report) is both slow and
inconsiderate. Every connector routes its raw JSON responses through this
cache before parsing, keyed by a hash of (source, endpoint, sorted params).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).parent / "gis_cache.sqlite"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 1 week — cadastral/zoning data changes rarely


class GisCache:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(source: str, endpoint: str, params: dict[str, Any]) -> str:
        canonical = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
        return f"{source}:{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"

    def get(self, source: str, endpoint: str, params: dict[str, Any]) -> dict | None:
        key = self.make_key(source, endpoint, params)
        row = self._conn.execute(
            "SELECT response_json, fetched_at FROM cache_entries WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        response_json, fetched_at = row
        if time.time() - fetched_at > self.ttl_seconds:
            return None
        return json.loads(response_json)

    def set(self, source: str, endpoint: str, params: dict[str, Any], response: dict) -> None:
        key = self.make_key(source, endpoint, params)
        self._conn.execute(
            """
            INSERT INTO cache_entries (cache_key, source, endpoint, response_json, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_json = excluded.response_json,
                fetched_at = excluded.fetched_at
            """,
            (key, source, endpoint, json.dumps(response), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_SHARED_CACHE: GisCache | None = None


def get_shared_cache() -> GisCache:
    """Process-wide cache instance for use from FastAPI request handlers —
    avoids opening a new sqlite connection per request."""
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = GisCache()
    return _SHARED_CACHE
