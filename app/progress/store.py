"""
Thread-safe SSE progress store.
One ProgressRun per active scrape, multiple subscribers (browser tabs) per run.
"""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProgressRun:
    run_id: str
    status: str = "running"       # running | completed | failed
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None

    routes_total: int = 0
    routes_done: int = 0
    listings_total: int = 0
    listings_done: int = 0
    listings_upserted: int = 0

    last_message: str = ""
    error: str = ""

    _queues: list[queue.Queue] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def push(self, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait({"type": event_type, "data": data})
                except queue.Full:
                    pass

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "routes_total": self.routes_total,
            "routes_done": self.routes_done,
            "listings_total": self.listings_total,
            "listings_done": self.listings_done,
            "listings_upserted": self.listings_upserted,
            "last_message": self.last_message,
            "error": self.error,
        }


class ProgressStore:
    """Singleton-style store holding active and recent scrape runs."""

    def __init__(self) -> None:
        self._runs: dict[str, ProgressRun] = {}
        self._lock = threading.Lock()

    def create(self, run_id: str) -> ProgressRun:
        run = ProgressRun(run_id=run_id)
        with self._lock:
            self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> ProgressRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def all_runs(self) -> list[ProgressRun]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)

    def cleanup_old(self, keep: int = 20) -> None:
        with self._lock:
            sorted_runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
            for run in sorted_runs[keep:]:
                del self._runs[run.run_id]


progress_store = ProgressStore()
