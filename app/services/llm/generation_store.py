"""In-memory store for AI valuation generation progress (Phase 7, Tier 3).

Unlike app/progress/store.py's DB-backed ScrapeRun pattern (needed because a
scrape can run for hours and must survive a uvicorn restart), one
generation call takes seconds -- a plain in-memory dict is enough and
avoids adding DB-polling overhead for something this short-lived.
"""
import threading
import uuid

from app.services.llm.valuation_chain import GenerationProgress

_lock = threading.Lock()
_runs: dict[str, GenerationProgress] = {}


def create_run() -> str:
    run_id = str(uuid.uuid4())
    with _lock:
        _runs[run_id] = GenerationProgress()
    return run_id


def update(run_id: str, progress: GenerationProgress) -> None:
    with _lock:
        _runs[run_id] = progress


def get(run_id: str) -> GenerationProgress | None:
    with _lock:
        return _runs.get(run_id)


def cleanup_old(keep: int = 50) -> None:
    with _lock:
        if len(_runs) > keep:
            for k in list(_runs.keys())[:-keep]:
                del _runs[k]
