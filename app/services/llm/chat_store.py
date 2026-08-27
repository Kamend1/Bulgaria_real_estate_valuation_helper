"""In-memory store for assistant chat turn progress (Tier 2, 2026-08-26) --
mirrors generation_store.py exactly (same short-lived-in-memory reasoning:
one turn takes seconds, no DB-backed durability needed across a restart)."""
import threading
import uuid

from app.services.llm.assistant_chain import ChatProgress

_lock = threading.Lock()
_turns: dict[str, ChatProgress] = {}


def create_turn() -> str:
    turn_id = str(uuid.uuid4())
    with _lock:
        _turns[turn_id] = ChatProgress()
    return turn_id


def update(turn_id: str, progress: ChatProgress) -> None:
    with _lock:
        _turns[turn_id] = progress


def get(turn_id: str) -> ChatProgress | None:
    with _lock:
        return _turns.get(turn_id)


def cleanup_old(keep: int = 50) -> None:
    with _lock:
        if len(_turns) > keep:
            for k in list(_turns.keys())[:-keep]:
                del _turns[k]
