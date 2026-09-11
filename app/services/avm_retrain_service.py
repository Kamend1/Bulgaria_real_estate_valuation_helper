"""
Automatic AVM retraining hook (Phase 14 Tier 2.1) — run at the end of every
scrape (see scrape_service.py's step 10) to keep every segment's active
model as current as the corpus, without ever exposing a "retrain" button on
the web (that stayed a deliberate Phase 5 decision).

Admin/maintainer-only by construction, not by a role check: this only does
anything on a machine whose .env already holds R2_MAINTAINER_* -- the same
credential scripts/train_avm_model.py's --push-to-r2 requires. A deployed
instance without that credential (any colleague's/client's machine) silently
skips this step, exactly like a missing OPENAI_API_KEY skips the embeddings
backfill step right before it.

Retrains unconditionally on every successful scrape (2026-09-11, following
a real incident where a growth-percentage gate silently skipped every
segment because one scrape's worth of new rows -- a few percent against a
150K+ row base -- never came close to the threshold; the owner decided
"every scrape retrains" was the simpler, more predictable policy than
tuning that threshold). scripts/train_avm_model.py's own --min-rows guard
still protects a segment that genuinely doesn't have enough data -- it
no-ops for that one segment (exit 0, no model written) rather than erroring,
so calling it for all 5 segments every time is safe even for a
still-thin segment.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Callable

from sqlalchemy.orm import Session

from app.config import settings
from utils.ml.avm_features import SEGMENT_PROPERTY_TYPES

OnProgress = Callable[[str], None]


def maybe_retrain_avm_models(db: Session, on_progress: OnProgress | None = None) -> list[dict]:
    """Retrains every segment via scripts/train_avm_model.py --push-to-r2
    as a subprocess (same mechanism the manual CLI usage already uses).
    `db` is accepted for call-site compatibility (scrape_service.py passes
    its own session) but is no longer read here -- retraining is now
    unconditional, not growth-gated. Returns a list of {segment, action,
    detail} for logging -- never raises, every failure mode degrades to a
    skipped/failed entry.
    """
    log = on_progress or (lambda msg: None)
    results: list[dict] = []

    if not settings.r2_maintainer_access_key_id:
        log("Автоматично AVM пре-трениране: пропуснато (няма R2_MAINTAINER_* на тази машина).")
        return results

    for segment in SEGMENT_PROPERTY_TYPES:
        try:
            log(f"AVM: пре-трениране на {segment}…")
            proc = subprocess.run(
                [sys.executable, "-m", "scripts.train_avm_model", "--segment", segment, "--push-to-r2"],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode == 0:
                if "SKIPPED" in (proc.stdout or ""):
                    results.append({"segment": segment, "action": "skipped", "detail": "под прага за минимален брой редове"})
                    log(f"AVM: {segment} пропуснат (твърде малко редове).")
                else:
                    results.append({"segment": segment, "action": "retrained", "detail": "успешно"})
                    log(f"AVM: {segment} пре-трениран успешно.")
            else:
                tail = (proc.stderr or proc.stdout or "")[-500:]
                results.append({"segment": segment, "action": "failed", "detail": tail})
                log(f"AVM: {segment} пре-трениране се провали (код {proc.returncode}).")
        except Exception as exc:
            results.append({"segment": segment, "action": "failed", "detail": str(exc)})
            log(f"AVM: {segment} пре-трениране хвърли грешка: {exc}")

    return results
