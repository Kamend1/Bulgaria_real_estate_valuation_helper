"""
Automatic AVM retraining hook (Phase 14 Tier 2.1) — run at the end of every
scrape (see scrape_service.py's step 10) to keep each segment's active model
roughly current with the growing corpus, without ever exposing a "retrain"
button on the web (that stayed a deliberate Phase 5 decision).

Admin/maintainer-only by construction, not by a role check: this only does
anything on a machine whose .env already holds R2_MAINTAINER_* -- the same
credential scripts/train_avm_model.py's --push-to-r2 requires. A deployed
instance without that credential (any colleague's/client's machine) silently
skips this step, exactly like a missing OPENAI_API_KEY skips the embeddings
backfill step right before it.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Callable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AvmModel
from utils.ml.avm_features import SEGMENT_PROPERTY_TYPES

OnProgress = Callable[[str], None]


def _count_eligible_rows(db: Session, slugs: list[str]) -> int:
    # Mirrors train_avm_model.py::_load_segment_df's WHERE clause exactly --
    # this must count the same rows that script would actually train on, or
    # the growth comparison below is meaningless.
    result = db.execute(
        text("""
            SELECT COUNT(*) FROM listings
            WHERE training_eligible = TRUE
              AND deal_type_normalized = 'sale'
              AND property_type_slug = ANY(:slugs)
              AND price_per_sqm_model IS NOT NULL
        """),
        {"slugs": list(slugs)},
    )
    return result.scalar() or 0


def maybe_retrain_avm_models(db: Session, on_progress: OnProgress | None = None) -> list[dict]:
    """Checks every segment for enough row growth to justify a retrain, and
    kicks off scripts/train_avm_model.py --push-to-r2 as a subprocess for
    each one that qualifies (same mechanism the manual CLI usage already
    uses). Returns a list of {segment, action, detail} for logging -- never
    raises, every failure mode degrades to a skipped/failed entry.
    """
    log = on_progress or (lambda msg: None)
    results: list[dict] = []

    if not settings.r2_maintainer_access_key_id:
        log("Автоматично AVM пре-трениране: пропуснато (няма R2_MAINTAINER_* на тази машина).")
        return results

    for segment, slugs in SEGMENT_PROPERTY_TYPES.items():
        try:
            current_count = _count_eligible_rows(db, slugs)
            active = (
                db.query(AvmModel)
                .filter(AvmModel.segment == segment, AvmModel.is_active.is_(True))
                .order_by(AvmModel.trained_at.desc())
                .first()
            )
            if active is None:
                results.append({"segment": segment, "action": "skipped", "detail": "няма активен модел все още — трениране очаква --min-rows прага през ръчния CLI път"})
                continue

            threshold_count = active.training_row_count * (1 + settings.avm_auto_retrain_growth_pct / 100)
            if current_count < threshold_count:
                results.append({
                    "segment": segment, "action": "skipped",
                    "detail": f"{current_count} реда (нужни {threshold_count:.0f}+ за пре-трениране, база {active.training_row_count})",
                })
                continue

            log(f"AVM: {segment} нарасна {active.training_row_count} -> {current_count} реда, стартирам пре-трениране…")
            proc = subprocess.run(
                [sys.executable, "-m", "scripts.train_avm_model", "--segment", segment, "--push-to-r2"],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode == 0:
                results.append({"segment": segment, "action": "retrained", "detail": f"{current_count} реда"})
                log(f"AVM: {segment} пре-трениран успешно.")
            else:
                tail = (proc.stderr or proc.stdout or "")[-500:]
                results.append({"segment": segment, "action": "failed", "detail": tail})
                log(f"AVM: {segment} пре-трениране се провали (код {proc.returncode}).")
        except Exception as exc:
            results.append({"segment": segment, "action": "failed", "detail": str(exc)})
            log(f"AVM: {segment} пре-трениране хвърли грешка: {exc}")

    return results
