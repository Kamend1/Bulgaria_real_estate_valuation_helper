"""
Standalone scrape script — runs as a separate OS process.

Launched by FastAPI via subprocess.Popen:
  python -m scripts.run_scrape --run-id <uuid>

Reads all parameters from scrape_runs table (not from CLI args).
Writes progress directly to DB. Supports graceful stop via stop_requested flag.
"""

import argparse
import contextlib
import csv as _csv
import io
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Project root on sys.path so `app.*` and `utils.*` are importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import text

from app.config import settings
from app.db.session import db_session
from app.services.scrape_service import (
    _build_listing_values,
    _ingest_csv_to_db,
    sync_taxonomy_to_db,
)

_DEAL_SLUG_TO_NORM = {"prodazhbi": "sale", "naemi": "rent"}
_HEARTBEAT_INTERVAL = 30   # seconds between stdout-triggered DB writes
_DEDICATED_HEARTBEAT_INTERVAL = 20  # dedicated thread: always beats regardless of output
_ROUTE_RE  = re.compile(r"\[(\d+)/(\d+)\]\s+routes?\s+completed", re.IGNORECASE)
_LISTING_RE = re.compile(r"\[(\d+)/(\d+)\]\s+listings?\s+processed", re.IGNORECASE)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_update(run_uuid, **kwargs):
    """Write any subset of scrape_runs columns to DB, always setting last_heartbeat_at."""
    cols = ["last_heartbeat_at = now()"]
    params = {"id": run_uuid}
    for k, v in kwargs.items():
        cols.append(f"{k} = :{k}")
        params[k] = v
    with db_session() as s:
        s.execute(text(f"UPDATE scrape_runs SET {', '.join(cols)} WHERE id = :id"), params)


def _start_heartbeat(run_uuid) -> threading.Event:
    """Dedicated thread that keeps last_heartbeat_at fresh regardless of stdout output."""
    stop_evt = threading.Event()

    def _beat():
        while not stop_evt.wait(_DEDICATED_HEARTBEAT_INTERVAL):
            try:
                _db_update(run_uuid)
            except Exception:
                pass

    t = threading.Thread(target=_beat, daemon=True, name="scrape-heartbeat")
    t.start()
    return stop_evt


def _check_stop(run_uuid) -> bool:
    with db_session() as s:
        row = s.execute(
            text("SELECT stop_requested FROM scrape_runs WHERE id = :id"),
            {"id": run_uuid},
        ).fetchone()
    return bool(row and row[0])


# ── Progress capture (intercepts scraper prints → DB) ─────────────────────────

class _DBCapture(io.StringIO):
    """Wraps stdout, parses scraper progress lines, writes to DB throttled.
    Accumulates all lines so they can be written as log_text at completion.
    """

    def __init__(self, run_uuid, real_stdout):
        super().__init__()
        self._uuid = run_uuid
        self._real = real_stdout
        self._buf = ""
        self._pending: dict = {}
        self._last = 0.0
        self._log_lines: list[str] = []

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line.strip())
        return len(s)

    def flush(self):
        self._real.flush()

    def _handle(self, line: str):
        if not line:
            return
        self._log_lines.append(line)
        self._pending["last_message"] = line[:500]
        m = _ROUTE_RE.search(line)
        if m:
            self._pending["routes_done"]  = int(m.group(1))
            self._pending["routes_total"] = int(m.group(2))
        m = _LISTING_RE.search(line)
        if m:
            self._pending["listings_downloaded"] = int(m.group(1))
            self._pending["listings_found"]      = int(m.group(2))
        if time.time() - self._last >= _HEARTBEAT_INTERVAL:
            self._flush()

    def _flush(self):
        if self._pending:
            try:
                _db_update(self._uuid, **self._pending)
            except Exception:
                pass
            self._pending.clear()
        self._last = time.time()

    def final_flush(self):
        self._flush()

    def append_log(self, line: str) -> None:
        """Manually append a line (for phases that run outside redirect_stdout)."""
        if line:
            self._log_lines.append(line)

    def get_log_text(self) -> str:
        return "\n".join(self._log_lines)


# ── Archiving (Phase 5) ───────────────────────────────────────────────────────

def _archive_stale(run_uuid, deal_types_normalized: list[str], run_started_at) -> int:
    """
    Mark listings in scope that were NOT seen in this run as archived.
    Only called after a full national run (geo_paths=[]).
    """
    with db_session() as s:
        result = s.execute(text("""
            UPDATE listings
               SET status             = 'archived',
                   archived_at        = now(),
                   archived_by_run_id = :run_id
             WHERE status = 'active'
               AND deal_type_normalized = ANY(:dtypes)
               AND last_seen_at < :started_at
        """), {
            "run_id": run_uuid,
            "dtypes": deal_types_normalized,
            "started_at": run_started_at,
        })
        return result.rowcount


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    try:
        run_uuid = uuid.UUID(args.run_id)
    except ValueError:
        print(f"Invalid --run-id: {args.run_id}", file=sys.stderr)
        sys.exit(1)

    # Read parameters from DB
    with db_session() as s:
        row = s.execute(
            text("SELECT deal_types, geo_paths, property_types, started_at "
                 "FROM scrape_runs WHERE id = :id"),
            {"id": run_uuid},
        ).fetchone()
    if row is None:
        print(f"Run {run_uuid} not found in DB", file=sys.stderr)
        sys.exit(1)

    deal_types:     list[str] = list(row[0] or [])
    geo_paths:      list[str] = list(row[1] or [])
    property_types: list[str] = list(row[2] or [])
    run_started_at             = row[3]

    deal_types_normalized = [_DEAL_SLUG_TO_NORM.get(d, d) for d in deal_types]
    full_coverage = (len(geo_paths) == 0)

    # Register PID and start dedicated heartbeat
    _db_update(run_uuid, pid=os.getpid(), status="running", phase="init")
    hb_stop = _start_heartbeat(run_uuid)

    run_dir = _PROJECT_ROOT / settings.scrape_runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    capture = _DBCapture(run_uuid, sys.stdout)

    try:
        with contextlib.redirect_stdout(capture):
            from utils.fetch_data.fetch_data_utils import (
                ScrapeSelection,
                collect_listing_urls_for_routes_parallel_streaming,
                download_and_parse_listing_batch_streaming,
                load_valid_deal_types,
                load_valid_geo_paths,
                load_valid_property_types,
                refresh_taxonomy,
            )

            # ── Phase: taxonomy (Notebook 01) ────────────────────────────────
            _taxonomy_dir = _PROJECT_ROOT / settings.taxonomy_dir
            _db_update(run_uuid, phase="taxonomy", last_message="Обновяване на таксономия от imot.bg…")
            refresh_taxonomy(str(_taxonomy_dir))

            # ── Phase: routes ────────────────────────────────────────────────
            _db_update(run_uuid, phase="routes", last_message="Зареждане на таксономия…")
            sync_taxonomy_to_db(str(_taxonomy_dir))

            valid_deal_types     = load_valid_deal_types(str(_taxonomy_dir / "valid_deal_types.csv"))
            valid_geo_paths      = load_valid_geo_paths(str(_taxonomy_dir / "valid_geo_paths.csv"))
            valid_property_types = load_valid_property_types(str(_taxonomy_dir / "valid_property_types.csv"))

            if geo_paths:
                # Explicit selection — validate and build as-is
                selection = ScrapeSelection(
                    deal_types=deal_types,
                    geo_paths=geo_paths,
                    property_types=property_types or [],
                )
                route_urls = selection.build_urls(
                    valid_deal_types=valid_deal_types,
                    valid_geo_paths=valid_geo_paths,
                    valid_property_types=valid_property_types,
                )
            else:
                # Full national coverage — each deal_type has its own geo_paths set
                route_urls = []
                for dt in deal_types:
                    dt_geo_paths = sorted({gp for (d, gp) in valid_geo_paths if d == dt})
                    if not dt_geo_paths:
                        continue
                    sel = ScrapeSelection(
                        deal_types=[dt],
                        geo_paths=dt_geo_paths,
                        property_types=property_types or [],
                    )
                    route_urls.extend(sel.build_urls(
                        valid_deal_types=valid_deal_types,
                        valid_geo_paths=valid_geo_paths,
                        valid_property_types=valid_property_types,
                    ))
            _db_update(run_uuid, routes_total=len(route_urls),
                       last_message=f"Намерени {len(route_urls)} маршрута")

            if _check_stop(run_uuid):
                _db_update(run_uuid, status="interrupted", last_message="Спрян от потребителя")
                return

            route_result = collect_listing_urls_for_routes_parallel_streaming(
                route_urls=route_urls,
                checkpoint_dir=str(run_dir),
                max_workers=settings.scrape_max_workers_routes,
                delay_seconds=settings.scrape_delay_seconds,
            )
            capture.final_flush()

            # Read deduplicated listing URLs
            listing_urls_path = route_result["listing_urls_path"]
            seen: set[str] = set()
            listing_urls: list[str] = []
            if Path(listing_urls_path).exists():
                with open(listing_urls_path, encoding="utf-8", errors="replace") as f:
                    for r in _csv.DictReader(f):
                        url = r.get("listing_url", "").strip()
                        if url and url not in seen:
                            seen.add(url)
                            listing_urls.append(url)

            _db_update(run_uuid, listings_found=len(listing_urls),
                       phase="download",
                       last_message=f"Намерени {len(listing_urls)} обяви — стартира изтегляне")

            if _check_stop(run_uuid):
                _db_update(run_uuid, status="interrupted", last_message="Спрян от потребителя")
                return

            # ── Phase: download ──────────────────────────────────────────────
            download_and_parse_listing_batch_streaming(
                listing_urls=listing_urls,
                output_dir=str(run_dir),
                max_workers=settings.scrape_max_workers_listings,
            )
            capture.final_flush()

        # ── Phase: ingest ────────────────────────────────────────────────────────────────────────
        capture.append_log("─── Фаза: вписване в БД ───")
        _db_update(run_uuid, phase="ingest", last_message="Вписване в базата данни…")

        csv_path = run_dir / "parsed_listings.csv"
        ingested = 0
        if csv_path.exists():
            def _ingest_cb(n: int):
                capture.append_log(f"Вписани {n} обяви…")
                _db_update(run_uuid, listings_ingested=n,
                           last_message=f"Вписани {n} обяви…")

            ingested = _ingest_csv_to_db(
                str(csv_path), run_uuid,
                progress_callback=_ingest_cb,
            )
        else:
            _no_csv_msg = "ВНИМАНИЕ: parsed_listings.csv не е намерен"
            capture.append_log(_no_csv_msg)
            _db_update(run_uuid, last_message=_no_csv_msg)

        # ── Archive ───────────────────────────────────────────────────────────────────────────
        archived = 0
        if full_coverage and ingested > 0:
            capture.append_log("─── Фаза: архивиране на стари обяви ───")
            _db_update(run_uuid, phase="archive", last_message="Архивиране на стари обяви…")
            archived = _archive_stale(run_uuid, deal_types_normalized, run_started_at)
            capture.append_log(f"Архивирани: {archived} обяви")
        elif full_coverage and ingested == 0:
            _skip_msg = "Архивирането е пропуснато — 0 вписани обяви"
            capture.append_log(_skip_msg)
            _db_update(run_uuid, last_message=_skip_msg)

        # ── Done ────────────────────────────────────────────────────────────────────────────────
        done_msg = f"Готово. Вписани: {ingested}, Архивирани: {archived}"
        capture.append_log(f"─── {done_msg} ───")
        with db_session() as s:
            s.execute(text("""
                UPDATE scrape_runs
                   SET status              = 'completed',
                       finished_at         = now(),
                       listings_upserted   = :n,
                       listings_ingested   = :n,
                       phase               = 'done',
                       last_message        = :msg,
                       last_heartbeat_at   = now(),
                       log_text            = :log_text
                 WHERE id = :id
            """), {
                "id": run_uuid,
                "n": ingested,
                "msg": done_msg,
                "log_text": capture.get_log_text(),
            })

    except Exception as exc:
        import traceback
        with db_session() as s:
            s.execute(text("""
                UPDATE scrape_runs
                   SET status        = 'failed',
                       finished_at   = now(),
                       error_message = :err,
                       last_heartbeat_at = now(),
                       log_text      = :log_text
                 WHERE id = :id
            """), {
                "id": run_uuid,
                "err": str(exc)[:2000],
                "log_text": capture.get_log_text(),
            })
        print(f"SCRAPE FAILED: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
    finally:
        hb_stop.set()

if __name__ == "__main__":
    main()
