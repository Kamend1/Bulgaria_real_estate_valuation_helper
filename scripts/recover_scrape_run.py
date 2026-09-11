"""
One-off recovery for a scrape run that lost data to a transient failure
(real incident, 2026-09-11): a ~6-minute DNS outage on the scraping
machine coincided exactly with the download of the entire rent segment
of a run, wiping out 100% of it (37,193/37,193 failed with the same
NameResolutionError). load_completed_listing_urls() used to treat any row
in download_manifest.csv -- success or failure -- as "already done", so
even a resumed run would have skipped retrying these forever; see the fix
in utils/fetch_data/fetch_data_utils.py.

This does NOT re-crawl routes -- it only re-attempts the URLs already
recorded in the given run's own download_manifest.csv that never actually
got HTML downloaded, then re-ingests that run's full parsed_listings.csv
(idempotent upsert, safe to re-run over already-known rows) and refreshes
the same downstream state a normal scrape's tail does: analytics view,
embeddings backfill, AVM retrain.

Usage (from project root):
    python -m scripts.recover_scrape_run --run-id <scrape_run uuid>
    python -m scripts.recover_scrape_run --run-id <uuid> --skip-avm-retrain

Safe to re-run: download/parse skips anything already successfully
downloaded, and DB ingestion upserts by ad_url.
"""
import argparse
import csv
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.db.models import ScrapeRun
from app.db.session import db_session
from app.services.scrape_service import _ingest_csv_to_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed downloads for one scrape run and re-ingest")
    parser.add_argument("--run-id", required=True, help="ScrapeRun.id (also the data/scrape_runs/<run-id>/ directory name)")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--skip-analytics", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-avm-retrain", action="store_true")
    args = parser.parse_args()

    db_run_id = uuid.UUID(args.run_id)
    run_dir = Path(settings.scrape_runs_dir) / args.run_id
    manifest_path = run_dir / "download_manifest.csv"
    if not manifest_path.exists():
        print(f"No download_manifest.csv found at {manifest_path} -- wrong --run-id?")
        return

    with db_session() as db:
        run_row = db.get(ScrapeRun, db_run_id)
        if run_row is None:
            print(f"No ScrapeRun row with id {args.run_id} -- wrong --run-id?")
            return
        print(f"Recovering run {args.run_id} (status={run_row.status}, started {run_row.started_at})")

    urls: list[str] = []
    seen: set[str] = set()
    with open(manifest_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            u = (row.get("listing_url") or "").strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
    print(f"Loaded {len(urls)} unique listing URLs originally attempted for this run.")

    from utils.fetch_data.fetch_data_utils import (
        download_and_parse_listing_batch_streaming,
        load_completed_listing_urls,
    )

    already_ok = load_completed_listing_urls(manifest_path)
    print(f"Already successfully downloaded: {len(already_ok)}")
    print(f"Will retry: {len(urls) - len(already_ok)}")

    download_and_parse_listing_batch_streaming(
        listing_urls=urls,
        output_dir=str(run_dir),
        max_workers=args.max_workers,
    )

    still_failed = len(urls) - len(load_completed_listing_urls(manifest_path))
    print(f"After retry, still failed: {still_failed}")

    parsed_path = run_dir / "parsed_listings.csv"
    upserted = None
    if parsed_path.exists():
        print("Ingesting parsed_listings.csv into the DB (idempotent upsert over the FULL file)…")
        upserted = _ingest_csv_to_db(str(parsed_path), db_run_id)
        print(f"Upserted in this pass (includes re-upserts of already-known rows): {upserted}")
    else:
        print("parsed_listings.csv not found -- nothing to ingest.")

    with db_session() as db:
        run_row = db.get(ScrapeRun, db_run_id)
        if run_row and upserted is not None:
            run_row.listings_upserted = upserted

    if not args.skip_analytics:
        try:
            from app.services.analytics_service import compute_price_events, refresh_mv
            print("Recomputing price events + refreshing analytics view…")
            n_events = compute_price_events(db_run_id)
            refresh_mv()
            print(f"Price events: {n_events}. Analytics view refreshed.")
        except Exception as exc:
            print(f"WARNING: analytics refresh failed: {exc}")

    if not args.skip_embeddings:
        try:
            from app.services.llm.embed_backfill import backfill_embeddings
            print("Backfilling embeddings for rows touched by this run…")
            with db_session() as db:
                n = backfill_embeddings(db, run_id=db_run_id)
            print(f"Embeddings: {n} listings embedded/updated.")
        except Exception as exc:
            print(f"WARNING: embeddings backfill failed: {exc}")

    if not args.skip_avm_retrain:
        try:
            from app.services.avm_retrain_service import maybe_retrain_avm_models
            print("Running AVM retrain hook (unconditional, every segment)…")
            with db_session() as db:
                results = maybe_retrain_avm_models(db, on_progress=print)
            for r in results:
                print(" ", r)
        except Exception as exc:
            print(f"WARNING: AVM retrain failed: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
