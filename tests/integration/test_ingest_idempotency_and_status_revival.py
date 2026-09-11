"""
Regression tests for a real incident (2026-09-11): scripts/recover_scrape_run.py
re-ingested the FULL cumulative parsed_listings.csv for a run (necessary to
recover lost rent data), which re-processed already-ingested rows too and
exposed two latent gaps in _ingest_rows_to_db:

1. No idempotency guard on listing_snapshots insertion -- a bare
   session.add(ListingSnapshot(...)) per successful upsert created 129,197
   true duplicate rows for one run, inflating every mv_analytics_flat-derived
   count (one row per snapshot, not deduplicated by listing).
2. The listings upsert's ON CONFLICT DO UPDATE SET clause never touched
   status/archived_at/archived_by_run_id -- 49,047 listings that a prior run
   had (wrongly, due to a transient DNS outage) archived stayed stuck as
   status='archived' even after being successfully re-confirmed live.

Both were fixed: a unique constraint on listing_snapshots(listing_id,
scrape_run_id) + on_conflict_do_nothing for (1), and status='active'/
archived_at=NULL/archived_by_run_id=NULL added to the upsert SET clause for
(2). _ingest_rows_to_db opens its own real DB session/commit (it's meant to
run in a background process, not a FastAPI request) -- these tests use
disposable ad_url values under a clearly-fake domain and delete every row
they create in a finally block, per this repo's rule to never leave test
data behind in the real database.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from app.db.models import ScrapeRun
from app.db.session import db_session
from app.services.scrape_service import _ingest_rows_to_db

_TEST_URL = "https://test.invalid/regression-ingest-idempotency-1"


def _make_row(**overrides) -> dict:
    row = {
        "ad_url": _TEST_URL,
        "ad_id": "test-1",
        "deal_type_normalized": "rent",
        "property_type_slug": "one_bedroom",
        "total_price": "500",
        "currency": "EUR",
        "price_per_sqm_model": "10",
        "area_sqm_model": "50",
        "geo_category": "sofia_center",
        "views": "1",
        "vat_status": None,
        "floor_model": "2",
        "total_floors_model": "5",
        "features_count": "0",
        "training_eligible": "false",
        "published_date": None,
    }
    row.update(overrides)
    return row


def _cleanup(run_id):
    with db_session() as db:
        db.execute(text("DELETE FROM listing_snapshots WHERE scrape_run_id = :rid"), {"rid": run_id})
        db.execute(text("DELETE FROM listings WHERE ad_url = :url"), {"url": _TEST_URL})
        db.execute(text("DELETE FROM scrape_runs WHERE id = :rid"), {"rid": run_id})


def test_reingesting_same_row_does_not_duplicate_snapshots():
    run_id = uuid.uuid4()
    try:
        with db_session() as db:
            db.add(ScrapeRun(id=run_id, status="running"))

        _ingest_rows_to_db([_make_row()], run_id)
        _ingest_rows_to_db([_make_row()], run_id)
        _ingest_rows_to_db([_make_row()], run_id)

        with db_session() as db:
            count = db.execute(
                text("SELECT count(*) FROM listing_snapshots WHERE scrape_run_id = :rid"),
                {"rid": run_id},
            ).scalar()
        assert count == 1
    finally:
        _cleanup(run_id)


def test_reingesting_row_for_different_run_still_creates_new_snapshot():
    run_id_a = uuid.uuid4()
    run_id_b = uuid.uuid4()
    try:
        with db_session() as db:
            db.add(ScrapeRun(id=run_id_a, status="running"))
            db.add(ScrapeRun(id=run_id_b, status="running"))

        _ingest_rows_to_db([_make_row()], run_id_a)
        _ingest_rows_to_db([_make_row()], run_id_b)

        with db_session() as db:
            count_a = db.execute(
                text("SELECT count(*) FROM listing_snapshots WHERE scrape_run_id = :rid"),
                {"rid": run_id_a},
            ).scalar()
            count_b = db.execute(
                text("SELECT count(*) FROM listing_snapshots WHERE scrape_run_id = :rid"),
                {"rid": run_id_b},
            ).scalar()
        assert count_a == 1
        assert count_b == 1
    finally:
        _cleanup(run_id_a)
        _cleanup(run_id_b)


def test_reingesting_an_archived_listing_revives_it_to_active():
    run_id = uuid.uuid4()
    try:
        with db_session() as db:
            db.add(ScrapeRun(id=run_id, status="running"))

        _ingest_rows_to_db([_make_row()], run_id)

        with db_session() as db:
            db.execute(
                text(
                    "UPDATE listings SET status = 'archived', archived_at = now(), "
                    "archived_by_run_id = :rid WHERE ad_url = :url"
                ),
                {"rid": run_id, "url": _TEST_URL},
            )

        with db_session() as db:
            row = db.execute(
                text("SELECT status FROM listings WHERE ad_url = :url"), {"url": _TEST_URL}
            ).fetchone()
        assert row.status == "archived"

        _ingest_rows_to_db([_make_row()], run_id)

        with db_session() as db:
            row = db.execute(
                text(
                    "SELECT status, archived_at, archived_by_run_id FROM listings WHERE ad_url = :url"
                ),
                {"url": _TEST_URL},
            ).fetchone()
        assert row.status == "active"
        assert row.archived_at is None
        assert row.archived_by_run_id is None
    finally:
        _cleanup(run_id)
