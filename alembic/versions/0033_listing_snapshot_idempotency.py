"""Add unique constraint on listing_snapshots(listing_id, scrape_run_id) (2026-09-11)

Revision ID: 0033
Revises: 0032
Create Date: 2026-09-11

Real incident (2026-09-11): scripts/recover_scrape_run.py re-ingested the
FULL cumulative parsed_listings.csv for a run to recover lost rent data,
which re-processed already-ingested rows too. _ingest_rows_to_db had no
idempotency guard on snapshot insertion (a bare session.add(ListingSnapshot
(...)) per successful upsert), so re-ingesting the same CSV rows created
129,197 true duplicate snapshot rows for that run, inflating every
mv_analytics_flat-derived count (one row per snapshot, not deduplicated by
listing). The duplicates were deleted manually for the affected run; this
constraint plus the matching on_conflict_do_nothing in scrape_service.py
(_ingest_rows_to_db) prevent it from ever recurring for ANY run.

Pre-existing duplicates (if any exist for other historical runs) must be
cleaned up before this migration can apply -- the upgrade() below does that
defensively, keeping the lowest id per (listing_id, scrape_run_id) group,
before creating the constraint.
"""
from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM listing_snapshots a
        USING listing_snapshots b
        WHERE a.listing_id = b.listing_id
          AND a.scrape_run_id = b.scrape_run_id
          AND a.id > b.id
        """
    )
    op.create_unique_constraint(
        "uq_listing_snapshots_listing_run",
        "listing_snapshots",
        ["listing_id", "scrape_run_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_listing_snapshots_listing_run", "listing_snapshots", type_="unique")
