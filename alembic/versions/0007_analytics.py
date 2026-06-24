"""Analytics: listing_price_events table + mv_analytics_flat materialized view

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-24
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. listing_price_events ───────────────────────────────────────────────
    op.create_table(
        "listing_price_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("listing_id", sa.BigInteger(),
                  sa.ForeignKey("listings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("from_run_id", PGUUID(as_uuid=True),
                  sa.ForeignKey("scrape_runs.id"), nullable=True),
        sa.Column("to_run_id", PGUUID(as_uuid=True),
                  sa.ForeignKey("scrape_runs.id"), nullable=True),
        sa.Column("observed_at", TIMESTAMP(timezone=True), server_default=func.now()),
        sa.Column("price_before", sa.Numeric(14, 2)),
        sa.Column("price_after", sa.Numeric(14, 2)),
        sa.Column("ppsqm_before", sa.Numeric(10, 2)),
        sa.Column("ppsqm_after", sa.Numeric(10, 2)),
        sa.Column("change_pct", sa.Numeric(8, 4)),
        sa.Column("direction", sa.Text()),
        sa.Column("days_between", sa.Integer()),
        sa.UniqueConstraint("listing_id", "to_run_id", name="uq_price_event_listing_run"),
    )
    op.create_index("ix_price_events_listing_id", "listing_price_events", ["listing_id"])
    op.create_index("ix_price_events_to_run_id", "listing_price_events", ["to_run_id"])
    op.create_index("ix_price_events_direction", "listing_price_events", ["direction"])

    # ── 2. Performance indexes on listing_snapshots ───────────────────────────
    op.create_index(
        "ix_snapshots_listing_scraped",
        "listing_snapshots",
        ["listing_id", "scraped_at"],
    )
    op.create_index(
        "ix_snapshots_run_id",
        "listing_snapshots",
        ["scrape_run_id"],
    )

    # ── 3. Materialized view: flat analytical table ───────────────────────────
    op.execute("""
        CREATE MATERIALIZED VIEW mv_analytics_flat AS
        SELECT
            ls.id                       AS snapshot_id,
            ls.scraped_at,
            ls.scrape_run_id,
            sr.started_at               AS run_started_at,
            ls.price_per_sqm_model,
            ls.total_price,
            ls.area_sqm_model,
            ls.days_on_market,
            l.deal_type_normalized,
            l.geo_category,
            l.title_city_model,
            l.title_geo_2_model,
            l.property_type_slug,
            l.construction_type_model,
            l.id                        AS listing_id
        FROM listing_snapshots ls
        JOIN listings l  ON l.id  = ls.listing_id
        JOIN scrape_runs sr ON sr.id = ls.scrape_run_id
        WHERE l.training_eligible = TRUE
          AND ls.price_per_sqm_model IS NOT NULL
          AND sr.status IN ('completed', 'imported')
        WITH DATA
    """)

    # Unique index required for CONCURRENTLY refresh
    op.execute(
        "CREATE UNIQUE INDEX mv_analytics_flat_snapshot_id "
        "ON mv_analytics_flat (snapshot_id)"
    )
    # Query indexes
    op.execute("CREATE INDEX mv_af_run      ON mv_analytics_flat (scrape_run_id)")
    op.execute("CREATE INDEX mv_af_run_date ON mv_analytics_flat (run_started_at)")
    op.execute("CREATE INDEX mv_af_deal     ON mv_analytics_flat (deal_type_normalized)")
    op.execute("CREATE INDEX mv_af_geo      ON mv_analytics_flat (geo_category)")
    op.execute("CREATE INDEX mv_af_city     ON mv_analytics_flat (title_city_model)")
    op.execute("CREATE INDEX mv_af_quarter  ON mv_analytics_flat (title_geo_2_model)")
    op.execute("CREATE INDEX mv_af_prop     ON mv_analytics_flat (property_type_slug)")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_analytics_flat")
    op.drop_index("ix_snapshots_run_id", "listing_snapshots")
    op.drop_index("ix_snapshots_listing_scraped", "listing_snapshots")
    op.drop_index("ix_price_events_direction", "listing_price_events")
    op.drop_index("ix_price_events_to_run_id", "listing_price_events")
    op.drop_index("ix_price_events_listing_id", "listing_price_events")
    op.drop_table("listing_price_events")
