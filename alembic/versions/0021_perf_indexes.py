"""Add targeted composite/partial indexes for slow filter+sort+vector queries

Revision ID: 0021
Revises: 0020
Create Date: 2026-08-25

Measured via EXPLAIN (ANALYZE, BUFFERS) against the live DB (see audit,
2026-08-25): all three queries below were 700ms-1.7s each, because the
existing indexes are single-column only -- the planner had to BitmapAnd
several separate bitmap scans, or (for the vector search) fall back to a
brute-force Nested Loop with no ANN index at all.

Not using CREATE INDEX CONCURRENTLY: this is a single-user/local app, not a
zero-downtime production service, so a brief write-lock during migration is
an acceptable tradeoff for the simplicity of running inside Alembic's normal
transactional migration. The HNSW build over ~177K embeddings may take a
noticeable (but one-time) amount of time to run.
"""
from alembic import op
import sqlalchemy as sa

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Fixes search_listings' count + main queries (property_type_slug +
    # geo_category filter, ORDER BY last_seen_at DESC LIMIT 50): status='active'
    # alone matches ~63% of listings, so it's a bad primary bitmap filter --
    # made a partial-index qualifier instead of an indexed column so the
    # actually-selective columns (type, geo) plus the sort column do the work.
    op.create_index(
        "ix_listings_type_geo_active_lastseen",
        "listings",
        ["property_type_slug", "geo_category", sa.text("last_seen_at DESC")],
        postgresql_where=sa.text("status = 'active'"),
    )

    # Fixes get_market_trend's mv_analytics_flat query: the existing mv_af_geo
    # index alone returns 34K rows, then deal_type_normalized is applied as a
    # non-indexed post-scan Filter.
    op.create_index(
        "ix_mv_af_deal_geo",
        "mv_analytics_flat",
        ["deal_type_normalized", "geo_category"],
    )

    # Fixes retrieve_comparables' vector search: no ANN index existed at all,
    # so pgvector's <=> operator forced a brute-force scan + sort over the
    # full candidate set. Deliberately deferred in Phase 7 Tier 1 pending
    # full-scale embedding backfill -- now justified (~177K rows, ~1.5s/query).
    op.execute(
        "CREATE INDEX ix_listing_embeddings_hnsw ON listing_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_listing_embeddings_hnsw")
    op.drop_index("ix_mv_af_deal_geo", table_name="mv_analytics_flat")
    op.drop_index("ix_listings_type_geo_active_lastseen", table_name="listings")
