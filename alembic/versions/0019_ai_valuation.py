"""Add pgvector extension, listing_embeddings, and ai_valuation_runs tables (Phase 7)

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None

# text-embedding-3-small's default dimensionality. If a different provider/
# model with a different dimension is ever added, that's a new row shape —
# store it as a *separate* table version or widen via a follow-up migration,
# don't silently truncate/pad vectors from a mismatched model.
EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "listing_embeddings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("listing_id", sa.BigInteger(), sa.ForeignKey("listings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("embedded_text", sa.Text(), nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
        # One current embedding per (listing, provider, model) -- re-embedding
        # with the same provider/model upserts rather than accumulating rows.
        sa.UniqueConstraint("listing_id", "provider", "model", name="uq_listing_embeddings_listing_provider_model"),
    )
    op.create_index("ix_listing_embeddings_listing_id", "listing_embeddings", ["listing_id"])
    # No HNSW/ivfflat index yet -- deliberately deferred (see plan doc Tier 1)
    # until row counts make a brute-force scan visibly slow.

    op.create_table(
        "ai_valuation_runs",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PGUUID(as_uuid=True), sa.ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(10, 4), nullable=True),
        sa.Column("output", JSONB(), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_ai_valuation_runs_report_id", "ai_valuation_runs", ["report_id"])


def downgrade() -> None:
    op.drop_index("ix_ai_valuation_runs_report_id", table_name="ai_valuation_runs")
    op.drop_table("ai_valuation_runs")
    op.drop_index("ix_listing_embeddings_listing_id", table_name="listing_embeddings")
    op.drop_table("listing_embeddings")
    # Extension left installed on downgrade -- dropping it would fail if any
    # other database/table on this (possibly shared) Postgres instance also
    # depends on it, and CREATE EXTENSION IF NOT EXISTS makes re-upgrading
    # idempotent either way.
