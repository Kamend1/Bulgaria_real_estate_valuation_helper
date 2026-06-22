"""Add scrape progress columns + listing status/archiving columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── scrape_runs: progress + process management ─────────────────────────────
    op.add_column("scrape_runs", sa.Column("pid", sa.Integer, nullable=True))
    op.add_column("scrape_runs", sa.Column("phase", sa.String(20), nullable=True))
    op.add_column("scrape_runs", sa.Column("listings_downloaded", sa.Integer, server_default="0"))
    op.add_column("scrape_runs", sa.Column("listings_ingested", sa.Integer, server_default="0"))
    op.add_column("scrape_runs", sa.Column("last_heartbeat_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("scrape_runs", sa.Column("stop_requested", sa.Boolean, server_default="false", nullable=False))
    op.add_column("scrape_runs", sa.Column("last_message", sa.Text, nullable=True))

    # ── listings: active/archived status ──────────────────────────────────────
    op.add_column("listings", sa.Column("status", sa.String(20), server_default="active", nullable=False))
    op.add_column("listings", sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column(
        "listings",
        sa.Column(
            "archived_by_run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("scrape_runs.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_listings_status", "listings", ["status"])


def downgrade() -> None:
    op.drop_index("ix_listings_status", table_name="listings")
    op.drop_column("listings", "archived_by_run_id")
    op.drop_column("listings", "archived_at")
    op.drop_column("listings", "status")

    op.drop_column("scrape_runs", "last_message")
    op.drop_column("scrape_runs", "stop_requested")
    op.drop_column("scrape_runs", "last_heartbeat_at")
    op.drop_column("scrape_runs", "listings_ingested")
    op.drop_column("scrape_runs", "listings_downloaded")
    op.drop_column("scrape_runs", "phase")
    op.drop_column("scrape_runs", "pid")
