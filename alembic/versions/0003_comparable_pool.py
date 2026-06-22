"""Add comparable_pool table (unbounded analysis set with pin-for-report flag)

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-22
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "comparable_pool",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "listing_id",
            sa.BigInteger,
            sa.ForeignKey("listings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("comparable_type", sa.String(10), nullable=False),  # 'sale' | 'rent'
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("pinned_for_report", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("adjustment_pct", sa.Numeric(6, 2), nullable=True),
        sa.Column("analyst_note", sa.Text, nullable=True),
        sa.UniqueConstraint("listing_id", "comparable_type", name="uq_pool_listing_type"),
    )
    op.create_index("ix_pool_type", "comparable_pool", ["comparable_type"])
    op.create_index("ix_pool_pinned", "comparable_pool", ["pinned_for_report"])


def downgrade() -> None:
    op.drop_table("comparable_pool")
