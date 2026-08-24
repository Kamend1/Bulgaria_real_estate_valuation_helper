"""Add income_market_rationale to appraisal_reports (Phase 7, Tier 5)

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("income_market_rationale", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "income_market_rationale")
