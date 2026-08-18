"""Add structured adjustment_factors to comparable_pool + submarket_rationale to appraisal_reports

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "comparable_pool",
        sa.Column("adjustment_factors", JSONB, nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("submarket_rationale", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "submarket_rationale")
    op.drop_column("comparable_pool", "adjustment_factors")
