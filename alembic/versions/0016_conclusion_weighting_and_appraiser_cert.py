"""Add approach-weighting fields to appraisal_reports + appraiser cert no to users

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("weight_sales_pct", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("weight_income_pct", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("weight_residual_pct", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("weighting_rationale", sa.Text(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("appraiser_certificate_no", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "appraiser_certificate_no")
    op.drop_column("appraisal_reports", "weighting_rationale")
    op.drop_column("appraisal_reports", "weight_residual_pct")
    op.drop_column("appraisal_reports", "weight_income_pct")
    op.drop_column("appraisal_reports", "weight_sales_pct")
