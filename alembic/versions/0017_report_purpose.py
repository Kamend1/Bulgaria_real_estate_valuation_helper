"""Add report_purpose to appraisal_reports (differentiates front-matter boilerplate)

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("report_purpose", sa.String(30), nullable=False, server_default="market_opinion"),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "report_purpose")
