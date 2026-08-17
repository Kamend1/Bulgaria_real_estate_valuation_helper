"""Add cadastral id + legal description fields to appraisal_reports

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-17
"""
from alembic import op
import sqlalchemy as sa

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("subject_cadastral_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("legal_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("legal_description_source", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "legal_description_source")
    op.drop_column("appraisal_reports", "legal_description")
    op.drop_column("appraisal_reports", "subject_cadastral_id")
