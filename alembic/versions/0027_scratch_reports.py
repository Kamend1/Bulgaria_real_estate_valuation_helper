"""Add appraisal_reports.is_scratch (Phase 11 -- hypothetical scenarios)

Revision ID: 0027
Revises: 0026
Create Date: 2026-08-28

A "hypothetical property" conversation needs the full report toolset
(subject fields, income approach, AI assistant, comparables/AVM/GIS
panels) without cluttering the real /reports/ list. Rather than a
parallel schema, a scratch report is a completely normal AppraisalReport
row with is_scratch=True -- /reports/ filters it out by default, and
"Направи истински доклад" just flips the flag back to False. Existing
rows default to False (real reports, unaffected).
"""
from alembic import op
import sqlalchemy as sa

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("is_scratch", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "is_scratch")
