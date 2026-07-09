"""Add log_text to scrape_runs

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-09
"""
from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scrape_runs",
        sa.Column("log_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("scrape_runs", "log_text")
