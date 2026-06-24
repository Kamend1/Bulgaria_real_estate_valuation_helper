"""Add concluded_value_residual to appraisal_reports

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("concluded_value_residual", sa.Numeric(14, 2), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "concluded_value_residual")
