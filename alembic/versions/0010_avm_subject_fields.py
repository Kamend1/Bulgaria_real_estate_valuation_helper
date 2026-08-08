"""Add AVM subject fields to appraisal_reports

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("subject_property_type", sa.Text(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("subject_geo_category", sa.Text(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("subject_neighborhood", sa.Text(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("concluded_value_sales_source", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "concluded_value_sales_source")
    op.drop_column("appraisal_reports", "subject_neighborhood")
    op.drop_column("appraisal_reports", "subject_geo_category")
    op.drop_column("appraisal_reports", "subject_property_type")
