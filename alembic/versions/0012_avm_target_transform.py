"""Add target_transform to avm_models

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "avm_models",
        sa.Column("target_transform", sa.Text(), nullable=False, server_default="raw"),
    )


def downgrade() -> None:
    op.drop_column("avm_models", "target_transform")
