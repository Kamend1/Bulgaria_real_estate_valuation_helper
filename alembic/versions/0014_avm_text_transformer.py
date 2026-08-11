"""Add text_transformer_path to avm_models

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-10
"""
from alembic import op
import sqlalchemy as sa

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("avm_models", sa.Column("text_transformer_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("avm_models", "text_transformer_path")
