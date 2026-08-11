"""Add CatBoost companion + blend_weight fields to avm_models

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-10
"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("avm_models", sa.Column("companion_algorithm", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_model_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_quantile_low_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_quantile_high_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("blend_weight", sa.Numeric(4, 3), nullable=True))


def downgrade() -> None:
    op.drop_column("avm_models", "blend_weight")
    op.drop_column("avm_models", "companion_quantile_high_path")
    op.drop_column("avm_models", "companion_quantile_low_path")
    op.drop_column("avm_models", "companion_model_path")
    op.drop_column("avm_models", "companion_algorithm")
