"""Add avm_models table (segment-keyed model registry)

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "avm_models",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("segment", sa.Text(), nullable=False),
        sa.Column("trained_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
        sa.Column("algorithm", sa.Text(), nullable=False),
        sa.Column("feature_columns", JSONB(), nullable=False),
        sa.Column("hyperparams", JSONB(), nullable=False),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column("training_row_count", sa.Integer(), nullable=False),
        sa.Column("min_row_threshold", sa.Integer(), nullable=False),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("quantile_low_path", sa.Text(), nullable=True),
        sa.Column("quantile_high_path", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_avm_models_segment", "avm_models", ["segment"])
    # Only one active model per segment
    op.create_index(
        "uq_avm_models_active_per_segment",
        "avm_models",
        ["segment"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index("uq_avm_models_active_per_segment", table_name="avm_models")
    op.drop_index("ix_avm_models_segment", table_name="avm_models")
    op.drop_table("avm_models")
