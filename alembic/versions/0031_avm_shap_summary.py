"""Add avm_models.shap_summary (Phase 14 Tier 2.3, 2026-09-02)

Revision ID: 0031
Revises: 0030
Create Date: 2026-09-02

Global feature-importance summary (top features by mean |SHAP value| over a
training-set sample), computed once per training run via
shap.TreeExplainer on the LightGBM "point" model. Nullable -- older rows
trained before this column existed simply have no summary to show.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("avm_models", sa.Column("shap_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("avm_models", "shap_summary")
