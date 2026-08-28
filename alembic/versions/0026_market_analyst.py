"""Add market_documents + generalize agent_conversations for the market
analyst agent (Phase 10)

Revision ID: 0026
Revises: 0025
Create Date: 2026-08-28

Second, report-agnostic AI agent for free-form market research (time
series, segment/geo/construction-type cross-sections) -- distinct from the
existing report-scoped assistant. Reuses agent_conversations/agent_messages/
agent_llm_calls rather than duplicating them: report_id becomes nullable,
a new agent_type column distinguishes the two conversation kinds. Existing
rows get the default 'report_assistant' (their only kind so far), so no
data backfill is needed beyond the column default.

market_documents is a new, separate table (not report_documents) for the
market analyst's own reference library (market reports/articles/official
statistics) -- explicitly NOT tied to a report, shared across all users
like the underlying listings corpus itself.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("agent_conversations", "report_id", nullable=True)
    op.add_column(
        "agent_conversations",
        sa.Column("agent_type", sa.Text(), nullable=False, server_default="report_assistant"),
    )

    op.create_table(
        "market_documents",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("uploaded_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="processing"),
        sa.Column("extraction_method", sa.Text(), nullable=True),
        sa.Column("extracted_data", JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("market_documents")
    op.drop_column("agent_conversations", "agent_type")
    op.alter_column("agent_conversations", "report_id", nullable=False)
