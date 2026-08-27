"""Add agent_conversations/agent_messages (Tier 2, multi-agent chat console)

Revision ID: 0024
Revises: 0023
Create Date: 2026-08-26

One conversation per report (v1 scope, per the owner's own framing -- the
end goal is still writing an appraisal report, so the assistant lives in
that context, not as a report-agnostic general chat). agent_messages stores
enough to reconstruct the LangChain message list on the next turn: role,
content, and (for an assistant message that called tools) tool_calls
JSONB, or (for a tool-result message) which call it answers.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversations",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PGUUID(as_uuid=True), sa.ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
        sa.Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_agent_conversations_report_id", "agent_conversations", ["report_id"])

    op.create_table(
        "agent_messages",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", PGUUID(as_uuid=True), sa.ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),   # user | assistant | tool
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("tool_calls", JSONB(), nullable=True),   # [{id, name, args}], assistant messages that called tools
        sa.Column("tool_call_id", sa.Text(), nullable=True),   # tool messages: which call this answers
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_agent_messages_conversation_id", "agent_messages", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_messages_conversation_id", table_name="agent_messages")
    op.drop_table("agent_messages")
    op.drop_index("ix_agent_conversations_report_id", table_name="agent_conversations")
    op.drop_table("agent_conversations")
