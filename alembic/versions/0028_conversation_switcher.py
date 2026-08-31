"""Multiple named chat conversations + truncation flag (Phase 12)

Revision ID: 0028
Revises: 0027
Create Date: 2026-08-31

Two independent additions found necessary while diagnosing "responses cut
off mid-sentence" (see plan doc Phase 12):
- agent_messages.truncated: set when a specialist's final answer was cut
  short by the provider's max-tokens ceiling (finish_reason/stop_reason ==
  length/max_tokens), so the UI can show a visible warning instead of
  silently treating a cut-off answer as complete.
- ix_agent_conversations_user_agent_report: a composite index needed once
  users can have MORE than one conversation per (user, agent_type,
  report_id) -- today's get_or_create_* always reuses the single existing
  row, so this query pattern never mattered before; letting people start
  new named conversations means it will be hit on every page load.
"""
from alembic import op
import sqlalchemy as sa

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_messages",
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index(
        "ix_agent_conversations_user_agent_report",
        "agent_conversations",
        ["user_id", "agent_type", "report_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_conversations_user_agent_report", table_name="agent_conversations")
    op.drop_column("agent_messages", "truncated")
