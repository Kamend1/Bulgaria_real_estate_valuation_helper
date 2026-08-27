"""Add agent_llm_calls: per-call token/cost ledger (Tier 1, multi-agent chat)

Revision ID: 0023
Revises: 0022
Create Date: 2026-08-26

ai_valuation_runs logs ONE row per generate_valuation_backbone() invocation,
with TOTAL tokens summed across every internal LLM call in that run's
tool-calling loop (get_pool_stats decision, compute_income_valuation
decision, final narrative, ...) -- useful for "how much did this generation
cost", not for "which step ate the budget". That distinction mattered
directly during the 2026-08-25 gpt-5.4-pro truncation audit, where the
reasoning-heavy final call was the real culprit but the aggregate number
alone couldn't show that.

agent_llm_calls adds that per-call detail: one row per individual LLM API
call, optionally linked to the ai_valuation_runs row it was part of.
conversation_id is unused for now (nullable) -- reserved for Tier 2's chat
console, where calls won't belong to a single run but to an ongoing,
possibly-multi-agent conversation.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_llm_calls",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("ai_valuation_run_id", PGUUID(as_uuid=True), sa.ForeignKey("ai_valuation_runs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("conversation_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("call_label", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_agent_llm_calls_run_id", "agent_llm_calls", ["ai_valuation_run_id"])
    op.create_index("ix_agent_llm_calls_conversation_id", "agent_llm_calls", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_llm_calls_conversation_id", table_name="agent_llm_calls")
    op.drop_index("ix_agent_llm_calls_run_id", table_name="agent_llm_calls")
    op.drop_table("agent_llm_calls")
