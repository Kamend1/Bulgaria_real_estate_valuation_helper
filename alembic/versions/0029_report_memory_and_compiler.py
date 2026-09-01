"""Add report_agent_findings/report_compile_runs (Phase 13 -- report memory)

Revision ID: 0029
Revises: 0028
Create Date: 2026-09-01

Closes a real gap found while auditing the Supervisor graph: specialist
findings (OrchestratorState.findings) only ever lived for the duration of
one chat turn, and separate conversations on the same report had zero
awareness of each other (see AgentConversation's Phase 12 multi-conversation
docstring). report_agent_findings is an append-only log (mirrors
listing_snapshots/ai_valuation_runs -- history of rows, not one blob
overwritten in place) of every specialist's final answer, taggable by
source ("chat" or "compile") -- read by get_report_memory() so both the
Supervisor's routing prompt and the new Report Compiler action can see what
was already established about a report, regardless of which conversation
(or non-conversation compile run) produced it.

report_compile_runs is the provenance record for the new "Компилирай
доклада" action (comparables.py) -- which specialists were requested, on
which report, status -- source_id in report_agent_findings points here when
source='compile' (loosely, no FK: the two possible source tables are
mutually exclusive per row, same non-hard-FK provenance pattern already
used by ai_valuation_runs.output).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "report_compile_runs",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PGUUID(as_uuid=True), sa.ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("requested_domains", ARRAY(sa.Text()), nullable=False),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="running"),   # running | done | error
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("results", JSONB(), nullable=True),   # {domain: {"text": str, "proposals": [...]}} -- filled once status='done'
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_report_compile_runs_report_id", "report_compile_runs", ["report_id"])

    op.create_table(
        "report_agent_findings",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PGUUID(as_uuid=True), sa.ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("domain", sa.Text(), nullable=False),   # income | market | market_analysis | legal | auditor
        sa.Column("source", sa.Text(), nullable=False),   # 'chat' | 'compile'
        sa.Column("source_id", PGUUID(as_uuid=True), nullable=True),   # agent_conversations.id or report_compile_runs.id, no hard FK (polymorphic)
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_report_agent_findings_report_domain", "report_agent_findings", ["report_id", "domain", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_report_agent_findings_report_domain", table_name="report_agent_findings")
    op.drop_table("report_agent_findings")
    op.drop_index("ix_report_compile_runs_report_id", table_name="report_compile_runs")
    op.drop_table("report_compile_runs")
