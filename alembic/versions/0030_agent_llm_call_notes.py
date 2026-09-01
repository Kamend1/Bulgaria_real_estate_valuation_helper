"""Add agent_llm_calls.notes (Phase 13 cleanup, 2026-09-01)

Revision ID: 0030
Revises: 0029
Create Date: 2026-09-01

The Supervisor's routing decision (RouteDecision.reasoning) was generated
by the model on every routing call but discarded immediately after picking
next_specialist -- never logged anywhere, confirmed by grep during the
Phase 13 audit. This column gives it somewhere to live, so a supervisor's
call_log row is reviewable later (why did it route here?), not opaque.
Nullable and unused by every other call type -- only the supervisor node
populates it.
"""
from alembic import op
import sqlalchemy as sa

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_llm_calls", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_llm_calls", "notes")
