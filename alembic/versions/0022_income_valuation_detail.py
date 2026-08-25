"""Add income_valuation_detail/income_valuation_source to appraisal_reports

Revision ID: 0022
Revises: 0021
Create Date: 2026-08-25

Audit finding (2026-08-25): neither the manual income-approach calculator
(_income_analysis.html, pure client-side JS) nor the AI-computed income
valuation (compute_income_valuation tool result) ever persisted their full
NOI/direct-capitalization/DCF/sensitivity detail anywhere -- only the single
final concluded_value_income number survived past the browser session, so
generate_docx()/export_excel() had nothing to render beyond that bare
number. income_valuation_detail stores the full compute_income_valuation()
result (the SAME Python function now used by both the manual save route and
the AI tool, eliminating the JS/Python duplicate-formula drift risk this
audit also found); income_valuation_source records which path produced it
("manual" | "ai"), mirroring the existing concluded_value_sales_source /
legal_description_source provenance-tagging pattern.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "appraisal_reports",
        sa.Column("income_valuation_detail", JSONB(), nullable=True),
    )
    op.add_column(
        "appraisal_reports",
        sa.Column("income_valuation_source", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("appraisal_reports", "income_valuation_source")
    op.drop_column("appraisal_reports", "income_valuation_detail")
