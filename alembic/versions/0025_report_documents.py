"""Add report_documents (Tier 3, document upload/extraction)

Revision ID: 0025
Revises: 0024
Create Date: 2026-08-26

Scoped to a report (not a standalone "Project" concept) -- v1 scope per the
owner's own framing, same as agent_conversations. document_type drives
which extraction schema/prompt applies (see app/services/documents.py):
notarial_act/founding_document/partnership_agreement get text extraction +
structured fact extraction, sketch gets vision-based room/area extraction.
extracted_data is deliberately generic JSONB, not per-type columns -- the
schema differs by document_type and is expected to grow.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "report_documents",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PGUUID(as_uuid=True), sa.ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("uploaded_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),   # notarial_act | founding_document | partnership_agreement | sketch | other
        sa.Column("storage_path", sa.Text(), nullable=False),    # relative to settings.documents_dir
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="processing"),   # processing | ready | failed
        sa.Column("extraction_method", sa.Text(), nullable=True),   # text | ocr_vision
        sa.Column("extracted_data", JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    )
    op.create_index("ix_report_documents_report_id", "report_documents", ["report_id"])


def downgrade() -> None:
    op.drop_index("ix_report_documents_report_id", table_name="report_documents")
    op.drop_table("report_documents")
