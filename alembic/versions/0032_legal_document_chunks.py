"""Add legal_document_chunks (Phase 14 Tier 3.1, 2026-09-02)

Revision ID: 0032
Revises: 0031
Create Date: 2026-09-02

Mini-RAG for the legal document library, mirroring listing_embeddings'
pattern. Directly addresses a measured cost problem: reading one large
uploaded legal_standard document in full cost 114,000+ input tokens for a
single tool call. search_legal_document (orchestrator_graph.py) replaces
read_legal_document, returning only the top-k semantically relevant
"Чл. N"/"Раздел N" sections for a query instead of the whole document.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "legal_document_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("market_document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("market_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("market_document_id", "chunk_index", "provider", "model", name="uq_legal_document_chunks_doc_idx_provider_model"),
    )
    op.create_index("ix_legal_document_chunks_market_document_id", "legal_document_chunks", ["market_document_id"])


def downgrade() -> None:
    op.drop_index("ix_legal_document_chunks_market_document_id", table_name="legal_document_chunks")
    op.drop_table("legal_document_chunks")
