"""
Backfill legal_document_chunks for legal_standard documents uploaded before
chunking existed (Phase 14 Tier 3.1 -- app/services/market_documents.py::
chunk_and_embed_legal_document, app/services/llm/orchestrator_graph.py::
search_legal_document).

New uploads chunk automatically at extraction time -- this script exists
only to catch up documents already sitting in market_documents with a full
extracted_data["full_text"] but zero rows in legal_document_chunks (or to
re-chunk after switching embedding provider/model).

Usage (from project root):
    python -m scripts.chunk_legal_documents             # backfill every legal_standard doc missing chunks
    python -m scripts.chunk_legal_documents --force      # re-chunk even docs that already have chunks

Requires OPENAI_API_KEY (or whatever llm_default_embedding_provider points
to) in .env -- real cost, each call is billed by the provider. Cheap in
practice: text-embedding-3-small at $0.02/1M tokens, and legal documents
are typically a few hundred KB at most.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.db.models import LegalDocumentChunk, MarketDocument
from app.db.session import db_session
from app.services.market_documents import LEGAL_DOCUMENT_TYPE, chunk_and_embed_legal_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill legal_document_chunks for uploaded legal_standard documents")
    parser.add_argument("--force", action="store_true", help="Re-chunk documents that already have chunks")
    args = parser.parse_args()

    with db_session() as db:
        docs = (
            db.query(MarketDocument)
            .filter(MarketDocument.document_type == LEGAL_DOCUMENT_TYPE, MarketDocument.status == "ready")
            .all()
        )
        print(f"Found {len(docs)} ready legal_standard document(s).")
        for doc in docs:
            existing = db.query(LegalDocumentChunk).filter(LegalDocumentChunk.market_document_id == doc.id).count()
            if existing and not args.force:
                print(f"  {doc.filename}: already has {existing} chunk(s), skipping (--force to redo).")
                continue
            full_text = (doc.extracted_data or {}).get("full_text") or ""
            if not full_text:
                print(f"  {doc.filename}: no full_text stored, skipping.")
                continue
            n = chunk_and_embed_legal_document(db, doc, full_text)
            print(f"  {doc.filename}: {len(full_text)} chars -> {n} chunk(s).")

    print("Done.")


if __name__ == "__main__":
    main()
