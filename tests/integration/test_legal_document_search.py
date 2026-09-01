"""
Integration tests for the legal-document mini-RAG (Phase 14 Tier 3.1):
app/services/market_documents.py::chunk_and_embed_legal_document and
app/services/llm/orchestrator_graph.py::_search_legal_document_fn.

Embeddings are mocked (deterministic small vectors, no real API calls/cost)
-- real end-to-end verification against actual OpenAI embeddings + a real
uploaded document was done manually in this session (see chat history);
these tests lock in the DB-level behavior (chunk storage, upsert-by-
clearing, ordering, fallback) independent of embedding cost.
"""
from unittest.mock import MagicMock, patch

from app.db.models import LegalDocumentChunk, MarketDocument
from app.services import market_documents
from app.services.llm import orchestrator_graph

# legal_document_chunks.embedding is vector(1536) -- pgvector enforces exact
# dimensionality on insert, so toy vectors still need the real width, just
# with an obvious "signal" in the first two dims to control cosine distance.
_DIM = 1536
_VEC_A = [1.0, 0.0] + [0.0] * (_DIM - 2)
_VEC_B = [0.0, 1.0] + [0.0] * (_DIM - 2)
_VEC_QUERY_NEAR_A = [0.9, 0.1] + [0.0] * (_DIM - 2)


def _make_legal_doc(db_session) -> MarketDocument:
    doc = MarketDocument(
        filename="test.txt", document_type="legal_standard",
        storage_path="does-not-matter.txt", mime_type="text/plain", status="ready",
        extracted_data={"full_text": "Чл. 1. Първи текст.\nЧл. 2. Втори текст."},
    )
    db_session.add(doc)
    db_session.commit()
    return doc


def _fake_embeddings_model(vectors: list[list[float]], query_vector: list[float] | None = None):
    fake = MagicMock()
    fake.embed_documents.return_value = vectors
    fake.embed_query.return_value = query_vector or vectors[0]
    return fake


def test_chunk_and_embed_stores_one_row_per_chunk(db_session):
    doc = _make_legal_doc(db_session)
    fake = _fake_embeddings_model([_VEC_A, _VEC_B])
    with patch.object(market_documents, "resolve_embedding_model", return_value=("openai", "text-embedding-3-small")), \
         patch.object(market_documents, "get_embeddings_model", return_value=fake):
        n = market_documents.chunk_and_embed_legal_document(db_session, doc, doc.extracted_data["full_text"])

    assert n == 2
    rows = db_session.query(LegalDocumentChunk).filter(
        LegalDocumentChunk.market_document_id == doc.id
    ).order_by(LegalDocumentChunk.chunk_index).all()
    assert len(rows) == 2
    assert rows[0].heading == "Чл. 1."
    assert rows[1].heading == "Чл. 2."


def test_chunk_and_embed_is_idempotent_clears_old_rows_first(db_session):
    doc = _make_legal_doc(db_session)
    fake = _fake_embeddings_model([_VEC_A, _VEC_B])
    with patch.object(market_documents, "resolve_embedding_model", return_value=("openai", "text-embedding-3-small")), \
         patch.object(market_documents, "get_embeddings_model", return_value=fake):
        market_documents.chunk_and_embed_legal_document(db_session, doc, doc.extracted_data["full_text"])
        market_documents.chunk_and_embed_legal_document(db_session, doc, doc.extracted_data["full_text"])

    rows = db_session.query(LegalDocumentChunk).filter(LegalDocumentChunk.market_document_id == doc.id).all()
    assert len(rows) == 2   # not 4 -- the second call must not duplicate


def test_chunk_and_embed_returns_zero_for_empty_text(db_session):
    doc = _make_legal_doc(db_session)
    n = market_documents.chunk_and_embed_legal_document(db_session, doc, "")
    assert n == 0


def test_search_legal_document_returns_nearest_sections_first(db_session):
    doc = _make_legal_doc(db_session)
    embed_fake = _fake_embeddings_model([_VEC_A, _VEC_B])
    with patch.object(market_documents, "resolve_embedding_model", return_value=("openai", "text-embedding-3-small")), \
         patch.object(market_documents, "get_embeddings_model", return_value=embed_fake):
        market_documents.chunk_and_embed_legal_document(db_session, doc, doc.extracted_data["full_text"])

    query_fake = _fake_embeddings_model([_VEC_A], query_vector=_VEC_QUERY_NEAR_A)
    with patch.object(orchestrator_graph, "resolve_embedding_model", return_value=("openai", "text-embedding-3-small")), \
         patch.object(orchestrator_graph, "get_embeddings_model", return_value=query_fake):
        search = orchestrator_graph._search_legal_document_fn(db_session)
        result = search(str(doc.id), "въпрос близък до чл. 1", k=5)

    assert "sections" in result
    assert result["sections"][0]["heading"] == "Чл. 1."  # nearest to _VEC_A
    assert result["sections"][1]["heading"] == "Чл. 2."


def test_search_legal_document_falls_back_to_full_text_when_no_chunks_indexed(db_session):
    doc = _make_legal_doc(db_session)  # never chunked
    query_fake = _fake_embeddings_model([_VEC_A], query_vector=_VEC_A)
    with patch.object(orchestrator_graph, "resolve_embedding_model", return_value=("openai", "text-embedding-3-small")), \
         patch.object(orchestrator_graph, "get_embeddings_model", return_value=query_fake):
        search = orchestrator_graph._search_legal_document_fn(db_session)
        result = search(str(doc.id), "каквото и да е", k=5)

    assert "sections" not in result
    assert "fallback_full_text" in result
    assert "Чл. 1." in result["fallback_full_text"]


def test_search_legal_document_missing_document_returns_error(db_session):
    search = orchestrator_graph._search_legal_document_fn(db_session)
    result = search("00000000-0000-0000-0000-000000000000", "каквото и да е")
    assert "error" in result
