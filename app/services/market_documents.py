"""Market-research document library (Phase 10, 2026-08-28) -- upload/
extraction for the market analyst agent's own reference library (market
reports, research articles, official statistics). Deliberately separate
from app/services/documents.py's report-scoped documents: this library is
NOT tied to any report, shared across all users (see MarketDocument's own
docstring in app/db/models.py).

Reuses the shared PDF/DOCX/OCR mechanics from app/services/llm/doc_extraction.py
(the exact same code path app/services/documents.py uses) -- only the
document-type vocabulary and the extraction schema differ here.

Extraction stays purely descriptive ("what does this document say") --
cross-referencing it against live imot.bg data happens dynamically in
conversation via the query_market_stats tool, not baked in at upload time,
so it never goes stale as the corpus grows.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import MarketDocument
from app.services.llm.doc_extraction import extract_native_text, ocr_via_vision
from app.services.llm.providers import get_chat_model

DOCUMENT_TYPE_LABELS = {
    "market_report": "Пазарен анализ / доклад",
    "research_article": "Изследователска статия",
    "government_statistic": "Официална статистика (НСИ, БНБ и др.)",
    "news": "Новина / медийна публикация",
    "other": "Друго",
}


class MarketDocumentFacts(BaseModel):
    source_summary: str = Field(description="Какво представлява документът и кой е издателят/авторът")
    publish_date_or_period: str | None = Field(default=None, description="Дата на публикуване или период, за който се отнася")
    geographic_scope: str | None = Field(default=None, description="Географски обхват -- напр. София, национално, конкретен квартал")
    key_claims: list[str] = Field(default_factory=list, description="Конкретни твърдения/изводи от документа, всяко като отделно кратко твърдение")
    cited_figures: list[str] = Field(default_factory=list, description="Конкретни цифри/статистики, цитирани в документа, с достатъчно контекст (напр. '+8% ръст на цените в Лозенец за 2025 г.')")


def storage_dir() -> Path:
    p = Path(settings.market_documents_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_upload(uploaded_by: int, filename: str, file_bytes: bytes) -> str:
    """Writes the uploaded file to disk, returns storage_path (relative to
    storage_dir()). No per-report subdirectory -- this library is flat and
    shared, unlike app/services/documents.py's report-scoped layout."""
    ext = Path(filename).suffix.lower() or ""
    stored_name = f"{uuid.uuid4().hex}{ext}"
    (storage_dir() / stored_name).write_bytes(file_bytes)
    return stored_name


def _extract_structured_facts(text: str, document_type: str, provider: str | None, model: str | None) -> dict:
    label = DOCUMENT_TYPE_LABELS.get(document_type, document_type)
    chat = get_chat_model(provider, model, max_tokens=1500)
    structured = chat.with_structured_output(MarketDocumentFacts)
    system = (
        f"Извлечи фактите от този документ ({label}), релевантен за пазара на недвижими имоти "
        "в България. Ползвай само това, което реално пише в текста -- не допълвай и не "
        "предполагай. Обърни особено внимание на конкретни цифри/проценти/тенденции и "
        "географския им обхват -- те са най-полезни за сравнение с реални пазарни данни по-късно."
    )
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=text[:20000])])
    return result.model_dump()


def extract_document(db: Session, doc: MarketDocument, provider: str | None = None, model: str | None = None) -> None:
    """Runs the whole pipeline for one uploaded market document and
    persists the result. Synchronous/blocking -- call from a background
    thread, same convention as documents.extract_document."""
    file_path = storage_dir() / doc.storage_path
    try:
        raw_text = extract_native_text(file_path, doc.mime_type)
        if raw_text is not None:
            facts = _extract_structured_facts(raw_text, doc.document_type, provider, model)
            method = "text"
        else:
            ocr_text = ocr_via_vision(file_path, provider, model)
            facts = _extract_structured_facts(ocr_text, doc.document_type, provider, model)
            method = "ocr_vision"
        doc.extraction_method = method
        doc.extracted_data = facts
        doc.status = "ready"
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)[:2000]
    db.commit()
