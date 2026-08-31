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
    "legal_standard": "Правен/нормативен текст (закон, наредба, етичен кодекс)",
    "other": "Друго",
}

# legal_standard documents skip the compressed-facts extraction below --
# a paraphrased 5-bullet summary of a statute is actively wrong for the
# AppraiserLegalAgent (Phase 11), which needs to quote an exact чл./ал.,
# not a bullet someone else summarized. These get their FULL text stored
# verbatim in extracted_data["full_text"] instead -- see _extract_legal_metadata.
LEGAL_DOCUMENT_TYPE = "legal_standard"


class MarketDocumentFacts(BaseModel):
    source_summary: str = Field(description="Какво представлява документът и кой е издателят/авторът")
    publish_date_or_period: str | None = Field(default=None, description="Дата на публикуване или период, за който се отнася")
    geographic_scope: str | None = Field(default=None, description="Географски обхват -- напр. София, национално, конкретен квартал")
    key_claims: list[str] = Field(default_factory=list, description="Конкретни твърдения/изводи от документа, всяко като отделно кратко твърдение")
    cited_figures: list[str] = Field(default_factory=list, description="Конкретни цифри/статистики, цитирани в документа, с достатъчно контекст (напр. '+8% ръст на цените в Лозенец за 2025 г.')")


class LegalDocumentFacts(BaseModel):
    title: str = Field(description="Официално заглавие на нормативния акт/документа (напр. 'Наредба № 1 от 14.02.2007 г.')")
    issuing_body: str | None = Field(default=None, description="Издаващ/приемащ орган, ако е посочен в текста")
    effective_date_or_period: str | None = Field(default=None, description="Дата на влизане в сила или период на действие, ако е посочен")
    scope_summary: str = Field(description="Едно-две изречения какво урежда документът -- само за ориентация в списъка, не заместител на пълния текст")


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


def _extract_legal_metadata(text: str, provider: str | None, model: str | None) -> dict:
    """Only pulls identifying metadata (title/issuer/date/one-line scope) via
    LLM, from the document's opening -- the actual legal content is stored
    verbatim by the caller, never paraphrased through this call."""
    chat = get_chat_model(provider, model, max_tokens=500)
    structured = chat.with_structured_output(LegalDocumentFacts)
    system = (
        "Извлечи само идентифициращите метаданни на този нормативен/правен документ "
        "(заглавие, издаващ орган, дата, едноизреченско описание на обхвата). Не преразказвай "
        "съдържанието -- пълният текст се пази отделно и дословно."
    )
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=text[:4000])])
    return result.model_dump()


def extract_document(db: Session, doc: MarketDocument, provider: str | None = None, model: str | None = None) -> None:
    """Runs the whole pipeline for one uploaded market document and
    persists the result. Synchronous/blocking -- call from a background
    thread, same convention as documents.extract_document."""
    file_path = storage_dir() / doc.storage_path
    try:
        raw_text = extract_native_text(file_path, doc.mime_type)
        if raw_text is not None:
            method = "text"
        else:
            raw_text = ocr_via_vision(file_path, provider, model)
            method = "ocr_vision"

        if doc.document_type == LEGAL_DOCUMENT_TYPE:
            metadata = _extract_legal_metadata(raw_text, provider, model)
            doc.extracted_data = {**metadata, "full_text": raw_text}
        else:
            doc.extracted_data = _extract_structured_facts(raw_text, doc.document_type, provider, model)
        doc.extraction_method = method
        doc.status = "ready"
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)[:2000]
    db.commit()
