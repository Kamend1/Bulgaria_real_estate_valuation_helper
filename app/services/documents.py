"""Document upload/extraction (Tier 3, multi-agent chat, 2026-08-26).

Two genuinely different extraction paths, per the owner's own framing of
the problem:
  - Legal/administrative/text documents (notarial act, company founding/
    partnership documents, permits, certificates, leases, powers of
    attorney, court decisions, etc. -- see DOCUMENT_TYPE_LABELS for the
    full realistic range, deliberately broader than the handful of
    examples first discussed during design) are TEXT: native extraction
    first (PyMuPDF/python-docx), vision-LLM OCR only as a fallback when
    there's no text layer (a scanned document). Then a structured-
    extraction LLM call (.with_structured_output()) -- a few common types
    (notarial_act, company docs) get their own bespoke Pydantic schema;
    everything else shares GeneralDocumentFacts + a type-aware prompt hint
    (see _GENERIC_EXTRACTION_HINTS), which scales to new document types
    without a bespoke schema per type. Unlike the narrative-generation
    prompts elsewhere in this app, which deliberately avoid
    structured_output in favor of parsed markdown sections, extraction has
    no narrative/reasoning to lose from a rigid schema.
  - A скица (floor plan/architectural drawing) is fundamentally VISUAL, not
    a RAG/text problem: a vision-LLM call reads it CRITICALLY and broadly --
    room layout/areas/types, a layout assessment (functional? well-
    proportioned? natural light?), and whether it's consistent with the
    report's own declared subject data -- not just "list rooms and flag
    terraces" (an earlier version over-fit to that one example; broadened
    2026-08-26). Terrace area is the one piece with a real regulated
    answer -- what it counts as toward assessed area -- so that ONE
    calculation stays DETERMINISTIC Python (TERRACE_AREA_COEFFICIENT),
    mirroring this app's established guardrail that models read/interpret,
    Python computes; it's an optional sub-analysis alongside the broader
    qualitative read, not the sketch reading's whole purpose.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ReportDocument
from app.services.llm.doc_extraction import (
    MAX_OCR_PAGES,
    MIN_NATIVE_TEXT_CHARS,
    b64_image as _b64_image,
    extract_native_text as _extract_native_text,
    ocr_via_vision as _ocr_via_vision,
    render_pdf_pages_as_png as _render_pdf_pages_as_png,
)
from app.services.llm.providers import get_chat_model

# Bulgarian appraisal practice: covered/uncovered terraces and balconies
# count at a reduced percentage toward assessed/usable area, not their full
# footprint. This is a REGULATED convention, not a judgment call -- the
# vision model's job is only to correctly read room areas off the sketch;
# this coefficient (not the model) decides what that means for total area.
TERRACE_AREA_COEFFICIENT = 0.3


# ── Structured extraction schemas (one per document_type) ──────────────────────

class NotarialActFacts(BaseModel):
    owner_names: list[str] = Field(default_factory=list, description="Имена на собствениците/приобретателите по акта")
    cadastral_id: str | None = Field(default=None, description="Кадастрален идентификатор, ако е посочен")
    property_description: str | None = Field(default=None, description="Кратко описание на имота по акта (адрес, вид, граници)")
    area_sqm: float | None = Field(default=None, description="Площ на имота по акта, ако е посочена")
    encumbrances: str | None = Field(default=None, description="Тежести/ипотеки/възбрани върху имота, ако има")
    act_date: str | None = Field(default=None, description="Дата на нотариалния акт")


class CompanyDocumentFacts(BaseModel):
    company_name: str | None = Field(default=None, description="Наименование на дружеството")
    registered_address: str | None = Field(default=None, description="Седалище и адрес на управление")
    capital_amount: float | None = Field(default=None, description="Размер на капитала")
    capital_currency: str | None = Field(default=None, description="Валута на капитала (обикновено BGN)")
    founders_or_partners: list[str] = Field(default_factory=list, description="Учредители/съдружници")
    relevant_notes: str | None = Field(default=None, description="Друга информация, релевантна за непарична вноска в капитала")


class SketchRoom(BaseModel):
    name: str = Field(description="Име/номер на помещението, както е изписано на скицата")
    area_sqm: float | None = Field(default=None, description="Площ на помещението в кв.м, ако е изписана или изчислима по мащаба")
    room_type: str = Field(description="Свободен тип, напр. 'спалня', 'дневна', 'кухня', 'баня', 'коридор', 'тераса/балкон', 'склад', 'гараж', 'стълбище', 'друго'")


class SketchFacts(BaseModel):
    """Deliberately broader than just "list rooms and flag terraces" -- a
    скица/архитектурен чертеж is read critically for whatever it actually
    shows, terraces being just one possible detail among several, not the
    whole point (audit finding 2026-08-26: the first version over-fit the
    schema to the one terrace-area example given during design, losing the
    general "critically read this drawing" ask)."""
    drawing_type: str | None = Field(default=None, description="Тип на чертежа -- напр. разпределение на етаж, кадастрална скица, фасада, разрез")
    rooms: list[SketchRoom] = Field(default_factory=list)
    layout_assessment: str | None = Field(default=None, description="Критична оценка на разпределението -- функционалност, пропорции, естествена светлина, забелязани особености")
    consistency_notes: str | None = Field(default=None, description="Съответства ли изчертаното на декларираните данни за имота (площ, брой стаи), ако е имало такива")
    other_observations: str | None = Field(default=None, description="Друго забелязано -- асансьор, паркомясто, нередности в самия чертеж и т.н.")


class GeneralDocumentFacts(BaseModel):
    """Fallback schema for every document_type below that doesn't have its
    own bespoke schema (see the module docstring's 2026-08-26 note): a
    fixed structured schema per type doesn't scale to the realistically
    wide range of documents an appraiser actually encounters (permits,
    certificates, leases, powers of attorney, court decisions...), and
    building a bespoke one for each is over-engineering something a
    flexible schema + a type-specific prompt hint already covers well.
    notarial_act/company docs/sketch keep their own richer schemas because
    those are common enough and structured enough (fixed fields like
    cadastral_id, capital_amount) to earn it."""
    document_summary: str = Field(description="Кратко резюме какво представлява документът и какво съдържа")
    key_facts: list[str] = Field(default_factory=list, description="Ключови факти/данни от документа, всеки като отделно кратко твърдение")
    parties_or_names: list[str] = Field(default_factory=list, description="Споменати лица/страни/институции, ако има")
    dates: list[str] = Field(default_factory=list, description="Съществени дати, споменати в документа")
    relevance_to_valuation: str | None = Field(default=None, description="Как това е релевантно за оценката на имота, ако изобщо е")
    concerns_or_flags: str | None = Field(default=None, description="Забелязани несъответствия, липси или поводи за внимание")


_SCHEMAS: dict[str, type[BaseModel]] = {
    "notarial_act": NotarialActFacts,
    "founding_document": CompanyDocumentFacts,
    "partnership_agreement": CompanyDocumentFacts,
    "sketch": SketchFacts,
}

# Deliberately broader than the handful of examples first discussed during
# design (нотариален акт / учредителен документ / дружествен договор /
# скица) -- those were illustrative, not exhaustive, and the upload form
# over-fit to them literally (audit finding 2026-08-26, mirroring the same
# over-literal pattern already corrected once for the sketch schema itself,
# see SketchFacts' own docstring). This list is the realistic range of
# documents that actually support a Bulgarian real estate appraisal.
DOCUMENT_TYPE_LABELS = {
    "notarial_act": "Нотариален акт (документ за собственост)",
    "sketch": "Скица / архитектурен чертеж",
    "cadastral_extract": "Скица-извадка / удостоверение от СГКК",
    "construction_permit": "Разрешение за строеж",
    "occupancy_certificate": "Удостоверение за въвеждане в експлоатация / търпимост",
    "tax_assessment": "Удостоверение за данъчна оценка",
    "technical_passport": "Технически паспорт на сградата",
    "founding_document": "Учредителен акт (дружество)",
    "partnership_agreement": "Дружествен договор",
    "trade_registry_extract": "Актуално състояние (Търговски регистър)",
    "lease_agreement": "Договор за наем",
    "power_of_attorney": "Пълномощно",
    "court_decision": "Съдебно решение / определение",
    "other": "Друго",
}

_EXTRACTION_SYSTEM_PROMPTS = {
    "notarial_act": (
        "Извлечи фактите от този нотариален акт за недвижим имот. Ползвай само това, "
        "което реално пише в текста -- не допълвай и не предполагай."
    ),
    "founding_document": (
        "Извлечи фактите от този учредителен документ на дружество. Фокусирай се върху "
        "данни, релевантни за оценка на непарична вноска (чл. 72 ТЗ): капитал, седалище, "
        "учредители."
    ),
    "partnership_agreement": (
        "Извлечи фактите от този дружествен договор. Фокусирай се върху данни, релевантни "
        "за оценка на непарична вноска: капитал, седалище, съдружници."
    ),
}

# Per-type hints folded into the GENERIC extraction prompt for every
# document_type that isn't in _EXTRACTION_SYSTEM_PROMPTS above -- keeps the
# extraction genuinely type-aware without a bespoke Pydantic schema per
# type. "other" gets the generic hint too, which also fixes a real latent
# bug: before this, "other" was a selectable option with no _SCHEMAS entry
# at all, so choosing it crashed extraction (caught only as a stored
# status="failed" row, never surfaced to the uploader as "unsupported").
_GENERIC_EXTRACTION_HINTS = {
    "cadastral_extract": "Фокусирай се върху идентификатор, площ, адрес, собственици и граници по кадастъра.",
    "construction_permit": "Фокусирай се върху разрешения обем строителство, издател, номер и дата на разрешението, валидност.",
    "occupancy_certificate": "Фокусирай се върху вида на удостоверението (въвеждане в експлоатация / търпимост), издател, дата, обхват.",
    "tax_assessment": "Фокусирай се върху данъчната оценка (стойност), дата на издаване, издаващ орган, данни за имота.",
    "technical_passport": "Фокусирай се върху конструкция, година на строеж, етажност, инсталации, констатирано техническо състояние.",
    "trade_registry_extract": "Фокусирай се върху капитал, управители/съдружници, седалище и статус на дружеството.",
    "lease_agreement": "Фокусирай се върху наемна цена, срок, страни и предмет на договора -- полезно за доходния подход.",
    "power_of_attorney": "Фокусирай се върху упълномощител/пълномощник, обхват на пълномощията, срок, дата.",
    "court_decision": "Фокусирай се върху страните, предмета на делото, разпоредителната част, дата и влизане в сила.",
    "other": "Извлечи каквото е релевантно за оценката на имота.",
}


def _generic_extraction_prompt(document_type: str) -> str:
    label = DOCUMENT_TYPE_LABELS.get(document_type, document_type)
    hint = _GENERIC_EXTRACTION_HINTS.get(document_type, "")
    return (
        f"Извлечи ключовите факти от този документ ({label}), релевантни за оценка на "
        f"недвижим имот. Ползвай само това, което реално пише в текста -- не допълвай и "
        f"не предполагай. {hint}"
    ).strip()


def storage_dir() -> Path:
    p = Path(settings.documents_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_structured_facts(text: str, document_type: str, provider: str | None, model: str | None) -> dict:
    # Falls back to GeneralDocumentFacts + a generic type-aware prompt for
    # any document_type without its own bespoke schema/prompt (see both
    # dicts' own docstrings) -- every entry in DOCUMENT_TYPE_LABELS is
    # extractable this way, never a KeyError.
    schema = _SCHEMAS.get(document_type, GeneralDocumentFacts)
    chat = get_chat_model(provider, model, max_tokens=1500)
    structured = chat.with_structured_output(schema)
    system = _EXTRACTION_SYSTEM_PROMPTS.get(document_type) or _generic_extraction_prompt(document_type)
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=text[:20000])])
    return result.model_dump()


_TERRACE_KEYWORDS = ("тераса", "балкон", "veranda", "тер.", "балк.")


def compute_sketch_area_summary(facts: dict) -> dict | None:
    """Deterministic (never the model) -- see TERRACE_AREA_COEFFICIENT's own
    docstring for why this one specific calculation stays in Python, even
    though room classification itself is now free-text (matches by keyword
    substring, not a strict 'terrace' enum value the model must hit
    exactly). This is one OPTIONAL sub-analysis alongside SketchFacts'
    broader layout_assessment/consistency_notes -- not the sketch reading's
    whole purpose. Returns None if no room has a usable area_sqm (nothing
    to compute -- the broader qualitative fields still carry the read)."""
    rooms = facts.get("rooms") or []
    priced_rooms = [r for r in rooms if r.get("area_sqm") is not None]
    if not priced_rooms:
        return None
    terrace_total = sum(r["area_sqm"] for r in priced_rooms if any(kw in (r.get("room_type") or "").lower() for kw in _TERRACE_KEYWORDS))
    non_terrace_total = sum(r["area_sqm"] for r in priced_rooms) - terrace_total
    corrected = non_terrace_total + terrace_total * TERRACE_AREA_COEFFICIENT
    return {
        "non_terrace_area_sqm": round(non_terrace_total, 2),
        "terrace_area_sqm": round(terrace_total, 2),
        "terrace_coefficient": TERRACE_AREA_COEFFICIENT,
        "corrected_total_area_sqm": round(corrected, 2),
        "raw_total_area_sqm": round(non_terrace_total + terrace_total, 2),
    }


def extract_document(db: Session, doc: ReportDocument, provider: str | None = None, model: str | None = None) -> None:
    """Runs the whole pipeline for one uploaded document and persists the
    result (status/extraction_method/extracted_data or error_message).
    Synchronous/blocking -- call from a background thread, same convention
    as valuation_chain.generate_valuation_backbone."""
    file_path = storage_dir() / doc.storage_path
    try:
        if doc.document_type == "sketch":
            # Always vision -- a sketch has no "native text" path to try.
            facts = _vision_structured_extract_sketch(file_path, doc.report, provider, model)
            method = "ocr_vision"
            facts["area_summary"] = compute_sketch_area_summary(facts)
            doc.extraction_method = method
            doc.extracted_data = facts
            doc.status = "ready"
            db.commit()
            return

        raw_text = _extract_native_text(file_path, doc.mime_type)
        if raw_text is not None:
            facts = _extract_structured_facts(raw_text, doc.document_type, provider, model)
            method = "text"
        else:
            ocr_text = _ocr_via_vision(file_path, provider, model)
            facts = _extract_structured_facts(ocr_text, doc.document_type, provider, model)
            method = "ocr_vision"

        doc.extraction_method = method
        doc.extracted_data = facts
        doc.status = "ready"
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = str(exc)[:2000]
    db.commit()


def _vision_structured_extract_sketch(file_path: Path, report: AppraisalReport | None, provider: str | None, model: str | None) -> dict:
    """Reads a скица/архитектурен чертеж critically and broadly -- room
    list + areas is only ONE part of this, not the whole point (audit
    finding 2026-08-26). Compares against the report's own declared subject
    data when available, so a real mismatch (e.g. drawn area vs
    subject_area_sqm) gets surfaced explicitly rather than silently
    ignored."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        page_images = _render_pdf_pages_as_png(file_path, max_pages=2)
    else:
        page_images = [file_path.read_bytes()]

    chat = get_chat_model(provider, model, max_tokens=1800)
    structured = chat.with_structured_output(SketchFacts)

    declared = "Няма декларирани данни за имота за сравнение."
    if report is not None and (report.subject_property_type or report.subject_area_sqm):
        declared = (
            f"Декларирани данни за имота (за сравнение, не за сляпо потвърждение): "
            f"тип {report.subject_property_type or '—'}, площ {report.subject_area_sqm or '—'} кв.м."
        )

    content = [{"type": "text", "text": (
        "Това е скица/архитектурен чертеж на имот -- може да е разпределение на етаж, "
        "кадастрална скица, фасада или разрез. Прегледай го КРИТИЧНО, не просто изброявай:\n"
        "1. Изброй всяко видимо помещение с площ (ако е изписана или изчислима по мащаба) "
        "и свободен тип (спалня, дневна, кухня, баня, коридор, тераса/балкон, склад, "
        "гараж, стълбище, друго).\n"
        "2. Дай кратка оценка на разпределението -- функционално ли е, добри ли са "
        "пропорциите, естествена светлина, забелязани особености.\n"
        "3. Провери дали изчертаното съответства на декларираните данни по-долу -- ако има "
        "разминаване (площ, брой стаи), посочи го изрично в consistency_notes.\n"
        "4. Отбележи друго забелязано -- асансьор, паркомясто, нередности в самия чертеж.\n\n"
        f"{declared}"
    )}]
    for img_bytes in page_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_image(img_bytes)}"}})

    result = structured.invoke([HumanMessage(content=content)])
    return result.model_dump()


def save_upload(report_id, uploaded_by: int, filename: str, document_type: str, file_bytes: bytes, mime_type: str | None) -> tuple[str, Path]:
    """Writes the uploaded file to disk under a report-scoped subdirectory,
    returns (storage_path relative to documents_dir, absolute path)."""
    ext = Path(filename).suffix.lower() or ""
    stored_name = f"{uuid.uuid4().hex}{ext}"
    rel_dir = Path(str(report_id))
    abs_dir = storage_dir() / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    abs_path = abs_dir / stored_name
    abs_path.write_bytes(file_bytes)
    return str(rel_dir / stored_name), abs_path
