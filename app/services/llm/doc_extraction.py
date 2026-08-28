"""Shared low-level document extraction mechanics (Phase 10, 2026-08-28) --
native text extraction (PyMuPDF/python-docx) and vision-LLM OCR fallback.

Extracted out of app/services/documents.py (which used to own these
privately) so app/services/market_documents.py can reuse the exact same
PDF/DOCX/OCR handling for the market analyst's reference library, instead
of duplicating it. Neither caller's document_type/schema logic lives here
-- this module only turns a file on disk into plain text.
"""
from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.services.llm.providers import get_chat_model

MIN_NATIVE_TEXT_CHARS = 50   # below this, treat a PDF as "no text layer" (scanned)
MAX_OCR_PAGES = 8


def b64_image(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def render_pdf_pages_as_png(file_path: Path, max_pages: int = MAX_OCR_PAGES) -> list[bytes]:
    import fitz   # PyMuPDF
    images = []
    doc = fitz.open(file_path)
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=150)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def extract_native_text(file_path: Path, mime_type: str | None) -> str | None:
    """Text already embedded in the file (not scanned) -- None signals
    "try vision OCR instead", not an error."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        doc = fitz.open(file_path)
        try:
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
        return text if len(text.strip()) >= MIN_NATIVE_TEXT_CHARS else None
    if suffix == ".docx":
        import docx
        d = docx.Document(file_path)
        text = "\n".join(p.text for p in d.paragraphs)
        return text if len(text.strip()) >= MIN_NATIVE_TEXT_CHARS else None
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return text if len(text.strip()) >= MIN_NATIVE_TEXT_CHARS else None
    return None   # images (.jpg/.png/...) have no "native text" path


def ocr_via_vision(file_path: Path, provider: str | None, model: str | None) -> str:
    """Vision-LLM transcription -- used when a PDF has no text layer (a
    scanned document) or the upload is a photo/scan image directly."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        page_images = render_pdf_pages_as_png(file_path)
    else:
        page_images = [file_path.read_bytes()]

    chat = get_chat_model(provider, model, max_tokens=4000)
    content = [{"type": "text", "text": (
        "Транскрибирай ЦЕЛИЯ видим текст от следните страници на документ, дословно, "
        "на български. Без коментар, само транскрипцията, страница по страница."
    )}]
    for img_bytes in page_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(img_bytes)}"}})

    response = chat.invoke([HumanMessage(content=content)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return text
