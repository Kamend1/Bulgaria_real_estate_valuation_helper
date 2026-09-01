"""
Regex-based chunker for Bulgarian legal/regulatory documents (Phase 14
Tier 3.1) -- splits on "Чл. N" (article) and "Раздел N" (section) markers,
the two heading conventions actually observed in the uploaded
legal_standard documents. Deliberately NOT LangChain's
RecursiveCharacterTextSplitter -- there is no existing text-splitter
precedent anywhere in this codebase, and a small hand-written regex
splitter matches the established "explicit code over a new abstraction"
style already used throughout (retriever.py, comparable_service.py).

Pure function, no I/O -- app/services/market_documents.py owns embedding
and persisting the chunks this produces.
"""
from __future__ import annotations

import re

_MARKER_RE = re.compile(r"(?m)^(Чл\.\s*\d+[а-я]?\.?|Раздел\s+[IVXLCDM]+\.?|Раздел\s+\d+\.?)\s*")

# A single article/section longer than this gets further split into
# overlapping fixed-size windows -- keeps every stored/embedded chunk small
# enough that one search_legal_document result stays cheap to read, even
# for an unusually long article.
MAX_CHUNK_CHARS = 3000
_WINDOW_OVERLAP = 200


def _split_long_chunk(heading: str | None, body: str) -> list[tuple[str | None, str]]:
    if len(body) <= MAX_CHUNK_CHARS:
        return [(heading, body)]
    parts: list[tuple[str | None, str]] = []
    start = 0
    part_num = 1
    while start < len(body):
        end = min(start + MAX_CHUNK_CHARS, len(body))
        part_heading = f"{heading} (част {part_num})" if heading else None
        parts.append((part_heading, body[start:end]))
        if end >= len(body):
            break
        start = end - _WINDOW_OVERLAP
        part_num += 1
    return parts


def split_legal_text(full_text: str) -> list[dict]:
    """Returns [{"heading": str | None, "text": str}, ...] in document
    order. `text` always includes its own heading prefix (when there is
    one) so the chunk stays self-identifying once separated from its
    neighbors -- a search result showing just "...в срок от 30 дни" with
    no article number attached would be useless to cite.

    Falls back to plain fixed-size windows (heading=None throughout) when
    no "Чл./Раздел" markers are found at all -- some legal_standard uploads
    (a short circular, a translated excerpt) don't follow that convention,
    and a document with zero chunks would be unsearchable."""
    text = full_text.strip()
    if not text:
        return []

    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return [
            {"heading": None, "text": text[i:i + MAX_CHUNK_CHARS]}
            for i in range(0, len(text), MAX_CHUNK_CHARS - _WINDOW_OVERLAP)
        ]

    chunks: list[dict] = []
    preamble = text[:matches[0].start()].strip()
    if len(preamble) > 50:
        chunks.append({"heading": None, "text": preamble})

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        for part_heading, part_body in _split_long_chunk(heading, body):
            full = f"{part_heading} {part_body}" if part_heading else part_body
            chunks.append({"heading": part_heading, "text": full})
    return chunks
