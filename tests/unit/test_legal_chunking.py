"""
Unit tests for app/services/llm/legal_chunking.py (Phase 14 Tier 3.1) --
pure string logic, no DB/embedding calls.
"""
from app.services.llm.legal_chunking import MAX_CHUNK_CHARS, split_legal_text


def test_empty_text_returns_no_chunks():
    assert split_legal_text("") == []
    assert split_legal_text("   ") == []


def test_splits_on_article_markers():
    text = (
        "Чл. 1. Този закон урежда...\n"
        "продължение на текста на чл. 1.\n"
        "Чл. 2. Вторият член казва нещо друго.\n"
        "Чл. 3. Третият член казва трето нещо."
    )
    chunks = split_legal_text(text)
    assert len(chunks) == 3
    assert chunks[0]["heading"] == "Чл. 1."
    assert "Чл. 1." in chunks[0]["text"]
    assert "продължение" in chunks[0]["text"]
    assert chunks[1]["heading"] == "Чл. 2."
    assert chunks[2]["heading"] == "Чл. 3."


def test_splits_on_section_markers():
    text = "Раздел I\nОбщи положения текст.\nРаздел II\nСпециални правила текст."
    chunks = split_legal_text(text)
    assert len(chunks) == 2
    assert chunks[0]["heading"] == "Раздел I"
    assert chunks[1]["heading"] == "Раздел II"


def test_preamble_before_first_marker_becomes_its_own_heading_less_chunk():
    preamble = "З А К О Н за уреждане на нещо, приет с Указ № 5 от Народното събрание." * 2
    text = f"{preamble}\nЧл. 1. Първи член."
    chunks = split_legal_text(text)
    assert chunks[0]["heading"] is None
    assert chunks[0]["text"].strip() == preamble.strip()
    assert chunks[-1]["heading"] == "Чл. 1."


def test_short_preamble_is_dropped_not_kept_as_noise_chunk():
    text = "Увод\nЧл. 1. Първи член от документа."
    chunks = split_legal_text(text)
    assert len(chunks) == 1
    assert chunks[0]["heading"] == "Чл. 1."


def test_falls_back_to_fixed_windows_when_no_markers_found():
    text = "Обикновен текст без членове или раздели. " * 200
    chunks = split_legal_text(text)
    assert len(chunks) > 1
    assert all(c["heading"] is None for c in chunks)
    assert all(len(c["text"]) <= MAX_CHUNK_CHARS for c in chunks)


def test_long_article_is_split_into_overlapping_windows_with_part_suffix():
    body = "Дълъг текст на член. " * 300  # comfortably over MAX_CHUNK_CHARS
    text = f"Чл. 5. {body}"
    chunks = split_legal_text(text)
    assert len(chunks) > 1
    assert chunks[0]["heading"] == "Чл. 5. (част 1)"
    assert chunks[1]["heading"] == "Чл. 5. (част 2)"
    assert all(len(c["text"]) <= MAX_CHUNK_CHARS + len("Чл. 5. (част 99) ") for c in chunks)


def test_every_chunk_text_contains_its_own_heading_when_present():
    text = "Чл. 7. Текст на седмия член."
    chunks = split_legal_text(text)
    assert chunks[0]["text"].startswith("Чл. 7.")
