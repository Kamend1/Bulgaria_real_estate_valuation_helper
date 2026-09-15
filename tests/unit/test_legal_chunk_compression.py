"""
Unit tests for Phase 15 Tier 3.3's legal-lead delegation (2026-09-15):
orchestrator_graph._compress_legal_chunks -- a cheap model compresses large
retrieved legal chunks before the (potentially pricier) legal specialist's
own model reads them. Mocked chat model, no real LLM calls.
"""
from unittest.mock import MagicMock

from app.services.llm import orchestrator_graph as og


def _small_sections():
    return [{"heading": "Чл. 1", "text": "кратък текст"}]


def _large_sections(total_chars=8000):
    return [{"heading": "Чл. 1", "text": "х" * total_chars}]


def test_small_sections_pass_through_unchanged():
    sections = _small_sections()
    result = og._compress_legal_chunks(sections, "някакъв въпрос")
    assert result == sections


def test_large_sections_get_compressed(monkeypatch):
    chat = MagicMock()
    response = MagicMock()
    response.content = "сгъстен релевантен текст"
    chat.invoke.return_value = response
    monkeypatch.setattr(og, "get_chat_model", lambda *a, **k: chat)

    sections = _large_sections()
    result = og._compress_legal_chunks(sections, "конкретен въпрос")
    assert len(result) == 1
    assert result[0]["text"] == "сгъстен релевантен текст"
    assert "сгъстено" in result[0]["heading"]
    chat.invoke.assert_called_once()


def test_compression_failure_falls_back_to_original_sections(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(og, "get_chat_model", _boom)

    sections = _large_sections()
    result = og._compress_legal_chunks(sections, "въпрос")
    assert result == sections   # never lose the retrieval result on failure


def test_empty_compressed_text_falls_back_to_original_sections(monkeypatch):
    chat = MagicMock()
    response = MagicMock()
    response.content = ""
    chat.invoke.return_value = response
    monkeypatch.setattr(og, "get_chat_model", lambda *a, **k: chat)

    sections = _large_sections()
    result = og._compress_legal_chunks(sections, "въпрос")
    assert result == sections
