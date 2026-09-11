"""
Regression test for a real incident (2026-09-11): a claude-opus-5 run of
generate_valuation_backbone() hit the output-token cap EXACTLY (9000 of
9000, confirmed via agent_llm_calls), cutting the income REASONING section
off mid-sentence and never writing the income CAVEATS section at all --
but nothing in valuation_chain.py checked for this, so the truncated
narrative was persisted and shown to the appraiser as if it were complete.
Fixed by wiring in providers.is_length_truncated() (the same helper
Phase 12 already uses for the /assistant/ and /analyst/ chat loops) on the
final narrative-generating call, in both places that call can come from
(the tool loop's own final turn, and the tools-unbound fallback stream).

Heavily mocked (retrieve_comparables/get_pool_with_stats/build_tools/
get_chat_model) rather than hitting the real DB or a real LLM -- this test
is only about the truncation-detection wiring, not retrieval or generation
quality.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from app.services.llm import valuation_chain as vc

_NARRATIVE = (
    "## ОПИСАНИЕ НА ИМОТА\nтест\n"
    "## ОБОСНОВКА НА ИЗБОРА НА СРАВНИМИ\nтест\n"
    "## КОМЕНТАР ПО СРАВНИМИТЕ\nтест\n"
    "## РАЗСЪЖДЕНИЕ\nтекстът свършва насред изр"
)

_FAKE_COMP = {
    "id": 1, "ad_url": "https://example.com/1", "price_per_sqm_model": 1000.0,
    "area_sqm_model": 60.0, "distance": 0.1,
}


def _fake_report():
    return SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        subject_property_type="apartment", subject_area_sqm=60,
        subject_city="София", subject_geo_category="sofia_center",
        subject_neighborhood=None, subject_construction=None,
        subject_year=None, subject_floor=None, subject_total_floors=None,
        subject_description=None,
    )


def _patch_common(monkeypatch, chat_factory):
    monkeypatch.setattr(vc, "retrieve_comparables", lambda db, report, k, comparable_type: [_FAKE_COMP] * 4)
    monkeypatch.setattr(vc, "get_pool_with_stats", lambda db, ctype, report_id: {"stats": None})
    monkeypatch.setattr(vc, "build_tools", lambda db, report: [])
    monkeypatch.setattr(vc, "get_chat_model", chat_factory)
    monkeypatch.setattr(vc, "estimate_cost_usd", lambda *a, **k: 0.01)


def _chat_returning(content: str, response_metadata: dict, usage: dict):
    def fake_get_chat_model(provider, model_id, max_tokens=None, **kwargs):
        chat = MagicMock()

        def stream(messages):
            chunk = AIMessageChunk(content=content)
            chunk.usage_metadata = usage
            chunk.response_metadata = response_metadata
            yield chunk

        chat.stream = stream
        chat.bind_tools = lambda tools: chat  # no tools -- same object works for both
        return chat
    return fake_get_chat_model


def test_truncated_final_call_is_flagged(monkeypatch):
    _patch_common(monkeypatch, _chat_returning(
        _NARRATIVE, {"stop_reason": "max_tokens"}, {"input_tokens": 5000, "output_tokens": 9000},
    ))
    db = MagicMock()
    progress = vc.generate_valuation_backbone(db, _fake_report(), provider="anthropic", model="claude-opus-5")
    assert progress.status == "done"
    assert progress.result["truncated"] is True


def test_complete_final_call_is_not_flagged(monkeypatch):
    _patch_common(monkeypatch, _chat_returning(
        _NARRATIVE, {"stop_reason": "end_turn"}, {"input_tokens": 5000, "output_tokens": 1200},
    ))
    db = MagicMock()
    progress = vc.generate_valuation_backbone(db, _fake_report(), provider="anthropic", model="claude-opus-5")
    assert progress.status == "done"
    assert progress.result["truncated"] is False


def test_truncated_fallback_stream_is_flagged(monkeypatch):
    """When the first attempt never produces the expected section headers,
    generate_valuation_backbone() discards it and does a second, tools-
    unbound streamed call -- truncation must be checked on THAT call's
    response, not (only) the discarded first one."""
    call_count = {"n": 0}

    def fake_get_chat_model(provider, model_id, max_tokens=None, **kwargs):
        chat = MagicMock()

        def stream(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: no section headers at all -> triggers fallback.
                chunk = AIMessageChunk(content="случаен несвързан текст без заглавия")
                chunk.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
                chunk.response_metadata = {"stop_reason": "end_turn"}
                yield chunk
            else:
                chunk = AIMessageChunk(content=_NARRATIVE)
                chunk.usage_metadata = {"input_tokens": 5000, "output_tokens": 9000}
                chunk.response_metadata = {"stop_reason": "max_tokens"}
                yield chunk

        chat.stream = stream
        chat.bind_tools = lambda tools: chat
        return chat

    _patch_common(monkeypatch, fake_get_chat_model)
    db = MagicMock()
    progress = vc.generate_valuation_backbone(db, _fake_report(), provider="anthropic", model="claude-opus-5")
    assert progress.status == "done"
    assert progress.result["truncated"] is True
