"""
Unit tests for build_orchestrator_graph()'s per-domain model_overrides
plumbing (Phase 15 Tier 1, 2026-09-15) -- the foundation for cross-model
verification (Tier 2) and per-specialist model choice. Every specialist
node already resolves its own chat model independently inside
_build_tool_loop's closure; base_kwargs just decides which (provider,
model) pair each node's closure gets built with.

Heavily mocked: build_specialist_tools_and_prompt and get_report_memory are
replaced so this test exercises ONLY the override-selection logic in
base_kwargs, not tool construction or DB reads (both irrelevant to what's
being verified here, and both need a real DB/report to run for real).
"""
from unittest.mock import MagicMock, patch

from app.services.llm import orchestrator_graph as og


def _build_graph(model_overrides=None):
    db = MagicMock()
    report = MagicMock(id="report-1")
    with (
        patch.object(og, "build_specialist_tools_and_prompt", return_value=([], "prompt", "step")),
        patch.object(og, "get_report_memory", return_value={}),
        patch.object(og, "_build_tool_loop") as mock_loop,
    ):
        og.build_orchestrator_graph(
            db=db, report=report, documents_note="",
            provider="anthropic", model_id="claude-haiku-4-5",
            sampling_kwargs={"temperature": 0.5}, max_tokens=2000, max_tool_iterations=6,
            call_log=[], msg_seq=1, on_progress=lambda *a: None, on_message=lambda *a: None,
            model_overrides=model_overrides,
        )
        return {call.kwargs["call_label_prefix"].split("_", 1)[1]: call.kwargs for call in mock_loop.call_args_list}


def test_no_overrides_every_specialist_gets_the_same_model():
    calls = _build_graph(model_overrides=None)
    assert set(calls) == {"income", "market", "market_analysis", "legal", "zoning", "comp_quality"}
    for domain, kwargs in calls.items():
        assert (kwargs["provider"], kwargs["model_id"]) == ("anthropic", "claude-haiku-4-5"), domain
        assert kwargs["sampling_kwargs"] == {"temperature": 0.5}, domain


def test_override_for_one_domain_only_affects_that_domain():
    calls = _build_graph(model_overrides={"income": ("openai", "gpt-5.6-terra")})
    assert (calls["income"]["provider"], calls["income"]["model_id"]) == ("openai", "gpt-5.6-terra")
    # Overridden node gets no sampling overrides (safe default), not the
    # default-model sampling_kwargs, which may not be valid for a different
    # provider/model.
    assert calls["income"]["sampling_kwargs"] == {}
    for domain in ("market", "market_analysis", "legal", "zoning", "comp_quality"):
        assert (calls[domain]["provider"], calls[domain]["model_id"]) == ("anthropic", "claude-haiku-4-5")
        assert calls[domain]["sampling_kwargs"] == {"temperature": 0.5}


def test_override_matching_the_default_is_not_treated_as_overridden():
    """An override entry that happens to name the SAME (provider, model) as
    the turn's default must still get the real sampling_kwargs -- it isn't
    actually a different model, just an explicit no-op entry."""
    calls = _build_graph(model_overrides={"legal": ("anthropic", "claude-haiku-4-5")})
    assert calls["legal"]["sampling_kwargs"] == {"temperature": 0.5}
