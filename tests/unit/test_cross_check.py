"""
Unit tests for Phase 15 Tier 2's cross-model verification (2026-09-15):
critic_graph.run_cross_check() and orchestrator_graph._pick_different_provider/
_maybe_cross_check_proposal. Heavily mocked -- no real LLM calls, no real DB;
these tests are about the wiring (does a disagreement produce a warning
without ever blocking the proposal, does the picked verifier provider
actually differ from the one that made the claim), not model quality.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.llm import critic_graph, orchestrator_graph as og


def _fake_chat(parsed, usage=None):
    chat = MagicMock()
    structured = MagicMock()
    raw = MagicMock()
    raw.usage_metadata = usage or {"input_tokens": 50, "output_tokens": 10}
    structured.invoke.return_value = {"raw": raw, "parsed": parsed}
    chat.with_structured_output.return_value = structured
    return chat


class TestRunCrossCheck:
    def test_agreement_path(self, monkeypatch):
        parsed = critic_graph.CrossCheckResult(agrees=True, note="")
        monkeypatch.setattr(critic_graph, "get_chat_model", lambda *a, **k: _fake_chat(parsed))
        monkeypatch.setattr(
            critic_graph, "_gather_context_node",
            lambda db, report: (lambda state: {"context": {"sale_pool_stats": {"median_ppsqm": 1200}}}),
        )
        report = SimpleNamespace(id="r1")
        verdict, call_log = critic_graph.run_cross_check(MagicMock(), report, "openai", "gpt-5.6-luna", "claim text")
        assert verdict["agrees"] is True
        assert call_log[0]["call_label"] == "crosscheck"
        assert call_log[0]["input_tokens"] == 50

    def test_disagreement_path(self, monkeypatch):
        parsed = critic_graph.CrossCheckResult(agrees=False, note="Медианата на пула е 1200, твърдението цитира 1800.")
        monkeypatch.setattr(critic_graph, "get_chat_model", lambda *a, **k: _fake_chat(parsed))
        monkeypatch.setattr(
            critic_graph, "_gather_context_node",
            lambda db, report: (lambda state: {"context": {}}),
        )
        report = SimpleNamespace(id="r1")
        verdict, _ = critic_graph.run_cross_check(MagicMock(), report, "anthropic", "claude-haiku-4-5", "claim text")
        assert verdict["agrees"] is False
        assert "1800" in verdict["note"]

    def test_unparseable_response_defaults_to_agrees(self, monkeypatch):
        monkeypatch.setattr(critic_graph, "get_chat_model", lambda *a, **k: _fake_chat(None))
        monkeypatch.setattr(
            critic_graph, "_gather_context_node",
            lambda db, report: (lambda state: {"context": {}}),
        )
        report = SimpleNamespace(id="r1")
        verdict, _ = critic_graph.run_cross_check(MagicMock(), report, "openai", "gpt-5.6-luna", "claim")
        assert verdict["agrees"] is True   # never fabricate a disagreement on a parse failure


class TestPickDifferentProvider:
    def test_returns_a_different_configured_provider(self, monkeypatch):
        monkeypatch.setattr(og, "list_configured_providers", lambda: [("openai", "OpenAI"), ("anthropic", "Claude")])
        assert og._pick_different_provider("openai") == "anthropic"
        assert og._pick_different_provider("anthropic") == "openai"

    def test_falls_back_to_current_when_only_one_configured(self, monkeypatch):
        monkeypatch.setattr(og, "list_configured_providers", lambda: [("openai", "OpenAI")])
        assert og._pick_different_provider("openai") == "openai"

    def test_prefers_rotation_order_among_several_choices(self, monkeypatch):
        monkeypatch.setattr(og, "list_configured_providers", lambda: [
            ("openai", "OpenAI"), ("anthropic", "Claude"), ("google_genai", "Gemini"),
        ])
        # current="google_genai" -> anthropic comes before openai in _PROVIDER_ROTATION
        assert og._pick_different_provider("google_genai") == "anthropic"


class TestMaybeCrossCheckProposal:
    def _tool_result(self, field="submarket_rationale", text="Ново твърдение."):
        return {"proposed": True, "field": field, "field_label": "x", "text": text}

    def test_low_risk_field_skips_cross_check_entirely(self):
        db = MagicMock()
        warning = og._maybe_cross_check_proposal(
            db, "report-1", "openai", self._tool_result(field="subject_description"), [], "msg1_market",
        )
        assert warning is None
        db.get.assert_not_called()   # never even fetches the report -- zero extra cost for low-risk fields

    def test_agreement_returns_no_warning(self, monkeypatch):
        db = MagicMock()
        db.get.return_value = SimpleNamespace(id="report-1")
        monkeypatch.setattr(og, "resolve_chat_model", lambda provider, model: (provider, "some-model"))
        monkeypatch.setattr(critic_graph, "run_cross_check", lambda *a, **k: ({"agrees": True, "note": ""}, []))
        call_log = []
        warning = og._maybe_cross_check_proposal(db, "report-1", "openai", self._tool_result(), call_log, "msg1_market")
        assert warning is None
        assert call_log == []  # no sub_call_log entries to append when the mock returns none... but a real run would still log

    def test_disagreement_returns_visible_warning_not_a_block(self, monkeypatch):
        db = MagicMock()
        db.get.return_value = SimpleNamespace(id="report-1")
        monkeypatch.setattr(og, "resolve_chat_model", lambda provider, model: (provider, "some-model"))
        monkeypatch.setattr(
            critic_graph, "run_cross_check",
            lambda *a, **k: ({"agrees": False, "note": "Числата не съвпадат."}, [{"call_label": "crosscheck", "input_tokens": 10, "output_tokens": 5}]),
        )
        call_log = []
        warning = og._maybe_cross_check_proposal(db, "report-1", "openai", self._tool_result(), call_log, "msg1_market")
        assert warning is not None
        assert "Числата не съвпадат." in warning
        assert call_log[0]["call_label"] == "msg1_market_crosscheck"   # re-labeled under the specialist's own prefix

    def test_verifier_call_failure_is_swallowed(self, monkeypatch):
        db = MagicMock()
        db.get.return_value = SimpleNamespace(id="report-1")
        monkeypatch.setattr(og, "resolve_chat_model", lambda provider, model: (provider, "some-model"))
        monkeypatch.setattr(critic_graph, "run_cross_check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        warning = og._maybe_cross_check_proposal(db, "report-1", "openai", self._tool_result(), [], "msg1_market")
        assert warning is None   # a broken cross-check must never surface as an error to the appraiser
