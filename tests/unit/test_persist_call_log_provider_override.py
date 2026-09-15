"""
Regression test for a real bug found during Phase 15 Tier 2's live
verification (2026-09-15): assistant_chain._persist_call_log persisted
EVERY call_log entry under the turn's own provider/model, even when an
entry carried its own "provider"/"model" keys (as
critic_graph.run_cross_check's and run_critical_review's entries do,
re-appended by orchestrator_graph.py's _maybe_cross_check_proposal/
_auditor_node_fn) -- silently mislabeling a cross-model verifier's actual
provider/cost as whichever model produced the claim being checked. Mocked
DB (inspecting the constructed AgentLlmCall objects directly) rather than a
real one -- AgentConversation/User FK setup isn't needed to test this pure
per-entry-override-vs-fallback logic.
"""
from unittest.mock import MagicMock

from app.services.llm.assistant_chain import _persist_call_log


def test_entry_without_provider_falls_back_to_turn_default():
    db = MagicMock()
    _persist_call_log(db, "conv-1", "anthropic", "claude-haiku-4-5", [
        {"call_label": "msg1_market_step1", "input_tokens": 100, "output_tokens": 20},
    ])
    added = db.add.call_args_list[0].args[0]
    assert added.provider == "anthropic"
    assert added.model == "claude-haiku-4-5"


def test_entry_with_its_own_provider_overrides_the_turn_default():
    db = MagicMock()
    _persist_call_log(db, "conv-1", "openai", "gpt-5.6-luna", [
        {"call_label": "msg1_market_crosscheck", "input_tokens": 50, "output_tokens": 10,
         "provider": "anthropic", "model": "claude-haiku-4-5"},
    ])
    added = db.add.call_args_list[0].args[0]
    assert added.provider == "anthropic"
    assert added.model == "claude-haiku-4-5"


def test_mixed_entries_each_keep_their_own_attribution():
    db = MagicMock()
    _persist_call_log(db, "conv-1", "openai", "gpt-5.6-luna", [
        {"call_label": "msg1_market_step1", "input_tokens": 100, "output_tokens": 20},
        {"call_label": "msg1_market_crosscheck", "input_tokens": 50, "output_tokens": 10,
         "provider": "anthropic", "model": "claude-haiku-4-5"},
    ])
    step1, crosscheck = (c.args[0] for c in db.add.call_args_list)
    assert (step1.provider, step1.model) == ("openai", "gpt-5.6-luna")
    assert (crosscheck.provider, crosscheck.model) == ("anthropic", "claude-haiku-4-5")
