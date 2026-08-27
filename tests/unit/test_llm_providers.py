"""
Unit tests for app/services/llm/providers.py's "local" provider (LM Studio/
Ollama/vLLM, 2026-08-25) -- pure logic + mocked HTTP, no real network calls
and no dependency on a local server actually running. The three existing
providers (openai/anthropic/google_genai) are covered by live manual
verification per the plan doc, not automated tests -- this file focuses on
what "local" adds/changes, since that's genuinely new branching logic.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.services.llm import providers


@pytest.fixture(autouse=True)
def _clear_local_settings(monkeypatch):
    """Every test starts with no local server configured, overridden
    per-test as needed -- prevents a real .env's LOCAL_LLM_BASE_URL (e.g.
    the owner's own LM Studio config) from leaking into test behavior."""
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "")
    monkeypatch.setattr(providers.settings, "local_llm_api_key", "not-needed")


def test_get_chat_model_local_raises_when_base_url_unset():
    with pytest.raises(RuntimeError, match="LOCAL_LLM_BASE_URL"):
        providers.get_chat_model(provider="local", model="qwen2.5")


def test_get_chat_model_local_raises_when_no_model_given(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    with pytest.raises(RuntimeError, match="No local model"):
        providers.get_chat_model(provider="local", model=None)


def test_get_chat_model_local_constructs_chatopenai_with_base_url(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    model = providers.get_chat_model(provider="local", model="qwen2.5-7b-instruct", max_tokens=500)
    # No network call happens on construction -- safe to assert on the
    # constructed client's own config without a running local server.
    assert model.model_name == "qwen2.5-7b-instruct"
    assert str(model.openai_api_base) == "http://localhost:1234/v1"
    assert model.max_tokens == 500


def test_estimate_cost_usd_local_is_always_zero():
    assert providers.estimate_cost_usd("whatever-model", 10_000, 10_000, provider="local") == 0.0
    assert providers.estimate_cost_usd("qwen2.5", 0, 0, provider="local") == 0.0


def test_estimate_cost_usd_unaffected_for_other_providers():
    # Known model: real cost.
    cost = providers.estimate_cost_usd("gpt-5.4-mini", 1_000_000, 1_000_000, provider="openai")
    assert cost == pytest.approx(0.75 + 4.50)
    # Unknown model, no provider given (legacy call shape): still None, not 0.
    assert providers.estimate_cost_usd("some-unlisted-model", 100, 100) is None


def test_list_local_models_empty_when_base_url_unset():
    assert providers.list_local_models() == []


def test_list_local_models_parses_server_response(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    fake_response = MagicMock()
    fake_response.json.return_value = {"data": [{"id": "qwen2.5-7b-instruct"}, {"id": "llama-3.1-8b"}]}
    fake_response.raise_for_status.return_value = None
    with patch("httpx.get", return_value=fake_response) as mock_get:
        result = providers.list_local_models()
    assert result == [("qwen2.5-7b-instruct", "локален"), ("llama-3.1-8b", "локален")]
    called_url = mock_get.call_args.args[0]
    assert called_url == "http://localhost:1234/v1/models"


def test_list_local_models_returns_empty_on_connection_failure(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    with patch("httpx.get", side_effect=ConnectionError("no server")):
        assert providers.list_local_models() == []


def test_get_default_model_local_uses_first_available(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    with patch.object(providers, "list_local_models", return_value=[("qwen2.5", "локален"), ("llama-3.1", "локален")]):
        assert providers.get_default_model("local") == "qwen2.5"


def test_get_default_model_local_empty_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    with patch.object(providers, "list_local_models", return_value=[]):
        assert providers.get_default_model("local") == ""


def test_list_configured_providers_includes_local_only_when_base_url_set(monkeypatch):
    assert "local" not in dict(providers.list_configured_providers())
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    assert dict(providers.list_configured_providers())["local"] == "Локален модел"


def test_resolve_chat_model_local_resolves_via_get_default_model(monkeypatch):
    monkeypatch.setattr(providers.settings, "local_llm_base_url", "http://localhost:1234/v1")
    with patch.object(providers, "list_local_models", return_value=[("qwen2.5", "локален")]):
        provider, model = providers.resolve_chat_model(provider="local")
    assert provider == "local"
    assert model == "qwen2.5"
