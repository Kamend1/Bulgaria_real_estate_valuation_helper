"""
Unit tests for the Mistral AI provider (2026-09-15), added specifically for
its EU-only data-residency endpoint. No real network calls -- ChatMistralAI
construction doesn't hit the network, so these assert on the constructed
client's own config, mirroring test_llm_providers.py's "local" provider
pattern.
"""
from unittest.mock import MagicMock

import pytest

from app.services.llm import providers


@pytest.fixture(autouse=True)
def _fake_mistral_key(monkeypatch):
    monkeypatch.setattr(providers.settings, "mistral_api_key", "fake-key-not-real")


def test_get_chat_model_mistral_raises_when_key_unset(monkeypatch):
    monkeypatch.setattr(providers.settings, "mistral_api_key", "")
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        providers.get_chat_model(provider="mistral", model="mistral-small-2603")


def test_get_chat_model_mistral_routes_through_eu_endpoint():
    model = providers.get_chat_model(provider="mistral", model="mistral-small-2603", max_tokens=500)
    assert str(model.endpoint) == providers._MISTRAL_EU_ENDPOINT
    assert providers._MISTRAL_EU_ENDPOINT == "https://api.eu.mistral.ai/v1"


def test_list_configured_providers_includes_mistral_when_key_set():
    assert dict(providers.list_configured_providers())["mistral"] == "Mistral (ЕС)"


def test_list_configured_providers_excludes_mistral_when_key_unset(monkeypatch):
    monkeypatch.setattr(providers.settings, "mistral_api_key", "")
    assert "mistral" not in dict(providers.list_configured_providers())


def test_get_default_model_mistral_is_the_cheap_tier():
    assert providers.get_default_model("mistral") == "mistral-small-2603"


def test_model_tiers_mistral_premium_is_medium_not_large():
    """Real, verified pricing quirk (Mistral's own pricing page): Large 3
    ($0.50/$1.50) is cheaper than Medium 3.5 ($1.50/$7.50) -- the premium
    slot must be "medium", not "large", despite the name suggesting
    otherwise."""
    tiers = dict((mid, label) for mid, label in providers.list_available_models("mistral"))
    assert tiers["mistral-medium-2604"] == "premium"
    assert tiers["mistral-large-2512"] == "среден"
    assert tiers["mistral-small-2603"] == "евтин"
    cheap, mid, premium = (m for m, _ in providers.list_available_models("mistral"))
    cheap_price, mid_price, premium_price = (
        providers._PRICING_PER_1M_USD[cheap][0],
        providers._PRICING_PER_1M_USD[mid][0],
        providers._PRICING_PER_1M_USD[premium][0],
    )
    assert cheap_price < mid_price < premium_price


def test_sampling_support_mistral_no_top_k_no_seed():
    caps = providers.get_sampling_capabilities()["mistral"]
    assert caps["top_k"]["supported"] is False
    assert caps["seed"]["supported"] is False
    assert caps["frequency_penalty"]["supported"] is True
    assert caps["presence_penalty"]["supported"] is True


def test_temperature_range_mistral_caps_at_one():
    assert providers._TEMPERATURE_RANGE["mistral"] == (0.0, 1.0)


def test_build_sampling_kwargs_mistral_drops_top_k_and_seed():
    kwargs = providers.build_sampling_kwargs(
        "mistral", model="mistral-small-2603",
        temperature=0.5, top_p=0.9, top_k=40, frequency_penalty=0.2, presence_penalty=0.2, seed=42,
    )
    assert "top_k" not in kwargs
    assert "seed" not in kwargs
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["frequency_penalty"] == 0.2
    assert kwargs["presence_penalty"] == 0.2


def test_estimate_cost_usd_all_three_mistral_tiers_priced():
    for model_id in ("mistral-small-2603", "mistral-large-2512", "mistral-medium-2604"):
        cost = providers.estimate_cost_usd(model_id, 1_000_000, 1_000_000, provider="mistral")
        assert cost is not None
        assert cost > 0
