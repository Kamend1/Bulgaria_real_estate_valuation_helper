"""
Tests for the 2026-09-24 model upgrade: OpenAI's gpt-6 Luna/Sol/Astra and
Anthropic's claude-opus-5-5 as the middle tier. Every capability assertion
here mirrors something verified with a real API call that day (not assumed
from docs), see the comments in app/services/llm/providers.py. No network
calls -- ChatOpenAI construction doesn't hit the network.
"""
from types import SimpleNamespace

import pytest

from app.services.llm import providers


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    monkeypatch.setattr(providers.settings, "openai_api_key", "sk-test-not-real")


# ── tiers, defaults, pricing ────────────────────────────────────────────────

def test_openai_tiers_are_gpt6_luna_sol_astra_in_price_order():
    assert providers.list_available_models("openai") == [
        ("gpt-6-luna", "евтин"), ("gpt-6-sol", "среден"), ("gpt-6-astra", "premium"),
    ]
    assert providers.get_default_model("openai") == "gpt-6-luna"


def test_anthropic_middle_tier_is_opus_5_5():
    assert providers.list_available_models("anthropic") == [
        ("claude-haiku-4-5", "евтин"), ("claude-opus-5-5", "среден"), ("claude-opus-5", "premium"),
    ]
    assert providers.get_default_model("anthropic") == "claude-haiku-4-5"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_tier_input_prices_are_strictly_increasing(provider):
    """cheap < mid < premium by real price -- same guard the Mistral tiers
    got after Large turned out cheaper than Medium."""
    prices = [providers._PRICING_PER_1M_USD[m][0] for m, _ in providers.list_available_models(provider)]
    assert prices == sorted(prices) and len(set(prices)) == 3


def test_pricing_matches_vendor_pricing_pages_checked_2026_09_24():
    cost = lambda m, p: providers.estimate_cost_usd(m, 1_000_000, 1_000_000, provider=p)
    assert cost("gpt-6-luna", "openai") == pytest.approx(0.10 + 0.50)
    assert cost("gpt-6-sol", "openai") == pytest.approx(2.00 + 10.00)
    assert cost("gpt-6-astra", "openai") == pytest.approx(10.00 + 50.00)
    assert cost("claude-opus-5-5", "anthropic") == pytest.approx(4.00 + 20.00)
    # Stale until 2026-09-24: Sonnet 5's intro price ($2/$10) became permanent,
    # the announced rise to $3/$15 never happened.
    assert cost("claude-sonnet-5", "anthropic") == pytest.approx(2.00 + 10.00)


# ── OpenAI sampling profiles ────────────────────────────────────────────────

_ALL = dict(temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, seed=42)


def test_gpt6_astra_rejects_every_sampling_param():
    assert providers.build_sampling_kwargs("openai", model="gpt-6-astra", **_ALL) == {}


def test_gpt6_luna_keeps_temperature_top_p_seed_but_not_penalties():
    """Penalties returned reproducible HTTP 500s on gpt-6-luna; top_p works
    (unlike gpt-5.6, which rejected it)."""
    assert providers.build_sampling_kwargs("openai", model="gpt-6-luna", **_ALL) == {
        "temperature": 0.7, "top_p": 0.9, "seed": 42,
    }


def test_gpt6_sol_keeps_everything():
    kwargs = providers.build_sampling_kwargs("openai", model="gpt-6-sol", **_ALL)
    assert kwargs == {
        "temperature": 0.7, "top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.5, "seed": 42,
    }


def test_capabilities_expose_temperature_flag_for_gpt6_models():
    caps = providers.get_sampling_capabilities()
    assert caps["openai:gpt-6-astra"]["temperature"]["supported"] is False
    assert "openai:gpt-6-sol" not in caps  # identical to the provider default -> no narrowed entry
    assert caps["openai"]["temperature"]["supported"] is True


# ── OpenAI client construction ──────────────────────────────────────────────

@pytest.mark.parametrize("model_id", ["gpt-6-luna", "gpt-6-sol"])
def test_gpt6_luna_and_sol_force_reasoning_effort_none_on_chat_completions(model_id):
    """Both reject tool-bound Chat Completions requests unless effort is
    'none' (verified live) -- and every real call here binds tools."""
    model = providers.get_chat_model(provider="openai", model=model_id, max_tokens=50)
    assert model.reasoning_effort == "none"
    assert not model.use_responses_api


def test_gpt6_astra_is_routed_through_the_responses_api_without_forced_effort():
    """Astra rejects reasoning_effort='none' AND rejects tools on Chat
    Completions with any other effort -- Responses API is the only route."""
    model = providers.get_chat_model(provider="openai", model="gpt-6-astra", max_tokens=50)
    assert model.use_responses_api is True
    assert model.reasoning_effort is None


def test_explicit_use_responses_api_choice_is_respected():
    model = providers.get_chat_model(
        provider="openai", model="gpt-6-astra", max_tokens=50, use_responses_api=False,
    )
    assert model.use_responses_api is False


# ── Anthropic sampling profiles ─────────────────────────────────────────────

@pytest.mark.parametrize("model_id", [
    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5", "claude-fable-5-1",
])
def test_claude_5_generation_models_reject_all_sampling_params(model_id):
    """Verified live on all four: temperature (except exactly 1.0), top_p and
    top_k all return 400 "`<param>` is deprecated for this model"."""
    kwargs = providers.build_sampling_kwargs(
        "anthropic", model=model_id, temperature=0.5, top_p=0.9, top_k=40,
    )
    assert kwargs == {}


def test_claude_haiku_4_5_keeps_its_existing_profile():
    kwargs = providers.build_sampling_kwargs(
        "anthropic", model="claude-haiku-4-5", temperature=0.5, top_p=0.9, top_k=40,
    )
    # temperature wins over top_p (Anthropic rejects both together), top_k works.
    assert kwargs == {"temperature": 0.5, "top_k": 40}


def test_haiku_top_p_alone_still_passes_through():
    kwargs = providers.build_sampling_kwargs("anthropic", model="claude-haiku-4-5", top_p=0.9)
    assert kwargs == {"top_p": 0.9}


def test_capabilities_disable_temperature_for_opus_5_5_but_not_haiku():
    caps = providers.get_sampling_capabilities()
    opus = caps["anthropic:claude-opus-5-5"]
    assert opus["temperature"]["supported"] is False
    assert opus["top_p"]["supported"] is False
    assert opus["top_k"]["supported"] is False
    assert caps["anthropic"]["temperature"]["supported"] is True   # provider default (Haiku)
    assert "anthropic:claude-haiku-4-5" not in caps


# ── structured output ───────────────────────────────────────────────────────

class _FakeChat:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return "runnable"


def test_structured_output_switches_opus_5_5_to_json_schema():
    """claude-opus-5-5 rejects forced tool_choice (400 'type "tool" and
    "any" are not supported for this model'), which is what langchain-
    anthropic's default function_calling method uses -- it broke the
    orchestrator supervisor, critic, cross-check and document extraction."""
    chat = _FakeChat("claude-opus-5-5")
    assert providers.structured_output(chat, dict, include_raw=True) == "runnable"
    assert chat.calls == [(dict, {"include_raw": True, "method": "json_schema"})]


@pytest.mark.parametrize("model_id", ["claude-haiku-4-5", "claude-opus-5", "gpt-6-sol", "gemini-3.8-flash"])
def test_structured_output_leaves_other_models_on_their_default_method(model_id):
    chat = _FakeChat(model_id)
    providers.structured_output(chat, dict, include_raw=True)
    assert chat.calls == [(dict, {"include_raw": True})]


def test_structured_output_explicit_method_wins():
    chat = _FakeChat("claude-opus-5-5")
    providers.structured_output(chat, dict, method="function_calling")
    assert chat.calls == [(dict, {"method": "function_calling"})]


def test_structured_output_tolerates_mock_chats_with_no_string_model():
    """Existing tests hand back a MagicMock chat -- its .model isn't a str."""
    from unittest.mock import MagicMock
    chat = MagicMock()
    providers.structured_output(chat, dict, include_raw=True)
    chat.with_structured_output.assert_called_once_with(dict, include_raw=True)


def test_real_anthropic_client_exposes_the_model_attribute_the_helper_reads(monkeypatch):
    """Cheap guard against langchain-anthropic renaming the attribute --
    if this ever fails, structured_output() silently stops switching Opus
    5.5 to json_schema and the 400 comes back."""
    monkeypatch.setattr(providers.settings, "anthropic_api_key", "sk-ant-test-not-real")
    chat = providers.get_chat_model(provider="anthropic", model="claude-opus-5-5", max_tokens=50)
    assert chat.model == "claude-opus-5-5"


# ── truncation detection ────────────────────────────────────────────────────

def _resp(**meta):
    return SimpleNamespace(response_metadata=meta)


def test_is_length_truncated_detects_responses_api_incomplete_status():
    """Real shape from a max_tokens=30 gpt-6-astra call: no finish_reason at
    all; all 30 tokens were hidden reasoning, so no visible text either."""
    assert providers.is_length_truncated(
        _resp(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    ) is True


def test_is_length_truncated_ignores_completed_and_other_incomplete_reasons():
    assert providers.is_length_truncated(_resp(status="completed")) is False
    assert providers.is_length_truncated(
        _resp(status="incomplete", incomplete_details={"reason": "content_filter"})
    ) is False
    assert providers.is_length_truncated(_resp(status="incomplete")) is False


@pytest.mark.parametrize("meta", [
    {"finish_reason": "length"}, {"stop_reason": "max_tokens"}, {"finish_reason": "MAX_TOKENS"},
])
def test_is_length_truncated_still_detects_chat_completions_anthropic_google_signals(meta):
    assert providers.is_length_truncated(_resp(**meta)) is True
