"""Chat model provider abstraction (Phase 7).

Hand-rolled factory rather than LangChain's own `init_chat_model`: that
helper lives in the `langchain` umbrella package, which pulls in `langgraph`
as a required (non-optional) dependency just for one dispatch function. A
model built here is still a full `langchain_core` `BaseChatModel` -- every
provider exposes the same `.invoke()`/`.stream()`/`.bind_tools()`/
`.with_structured_output()` surface -- so nothing about the actual
abstraction is lost, only the extra dependency weight.

Adding a new provider (Anthropic/Gemini/local Ollama) is a new branch below
+ its own small `langchain-*` partner package + its own `*_api_key` setting
in app/config.py -- not a redesign.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

_DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    # Haiku 4.5 chosen deliberately over Anthropic's own stated default
    # (Opus 5) to match the cost tier of the other providers' defaults here
    # -- this whole phase is cost-sensitive by design (see plan doc); the
    # owner confirmed this trade-off explicitly (2026-08-24) rather than it
    # being silently decided.
    "anthropic": "claude-haiku-4-5",
    "google_genai": "gemini-3.5-flash-lite",
}

# 3 tiers per provider (cheap/mid/premium) -- cheap always matches
# _DEFAULT_MODELS[provider] above. Lets the UI offer a model choice per
# provider, not just a provider choice. gemini's premium slot deliberately
# uses the "-latest" alias, not a pinned id: the only fully-stable
# (non-preview) Gemini Pro-tier model at the time this was written
# (gemini-2.5-pro) is scheduled for retirement 2026-10-16, and there is no
# non-preview 3.x Pro model yet -- pinning a soon-to-retire id here would be
# worse than tracking Google's own forward-compatible pointer.
_MODEL_TIERS = {
    "openai": [
        ("gpt-5.4-mini", "евтин"),
        ("gpt-5.4", "среден"),
        ("gpt-5.4-pro", "premium"),
    ],
    "anthropic": [
        ("claude-haiku-4-5", "евтин"),
        ("claude-sonnet-5", "среден"),
        ("claude-opus-5", "premium"),
    ],
    "google_genai": [
        ("gemini-3.5-flash-lite", "евтин"),
        ("gemini-3.5-flash", "среден"),
        ("gemini-pro-latest", "premium"),
    ],
}

# USD per 1M tokens (input, output). Confidence varies by source -- treat
# estimated_cost_usd in ai_valuation_runs as an estimate, not a
# billing-accurate figure, regardless of source:
#   - claude-* rates: bundled claude-api skill's own authoritative table.
#   - gemini-3.5-flash-lite: Google's own docs (ai.google.dev), fetched
#     2026-08-24 -- HIGH confidence. (Two earlier aggregator-tracker
#     searches for this same model disagreed with each other -- $0.15/$1.25
#     vs $0.30/$2.50 -- neither matched this official figure; the direct
#     source wins.)
#   - everything else (gpt-5.4/-pro, gemini-3.5-flash): public pricing
#     trackers only, cross-checked across 2+ that roughly agreed -- MEDIUM
#     confidence (OpenAI's/Google's own pricing pages blocked direct fetch
#     in this environment).
# gemini-pro-latest has no fixed rate here (it's an alias that can point to
# a different underlying model over time) -- estimate_cost_usd() returns
# None for it, shown as "-" in the UI rather than a wrong number.
# Update this table if a provider reprices.
_PRICING_PER_1M_USD = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-pro": (15.00, 90.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.00),
}

# Every provider's chat model class names its "max output tokens" kwarg and
# its resolved-model attribute differently -- translated here so callers
# (valuation_chain.py) don't need provider-specific branches of their own.
_MAX_TOKENS_KWARG = {
    "openai": "max_tokens",
    "anthropic": "max_tokens",
    "google_genai": "max_output_tokens",
}


_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Claude",
    "google_genai": "Gemini",
}


def list_configured_providers() -> list[tuple[str, str]]:
    """(provider_key, display_label) pairs for providers that actually have
    an API key set in .env -- lets the UI only offer providers that will
    work, rather than surfacing all three regardless of configuration."""
    configured = []
    if settings.openai_api_key:
        configured.append(("openai", _PROVIDER_LABELS["openai"]))
    if settings.anthropic_api_key:
        configured.append(("anthropic", _PROVIDER_LABELS["anthropic"]))
    if settings.google_api_key:
        configured.append(("google_genai", _PROVIDER_LABELS["google_genai"]))
    return configured


def list_available_models(provider: str) -> list[tuple[str, str]]:
    """(model_id, tier_label) pairs for a provider, cheap tier first --
    used to populate the model-tier dropdown once a provider is chosen."""
    return _MODEL_TIERS.get(provider, [])


def get_default_model(provider: str) -> str:
    """The cheap-tier model id for a provider -- used to pre-select the
    right <option> in the UI's model dropdown."""
    return _DEFAULT_MODELS.get(provider, provider)


def resolve_chat_model(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Resolve (provider, model) against settings/defaults without
    constructing a client. Use this (not attribute-sniffing the constructed
    model afterward -- OpenAI's class exposes `model_name`, Anthropic's and
    Google's expose `model`; there is no attribute name that works for all
    three) whenever the caller needs to know the exact model id, e.g. for
    ai_valuation_runs logging or cost estimation."""
    provider = provider or settings.llm_default_provider
    model = model or _DEFAULT_MODELS.get(provider, provider)
    return provider, model


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Returns None (rather than 0) when the model isn't in the pricing
    table, so callers can distinguish "genuinely free" from "unknown
    price" -- ai_valuation_runs should show "-" for the latter, not "$0.00"."""
    rates = _PRICING_PER_1M_USD.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return round(input_tokens * in_rate / 1_000_000 + output_tokens * out_rate / 1_000_000, 6)


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain chat model for the given provider.

    provider: "openai" | "anthropic" | "google_genai" (default:
        settings.llm_default_provider).
    model: provider-specific model id; defaults to _DEFAULT_MODELS[provider].
    max_tokens: output token cap, translated to whatever kwarg name the
        provider's constructor actually uses (see _MAX_TOKENS_KWARG) --
        pass this instead of max_tokens/max_output_tokens directly in
        kwargs so switching providers doesn't silently drop the cap.
    kwargs: passed through as-is to the provider's chat model constructor
        for anything else provider-specific (e.g. temperature).
    """
    provider, model = resolve_chat_model(provider, model)
    if max_tokens is not None:
        kwargs[_MAX_TOKENS_KWARG[provider]] = max_tokens

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set -- add it to .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, **kwargs)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set -- add it to .env")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=settings.anthropic_api_key, **kwargs)

    if provider == "google_genai":
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set -- add it to .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, **kwargs)

    raise ValueError(f"Unknown LLM provider: {provider!r}")
