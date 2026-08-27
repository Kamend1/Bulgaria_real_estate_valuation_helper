"""Chat model provider abstraction (Phase 7).

Hand-rolled factory rather than LangChain's own `init_chat_model`: that
helper lives in the `langchain` umbrella package, which pulls in `langgraph`
as a required (non-optional) dependency just for one dispatch function. A
model built here is still a full `langchain_core` `BaseChatModel` -- every
provider exposes the same `.invoke()`/`.stream()`/`.bind_tools()`/
`.with_structured_output()` surface -- so nothing about the actual
abstraction is lost, only the extra dependency weight.

Adding a new provider is a new branch below + (usually) its own small
`langchain-*` partner package + its own `*_api_key` setting in
app/config.py -- not a redesign. "local" (2026-08-25, LM Studio/Ollama/vLLM)
is the exception that needs no new package at all: those all speak the
OpenAI Chat Completions API, so its branch just points the existing
ChatOpenAI/OpenAIEmbeddings classes at a configurable base_url.
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
    # "local" deliberately absent here -- there is no fixed catalog to
    # hardcode (depends entirely on what the user has loaded in LM Studio/
    # Ollama/vLLM right now). list_available_models() special-cases "local"
    # to query the server live instead of reading this dict.
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
    "local": "max_tokens",   # OpenAI-compatible servers use the same name
}


_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Claude",
    "google_genai": "Gemini",
    "local": "Локален модел",
}


def list_configured_providers() -> list[tuple[str, str]]:
    """(provider_key, display_label) pairs for providers that actually have
    an API key (or, for "local", a base_url) set in .env -- lets the UI
    only offer providers that will work, rather than surfacing all of them
    regardless of configuration. This checks CONFIGURATION, not live
    reachability -- a configured-but-currently-off local server fails at
    generation time with a clear error, the same way a bad API key would,
    rather than this function making a network call on every page render."""
    configured = []
    if settings.openai_api_key:
        configured.append(("openai", _PROVIDER_LABELS["openai"]))
    if settings.anthropic_api_key:
        configured.append(("anthropic", _PROVIDER_LABELS["anthropic"]))
    if settings.google_api_key:
        configured.append(("google_genai", _PROVIDER_LABELS["google_genai"]))
    if settings.local_llm_base_url:
        configured.append(("local", _PROVIDER_LABELS["local"]))
    return configured


def list_local_models(timeout: float = 1.5) -> list[tuple[str, str]]:
    """(model_id, "локален") pairs actually loaded on the configured local
    server right now, via the OpenAI-compatible GET /models endpoint that
    LM Studio/Ollama/vLLM all implement. Unlike _MODEL_TIERS there is no
    fixed catalog to hardcode -- it depends entirely on what the user has
    loaded locally. Returns [] (never raises) if the base_url is unset or
    the server isn't reachable -- the UI shows "no local models found"
    rather than a 500, matching the graceful-degradation pattern used
    elsewhere (e.g. avm_service's model_fetch_failed reason)."""
    if not settings.local_llm_base_url:
        return []
    import httpx
    try:
        resp = httpx.get(
            f"{settings.local_llm_base_url.rstrip('/')}/models",
            timeout=timeout,
            headers={"Authorization": f"Bearer {settings.local_llm_api_key}"},
        )
        resp.raise_for_status()
        return [(m["id"], "локален") for m in resp.json().get("data", [])]
    except Exception:
        return []


def list_available_models(provider: str) -> list[tuple[str, str]]:
    """(model_id, tier_label) pairs for a provider, cheap tier first --
    used to populate the model-tier dropdown once a provider is chosen.
    "local" is queried live (see list_local_models) instead of read from
    the static _MODEL_TIERS table the other providers use."""
    if provider == "local":
        return list_local_models()
    return _MODEL_TIERS.get(provider, [])


def get_default_model(provider: str) -> str:
    """The cheap-tier model id for a provider -- used to pre-select the
    right <option> in the UI's model dropdown. For "local", the first
    currently-loaded model (there is no fixed default), or "" if none."""
    if provider == "local":
        local_models = list_local_models()
        return local_models[0][0] if local_models else ""
    return _DEFAULT_MODELS.get(provider, provider)


def resolve_chat_model(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Resolve (provider, model) against settings/defaults without
    constructing a client. Use this (not attribute-sniffing the constructed
    model afterward -- OpenAI's class exposes `model_name`, Anthropic's and
    Google's expose `model`; there is no attribute name that works for all
    three) whenever the caller needs to know the exact model id, e.g. for
    ai_valuation_runs logging or cost estimation."""
    provider = provider or settings.llm_default_provider
    model = model or get_default_model(provider)
    return provider, model


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, provider: str | None = None) -> float | None:
    """Returns None (rather than 0) when the model isn't in the pricing
    table, so callers can distinguish "genuinely free" from "unknown
    price" -- ai_valuation_runs should show "-" for the latter, not "$0.00".
    Local models are the one genuine 0.0 case (no fixed catalog to price by
    model id, but the whole *provider* is free by construction) -- pass
    provider so this can special-case it instead of needing every possible
    local model name pre-populated in _PRICING_PER_1M_USD."""
    if provider == "local":
        return 0.0
    rates = _PRICING_PER_1M_USD.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return round(input_tokens * in_rate / 1_000_000 + output_tokens * out_rate / 1_000_000, 6)


# Per-provider (min, max) for temperature -- Anthropic's API rejects >1.0,
# OpenAI/Google/local (OpenAI-compatible) accept up to 2.0. Enforced in
# build_sampling_kwargs() rather than left to the provider to reject, so a
# stale UI value (e.g. left over from switching provider mid-edit) clamps
# instead of raising at generation time.
_TEMPERATURE_RANGE = {
    "openai": (0.0, 2.0),
    "anthropic": (0.0, 1.0),
    "google_genai": (0.0, 2.0),
    "local": (0.0, 2.0),
}

# Which sampling kwargs each provider's LangChain class actually declares
# as a constructor field -- verified directly against ChatOpenAI/
# ChatAnthropic/ChatGoogleGenerativeAI's own `model_fields` (2026-08-26),
# not assumed from API docs: a kwarg the class doesn't declare raises at
# construction time rather than being silently ignored. "local" reuses the
# ChatOpenAI class itself (see get_chat_model below), so it shares
# "openai"'s exact field set.
_SAMPLING_SUPPORT = {
    "openai":       {"top_k": False, "frequency_penalty": True,  "presence_penalty": True,  "seed": True},
    "anthropic":    {"top_k": True,  "frequency_penalty": False, "presence_penalty": False, "seed": False},
    "google_genai": {"top_k": True,  "frequency_penalty": True,  "presence_penalty": True,  "seed": True},
    "local":        {"top_k": False, "frequency_penalty": True,  "presence_penalty": True,  "seed": True},
}


def get_sampling_capabilities() -> dict:
    """Per-provider ranges/support for the chat console's model-parameter
    panel -- single source of truth shared by the template (which controls
    to grey out per provider) and build_sampling_kwargs (which values to
    actually forward to the provider)."""
    caps = {}
    for provider, (temp_min, temp_max) in _TEMPERATURE_RANGE.items():
        support = _SAMPLING_SUPPORT[provider]
        caps[provider] = {
            "temperature": {"min": temp_min, "max": temp_max},
            "top_p": {"min": 0.0, "max": 1.0},
            "top_k": {"supported": support["top_k"], "min": 1, "max": 500},
            "frequency_penalty": {"supported": support["frequency_penalty"], "min": -2.0, "max": 2.0},
            "presence_penalty": {"supported": support["presence_penalty"], "min": -2.0, "max": 2.0},
            "seed": {"supported": support["seed"]},
        }
    return caps


def build_sampling_kwargs(
    provider: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
) -> dict:
    """Filters + clamps user-supplied sampling params to what the resolved
    provider's class actually accepts (see _SAMPLING_SUPPORT), clamping
    in-range values instead of forwarding out-of-range ones. The UI already
    greys out unsupported controls per provider, but this is the actual
    enforcement point -- a stale form value from switching provider/model
    mid-edit must not reach the provider constructor and raise.

    Anthropic special case (found via live testing, 2026-08-26): the
    Messages API rejects a request that sets BOTH temperature and top_p at
    once ("`temperature` and `top_p` cannot both be specified for this
    model") -- unlike OpenAI/Google, which happily accept both together.
    Since the UI's two sliders both always carry a concrete value (there is
    no "unset" slider position), sending both unconditionally would break
    every Anthropic turn, not just a deliberately-combined edge case.
    temperature wins when both are given -- it is the more commonly
    understood knob and the UI's own top_p hint already tells the user to
    change one or the other, not both."""
    provider = provider or "openai"
    support = _SAMPLING_SUPPORT.get(provider, _SAMPLING_SUPPORT["openai"])
    temp_min, temp_max = _TEMPERATURE_RANGE.get(provider, (0.0, 2.0))
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = max(temp_min, min(temp_max, temperature))
    if top_p is not None and not (provider == "anthropic" and temperature is not None):
        kwargs["top_p"] = max(0.0, min(1.0, top_p))
    if top_k is not None and support["top_k"]:
        kwargs["top_k"] = max(1, int(top_k))
    if frequency_penalty is not None and support["frequency_penalty"]:
        kwargs["frequency_penalty"] = max(-2.0, min(2.0, frequency_penalty))
    if presence_penalty is not None and support["presence_penalty"]:
        kwargs["presence_penalty"] = max(-2.0, min(2.0, presence_penalty))
    if seed is not None and support["seed"]:
        kwargs["seed"] = int(seed)
    return kwargs


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

    if provider == "local":
        if not settings.local_llm_base_url:
            raise RuntimeError("LOCAL_LLM_BASE_URL is not set -- add it to .env (e.g. http://localhost:1234/v1)")
        if not model:
            raise RuntimeError(
                "No local model specified and none is currently loaded -- "
                "start LM Studio/Ollama, load a model, and select it in the UI"
            )
        # No new SDK dependency: LM Studio, Ollama, and vLLM all speak the
        # OpenAI Chat Completions API, so the existing ChatOpenAI class
        # works unmodified against a local base_url -- see app/config.py's
        # local_llm_base_url docstring for the reasoning.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=settings.local_llm_api_key, base_url=settings.local_llm_base_url, **kwargs)

    raise ValueError(f"Unknown LLM provider: {provider!r}")
