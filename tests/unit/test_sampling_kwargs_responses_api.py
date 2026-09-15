"""
Regression tests for two real incidents in app/services/llm/providers.py's
per-model sampling-capability logic (_sampling_support/build_sampling_kwargs/
get_sampling_capabilities):

1. (2026-09-11) A user picked gpt-5.4-pro in the /assistant/ or /analyst/
   chat model picker with presence_penalty set and got a raw
   "Responses.create() got an unexpected keyword argument 'presence_penalty'"
   -- langchain-openai's ChatOpenAI silently routes model ids matching a
   known prefix set (gpt-5-pro, gpt-5.2-pro, gpt-5.4-pro, gpt-5.5-pro)
   through OpenAI's Responses API instead of Chat Completions, and the
   Responses API's create() has no frequency_penalty/presence_penalty/seed
   parameters at all -- confirmed via
   inspect.signature(openai.resources.responses.Responses.create).
   gpt-5.4-pro itself was later retired from _MODEL_TIERS (superseded by the
   gpt-5.6 family, 2026-09-15), but _sampling_support's override logic is
   still exercised directly here so a historical ai_valuation_runs row (or
   any future re-introduction of a "-pro" id) stays covered.

2. (2026-09-15) Verifying the gpt-5.6 Luna/Terra/Sol upgrade with real
   invoke() calls surfaced a DIFFERENT restriction: these models stay on
   ordinary Chat Completions (unlike gpt-5.4-pro), but reject top_p/
   frequency_penalty/presence_penalty with a clean 400 "Unsupported
   parameter" (temperature and seed both work). Same underlying gap as (1)
   -- _SAMPLING_SUPPORT was per-provider only -- closed by the same
   per-model override mechanism, extended to also gate top_p (previously
   assumed universally supported).

The same live-verification pass also found google_genai's
frequency_penalty/presence_penalty marked as supported when EVERY tested
Gemini model actually rejects both ("Penalty is not enabled for this
model") -- a real, pre-existing bug unrelated to either incident above,
fixed in the same commit; covered by test_google_genai_penalties_are_off
below.
"""
from app.services.llm import providers


def test_build_sampling_kwargs_strips_unsupported_params_for_responses_api_model():
    kwargs = providers.build_sampling_kwargs(
        "openai", model="gpt-5.4-pro",
        temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, seed=42,
    )
    assert "presence_penalty" not in kwargs
    assert "frequency_penalty" not in kwargs
    assert "seed" not in kwargs
    # temperature/top_p ARE supported by the Responses API -- must stay.
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


def test_build_sampling_kwargs_strips_unsupported_params_for_gpt56_family():
    for model in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        kwargs = providers.build_sampling_kwargs(
            "openai", model=model,
            temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, seed=42,
        )
        assert "top_p" not in kwargs, model
        assert "presence_penalty" not in kwargs, model
        assert "frequency_penalty" not in kwargs, model
        # temperature and seed ARE supported (verified live) -- must stay.
        assert kwargs["temperature"] == 0.7, model
        assert kwargs["seed"] == 42, model


def test_build_sampling_kwargs_keeps_penalties_for_unrelated_openai_models():
    """A model matching none of the known override prefixes falls through
    to the plain provider default -- everything stays enabled."""
    kwargs = providers.build_sampling_kwargs(
        "openai", model="gpt-4o-mini",
        presence_penalty=0.5, frequency_penalty=0.5, top_p=0.9, seed=42,
    )
    assert kwargs["presence_penalty"] == 0.5
    assert kwargs["frequency_penalty"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["seed"] == 42


def test_build_sampling_kwargs_no_model_falls_back_to_provider_default():
    """Existing callers that never pass model (e.g. before this fix landed
    elsewhere) must keep working exactly as before -- model=None is the
    default, and with no model info there's nothing to narrow."""
    kwargs = providers.build_sampling_kwargs("openai", presence_penalty=0.5)
    assert kwargs["presence_penalty"] == 0.5


def test_get_sampling_capabilities_has_narrowed_entry_for_gpt56_sol():
    caps = providers.get_sampling_capabilities()
    assert "openai:gpt-5.6-sol" in caps
    sol_caps = caps["openai:gpt-5.6-sol"]
    assert sol_caps["top_p"]["supported"] is False
    assert sol_caps["presence_penalty"]["supported"] is False
    assert sol_caps["frequency_penalty"]["supported"] is False
    assert sol_caps["seed"]["supported"] is True   # NOT disabled for this family, unlike gpt-5.4-pro


def test_get_sampling_capabilities_does_not_narrow_unrelated_openai_models():
    caps = providers.get_sampling_capabilities()
    assert "openai:gpt-4o-mini" not in caps  # no override -- matches the provider default
    assert caps["openai"]["presence_penalty"]["supported"] is True
    assert caps["openai"]["top_p"]["supported"] is True


def test_google_genai_penalties_are_off():
    """Real, pre-existing bug (2026-09-15, found while verifying the
    gemini-3.8-flash upgrade): every tested Gemini model -- old and new --
    rejects frequency_penalty/presence_penalty with INVALID_ARGUMENT
    ("Penalty is not enabled for this model"), but _SAMPLING_SUPPORT
    marked both as supported for the whole provider. seed genuinely works
    and must stay enabled."""
    caps = providers.get_sampling_capabilities()
    assert caps["google_genai"]["frequency_penalty"]["supported"] is False
    assert caps["google_genai"]["presence_penalty"]["supported"] is False
    assert caps["google_genai"]["seed"]["supported"] is True

    kwargs = providers.build_sampling_kwargs(
        "google_genai", model="gemini-3.8-flash",
        frequency_penalty=0.5, presence_penalty=0.5, seed=42,
    )
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs
    assert kwargs["seed"] == 42


def test_responses_api_prefix_list_matches_real_openai_sdk_signature():
    """Cheap live-package sanity check (no network call): confirms the
    hardcoded mirror of langchain-openai's private prefix list still
    reflects reality -- that the Responses API genuinely lacks these three
    params -- so this regression can't silently drift back out of sync with
    the installed SDK version."""
    import inspect
    from openai.resources.responses.responses import Responses

    params = set(inspect.signature(Responses.create).parameters)
    assert "presence_penalty" not in params
    assert "frequency_penalty" not in params
    assert "seed" not in params
    assert "temperature" in params
    assert "top_p" in params
