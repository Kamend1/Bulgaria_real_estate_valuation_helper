"""
Regression test for a real incident (2026-09-11): a user picked gpt-5.4-pro
in the /assistant/ or /analyst/ chat model picker with presence_penalty set
and got a raw "Responses.create() got an unexpected keyword argument
'presence_penalty'" -- langchain-openai's ChatOpenAI silently routes model
ids matching a known prefix set (gpt-5-pro, gpt-5.2-pro, gpt-5.4-pro,
gpt-5.5-pro) through OpenAI's Responses API instead of Chat Completions, and
the Responses API's create() has no frequency_penalty/presence_penalty/seed
parameters at all -- confirmed directly via
inspect.signature(openai.resources.responses.Responses.create) against the
openai/langchain-openai versions installed in this environment.

_SAMPLING_SUPPORT in app/services/llm/providers.py was per-PROVIDER only,
so it happily forwarded these params for every OpenAI model regardless of
which API it actually uses. Fixed with _sampling_support(provider, model),
which narrows the three unsupported params for the affected model ids;
build_sampling_kwargs and get_sampling_capabilities were both updated to
take the model into account.
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


def test_build_sampling_kwargs_keeps_penalties_for_other_openai_models():
    kwargs = providers.build_sampling_kwargs(
        "openai", model="gpt-5.4-mini",
        presence_penalty=0.5, frequency_penalty=0.5, seed=42,
    )
    assert kwargs["presence_penalty"] == 0.5
    assert kwargs["frequency_penalty"] == 0.5
    assert kwargs["seed"] == 42


def test_build_sampling_kwargs_no_model_falls_back_to_provider_default():
    """Existing callers that never pass model (e.g. before this fix landed
    elsewhere) must keep working exactly as before -- model=None is the
    default, and with no model info there's nothing to narrow."""
    kwargs = providers.build_sampling_kwargs("openai", presence_penalty=0.5)
    assert kwargs["presence_penalty"] == 0.5


def test_get_sampling_capabilities_has_narrowed_entry_for_pro_model():
    caps = providers.get_sampling_capabilities()
    assert "openai:gpt-5.4-pro" in caps
    pro_caps = caps["openai:gpt-5.4-pro"]
    assert pro_caps["presence_penalty"]["supported"] is False
    assert pro_caps["frequency_penalty"]["supported"] is False
    assert pro_caps["seed"]["supported"] is False


def test_get_sampling_capabilities_does_not_narrow_other_openai_models():
    caps = providers.get_sampling_capabilities()
    assert "openai:gpt-5.4-mini" not in caps  # no override needed -- matches the provider default
    assert caps["openai"]["presence_penalty"]["supported"] is True


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
