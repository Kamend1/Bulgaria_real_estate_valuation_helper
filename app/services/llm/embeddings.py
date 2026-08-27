"""Embeddings provider abstraction (Phase 7) -- see providers.py for why
this is a hand-rolled factory rather than a LangChain built-in dispatcher.

IMPORTANT: retriever.py compares vectors by cosine distance, which is only
meaningful between vectors produced by the SAME provider+model. Changing
llm_default_embedding_provider/model after listings have already been
embedded means either re-embedding the whole corpus (scripts/embed_listings.py)
or keeping the old provider/model around for retrieval until it's done --
listing_embeddings' unique constraint is (listing_id, provider, model), so
both can coexist during a migration. This applies doubly to "local"
(2026-08-25): the existing ~177K-listing corpus is OpenAI-embedded, and a
local embedding model produces vectors from a DIFFERENT vector space --
adding "local" here does not touch that corpus, it only makes a local
model usable as a NEW, separate (listing_id, provider, model) track for
whoever explicitly asks for it. Re-embedding the whole corpus locally is a
distinct, much larger decision, not implied by this being available.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config import settings

_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
}


def resolve_embedding_model(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Resolve (provider, model) against settings/defaults without
    constructing a client -- callers that need to know the model id up
    front (e.g. scripts/embed_listings.py, to query already-embedded rows)
    use this instead of reaching into _DEFAULT_MODELS directly. For
    "local" there is no fixed default (see providers.get_default_model) --
    an explicit model id is required."""
    provider = provider or settings.llm_default_embedding_provider
    if provider == "local":
        return provider, model or ""
    model = model or _DEFAULT_MODELS.get(provider, provider)
    return provider, model


def get_embeddings_model(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return a LangChain Embeddings instance for the given provider.

    provider: "openai" | "local" (default: settings.llm_default_embedding_provider).
    model: provider-specific embedding model id; defaults to
        _DEFAULT_MODELS[provider]. Required (no default) for "local" -- use
        providers.list_local_models() to see what's actually loaded; not
        every locally-loaded model is embedding-capable, the server doesn't
        distinguish the two in its /models listing.
    """
    provider, model = resolve_embedding_model(provider, model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set -- add it to .env")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    if provider == "local":
        if not settings.local_llm_base_url:
            raise RuntimeError("LOCAL_LLM_BASE_URL is not set -- add it to .env (e.g. http://localhost:1234/v1)")
        if not model:
            raise RuntimeError("No local embedding model specified -- pick one from providers.list_local_models()")
        # Same reasoning as providers.get_chat_model()'s "local" branch --
        # LM Studio/Ollama/vLLM all implement the OpenAI embeddings
        # endpoint too, so no new SDK dependency is needed here either.
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=settings.local_llm_api_key, base_url=settings.local_llm_base_url)

    raise ValueError(f"Unknown embeddings provider: {provider!r}")
