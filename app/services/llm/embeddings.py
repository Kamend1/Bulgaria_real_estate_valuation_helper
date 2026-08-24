"""Embeddings provider abstraction (Phase 7) -- see providers.py for why
this is a hand-rolled factory rather than a LangChain built-in dispatcher.

IMPORTANT: retriever.py compares vectors by cosine distance, which is only
meaningful between vectors produced by the SAME provider+model. Changing
llm_default_embedding_provider/model after listings have already been
embedded means either re-embedding the whole corpus (scripts/embed_listings.py)
or keeping the old provider/model around for retrieval until it's done --
listing_embeddings' unique constraint is (listing_id, provider, model), so
both can coexist during a migration.
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
    use this instead of reaching into _DEFAULT_MODELS directly."""
    provider = provider or settings.llm_default_embedding_provider
    model = model or _DEFAULT_MODELS.get(provider, provider)
    return provider, model


def get_embeddings_model(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return a LangChain Embeddings instance for the given provider.

    provider: "openai" (default: settings.llm_default_embedding_provider).
    model: provider-specific embedding model id; defaults to
        _DEFAULT_MODELS[provider].
    """
    provider, model = resolve_embedding_model(provider, model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set -- add it to .env")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    raise ValueError(f"Unknown embeddings provider: {provider!r}")
