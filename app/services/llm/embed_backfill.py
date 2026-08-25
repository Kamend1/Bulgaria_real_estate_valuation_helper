"""Shared embedding backfill logic (Phase 7), used by both the standalone
scripts/embed_listings.py CLI and the automatic post-scrape hook in
scrape_service.py.

Finds listings that need a fresh embedding for a given (provider, model):
  - never embedded at all, OR
  - embedded before, but listing_to_text() no longer matches the text that
    was actually embedded (the listing's data changed on a re-scrape --
    upserting listings.* does NOT touch listing_embeddings, so without this
    check a re-scraped listing would silently keep serving a stale vector
    for the rest of its life).

When `run_id` is given, the candidate set is narrowed to listings touched
by that specific scrape run (last_scrape_run_id == run_id) -- this is what
the automatic post-scrape hook uses, so a routine re-scrape only re-embeds
the handful of listings it actually touched, not the whole corpus. The CLI
script omits `run_id` for a full-corpus catch-up pass.
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import Listing, ListingEmbedding
from app.services.llm.embeddings import get_embeddings_model, resolve_embedding_model
from app.services.llm.listing_doc import listing_to_text

BATCH_SIZE = 100


def find_embedding_candidates(
    db: Session,
    provider: str,
    model: str,
    run_id=None,
    limit: int | None = None,
) -> list[Listing]:
    """Listings that need a new/updated embedding for (provider, model)."""
    filters = [Listing.training_eligible.is_(True), Listing.status == "active"]
    if run_id is not None:
        filters.append(Listing.last_scrape_run_id == run_id)

    query = select(Listing).where(*filters).order_by(Listing.id)
    if limit:
        query = query.limit(limit)
    rows = db.execute(query).scalars().all()
    if not rows:
        return []

    existing = dict(
        db.execute(
            select(ListingEmbedding.listing_id, ListingEmbedding.embedded_text).where(
                ListingEmbedding.provider == provider,
                ListingEmbedding.model == model,
                ListingEmbedding.listing_id.in_([r.id for r in rows]),
            )
        ).all()
    )

    out = []
    for row in rows:
        current_text = listing_to_text(row)
        prior_text = existing.get(row.id)
        if prior_text is None or prior_text != current_text:
            out.append(row)
    return out


def backfill_embeddings(
    db: Session,
    provider: str | None = None,
    model: str | None = None,
    run_id=None,
    limit: int | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every candidate found by find_embedding_candidates(). Commits
    per batch. Returns the number of listings embedded/re-embedded.

    on_batch(done, total), if given, is called after each committed batch --
    used for CLI progress printing and for the scrape-run SSE log."""
    provider, model = resolve_embedding_model(provider, model)
    candidates = find_embedding_candidates(db, provider, model, run_id=run_id, limit=limit)
    if not candidates:
        return 0
    total = len(candidates)

    embeddings_model = get_embeddings_model(provider=provider, model=model)

    done = 0
    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start:start + BATCH_SIZE]
        texts = [listing_to_text(r) for r in batch]
        vectors = embeddings_model.embed_documents(texts)

        stmt = pg_insert(ListingEmbedding).values([
            {
                "listing_id": r.id,
                "provider": provider,
                "model": model,
                "embedding": vec,
                "embedded_text": text,
            }
            for r, text, vec in zip(batch, texts, vectors)
        ])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_listing_embeddings_listing_provider_model",
            set_={"embedding": stmt.excluded.embedding, "embedded_text": stmt.excluded.embedded_text},
        )
        db.execute(stmt)
        db.commit()
        done += len(batch)
        if on_batch:
            on_batch(done, total)

    return done
