"""
Backfill semantic-text embeddings for listings into `listing_embeddings`
(Phase 7 — app/services/llm/retriever.py reads these for AI-assisted
comparable retrieval).

Ports the standalone-script pattern already used by
scripts/train_avm_model.py: read live DB data, no web app involved, safe to
re-run (crash-resumable — already-embedded rows for the current
provider/model are skipped, matching listing_embeddings' unique constraint
on (listing_id, provider, model)).

Usage (from project root):
    python -m scripts.embed_listings                  # embed everything still missing
    python -m scripts.embed_listings --limit 50        # small test run
    python -m scripts.embed_listings --dry-run         # build + print text only, no API calls, no DB writes

Requires OPENAI_API_KEY in .env (real cost — each call is billed by OpenAI;
see app/services/llm/embeddings.py for the model used).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to a legacy codepage (e.g. cp1251) that can't
# encode every character that shows up in scraped Bulgarian listing text
# (smart quotes, em dashes, etc.) -- reconfigure stdout so --dry-run's
# printouts don't crash the whole run over a display-only limitation.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Listing, ListingEmbedding
from app.db.session import db_session
from app.services.llm.embeddings import get_embeddings_model, resolve_embedding_model
from app.services.llm.listing_doc import listing_to_text

BATCH_SIZE = 100


def _listings_needing_embedding(db, provider: str, model: str, limit: int | None):
    already = (
        select(ListingEmbedding.listing_id)
        .where(ListingEmbedding.provider == provider, ListingEmbedding.model == model)
    )
    query = (
        select(Listing)
        .where(
            Listing.training_eligible.is_(True),
            Listing.status == "active",
            Listing.id.notin_(already),
        )
        .order_by(Listing.id)
    )
    if limit:
        query = query.limit(limit)
    return db.execute(query).scalars().all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=None, help="Embeddings provider (default: settings.llm_default_embedding_provider)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of listings embedded (for a small test run)")
    parser.add_argument("--dry-run", action="store_true", help="Build embedded_text and print counts only -- no API calls, no DB writes")
    args = parser.parse_args()

    provider, model = resolve_embedding_model(args.provider)

    with db_session() as db:
        rows = _listings_needing_embedding(db, provider, model, args.limit)
        print(f"{len(rows)} listing(s) need embedding (provider={provider}, model={model})")
        if not rows:
            return

        if args.dry_run:
            for row in rows[:3]:
                print(f"--- id {row.id} ---")
                print(listing_to_text(row))
            print(f"(dry run — {len(rows)} total, showing up to 3)")
            return

        embeddings_model = get_embeddings_model(provider=provider, model=model)

        done = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
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
            print(f"  embedded {done}/{len(rows)}")

        print(f"Done — {done} listing(s) embedded with {provider}/{model}.")


if __name__ == "__main__":
    main()
