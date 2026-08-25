"""
Backfill semantic-text embeddings for listings into `listing_embeddings`
(Phase 7 — app/services/llm/retriever.py reads these for AI-assisted
comparable retrieval).

Ports the standalone-script pattern already used by
scripts/train_avm_model.py: read live DB data, no web app involved, safe to
re-run (crash-resumable -- candidate selection lives in
app/services/llm/embed_backfill.py, shared with the automatic post-scrape
hook in scrape_service.py, and picks up both never-embedded listings AND
previously-embedded listings whose text changed since -- see that module's
docstring).

Usage (from project root):
    python -m scripts.embed_listings                  # full-corpus catch-up: embed everything missing or changed
    python -m scripts.embed_listings --limit 50        # small test run
    python -m scripts.embed_listings --dry-run         # build + print text only, no API calls, no DB writes

Note: routine re-scrapes no longer need this run manually -- scrape_service.py
calls the same backfill automatically (scoped to the just-completed run) at
the end of every scrape. This script remains for the initial full-corpus
backfill and for catching up after enabling embeddings for the first time,
switching provider/model, or if the automatic hook ever fails.

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

from app.db.session import db_session
from app.services.llm.embed_backfill import backfill_embeddings, find_embedding_candidates
from app.services.llm.embeddings import resolve_embedding_model
from app.services.llm.listing_doc import listing_to_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=None, help="Embeddings provider (default: settings.llm_default_embedding_provider)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of listings embedded (for a small test run)")
    parser.add_argument("--dry-run", action="store_true", help="Build embedded_text and print counts only -- no API calls, no DB writes")
    args = parser.parse_args()

    provider, model = resolve_embedding_model(args.provider)

    with db_session() as db:
        if args.dry_run:
            rows = find_embedding_candidates(db, provider, model, limit=args.limit)
            print(f"{len(rows)} listing(s) need embedding (provider={provider}, model={model})")
            for row in rows[:3]:
                print(f"--- id {row.id} ---")
                print(listing_to_text(row))
            print(f"(dry run — {len(rows)} total, showing up to 3)")
            return

        def _progress(done: int, total: int) -> None:
            print(f"  embedded {done}/{total}")

        done = backfill_embeddings(db, provider=provider, model=model, limit=args.limit, on_batch=_progress)
        print(f"Done — {done} listing(s) embedded with {provider}/{model}.")


if __name__ == "__main__":
    main()
