"""
One-time import of existing parsed sales parquet files into PostgreSQL.

Usage (from project root):
    python -m scripts.import_historical_data

What it does:
  - Scans data/parsed_sales_runs/*/parsed_listings.parquet
  - Creates ONE synthetic scrape_run row per directory (status='imported')
  - Applies engineer_features() to every row
  - Upserts into listings (ON CONFLICT ad_url DO UPDATE)
  - Inserts listing_snapshots (one per listing per run)
  - Each row is wrapped in a SAVEPOINT — bad data is skipped, never kills the batch
  - Prints progress every 5,000 rows
  - Idempotent: safe to re-run (upsert + insert snapshot)
"""

import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import sqlalchemy
from sqlalchemy import exc as sa_exc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.db.models import Listing, ListingSnapshot, ScrapeRun
from app.db.session import SessionLocal
from app.services.scrape_service import _build_listing_values, sync_taxonomy_to_db
from utils.feature_engineering import engineer_features

CHUNK_SIZE = 500
PROGRESS_EVERY = 5_000

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def _parse_dir_timestamp(dirname: str) -> datetime:
    m = _TS_RE.search(dirname)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _find_parquet_dirs(base_dir: str) -> list[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "parsed_listings.parquet").exists()]
    )


def _upsert_chunk(session, chunk: list[dict], db_run_id: uuid.UUID, run_ts: datetime) -> tuple[int, int]:
    """
    Upsert a chunk of rows using per-row SAVEPOINTs.
    Returns (upserted, skipped).

    Two failure modes are handled separately:
    - Data errors (overflow, constraint): SAVEPOINT rolls back that row only; rest of chunk continues.
    - Connection errors (server dropped connection): session.rollback() resets the session;
      remaining rows in the chunk are skipped; next chunk will use a fresh connection from the pool.
    """
    upserted = 0
    skipped = 0
    connection_lost = False

    for raw_row in chunk:
        if connection_lost:
            # Don't attempt DB operations on a poisoned session.
            skipped += 1
            continue

        try:
            row = engineer_features(raw_row)
        except Exception:
            skipped += 1
            continue

        if not row.get("ad_url"):
            skipped += 1
            continue

        try:
            listing_vals = _build_listing_values(row, db_run_id)
        except Exception:
            skipped += 1
            continue

        # Each row gets its own SAVEPOINT so a data-level DB error (overflow, constraint, etc.)
        # only rolls back that single row and leaves the outer transaction intact.
        try:
            with session.begin_nested():
                stmt = (
                    pg_insert(Listing)
                    .values(**listing_vals)
                    .on_conflict_do_update(
                        index_elements=["ad_url"],
                        set_={
                            "last_seen_at": run_ts,
                            "last_scrape_run_id": listing_vals["last_scrape_run_id"],
                            "total_price": listing_vals["total_price"],
                            "price_per_sqm_model": listing_vals["price_per_sqm_model"],
                            "price_per_sqm": listing_vals["price_per_sqm"],
                            "area_sqm_model": listing_vals["area_sqm_model"],
                            "views": listing_vals["views"],
                            "vat_status": listing_vals["vat_status"],
                            "geo_category": listing_vals["geo_category"],
                            "floor_model": listing_vals["floor_model"],
                            "total_floors_model": listing_vals["total_floors_model"],
                            "features_count": listing_vals["features_count"],
                            "training_eligible": listing_vals["training_eligible"],
                        },
                    )
                    .returning(Listing.id, Listing.published_date)
                )

                result = session.execute(stmt)
                row_result = result.fetchone()
                if row_result is None:
                    skipped += 1
                    continue

                listing_id, pub_date = row_result
                dom = (run_ts.date() - pub_date).days if pub_date else None

                snap = ListingSnapshot(
                    listing_id=listing_id,
                    scrape_run_id=db_run_id,
                    scraped_at=run_ts,
                    total_price=listing_vals["total_price"],
                    currency=listing_vals["currency"],
                    price_per_sqm_model=listing_vals["price_per_sqm_model"],
                    area_sqm_model=listing_vals["area_sqm_model"],
                    vat_status=listing_vals["vat_status"],
                    views=listing_vals["views"],
                    days_on_market=dom,
                    parsed_data=None,
                )
                session.add(snap)
                upserted += 1

        except sa_exc.OperationalError as exc:
            # Connection-level error: server dropped the connection.
            # The savepoint rollback itself may have failed, leaving the session
            # in an invalid state. Reset it so the next chunk can reconnect.
            url = row.get("ad_url", "?")
            print(f"    SKIP {url[:80]}: {exc} [CONNECTION ERROR — resetting session]")
            skipped += 1
            connection_lost = True
            try:
                session.rollback()
            except Exception:
                pass

        except Exception as exc:
            # Data error: savepoint was automatically rolled back by the context manager.
            url = row.get("ad_url", "?")
            print(f"    SKIP {url[:80]}: {exc}")
            skipped += 1

    if not connection_lost:
        # Commit the whole chunk (all successful savepoints) in one round-trip.
        session.commit()
    else:
        print(f"    Chunk aborted mid-way due to connection loss — {upserted} rows in this chunk were NOT committed (will be re-imported on re-run).")

    return upserted, skipped


def _import_directory(parquet_path: Path, run_ts: datetime) -> int:
    print(f"\n{'='*60}")
    print(f"Importing: {parquet_path.parent.name}")
    print(f"Parquet:   {parquet_path}")
    print(f"Timestamp: {run_ts.isoformat()}")

    df = pd.read_parquet(parquet_path)
    total = len(df)
    print(f"Rows:      {total:,}")

    db_run_id = uuid.uuid4()
    session = SessionLocal()

    try:
        db_run = ScrapeRun(
            id=db_run_id,
            started_at=run_ts,
            finished_at=run_ts,
            status="imported",
            deal_types=["prodazhbi"],
            notes=f"historical import from {parquet_path.parent.name}",
        )
        session.add(db_run)
        session.commit()
        print(f"Created scrape_run: {db_run_id}")

        upserted_total = 0
        skipped_total = 0
        chunk: list[dict] = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            raw = row.where(pd.notna(row), None).to_dict()
            chunk.append(raw)

            if len(chunk) >= CHUNK_SIZE:
                ups, skip = _upsert_chunk(session, chunk, db_run_id, run_ts)
                upserted_total += ups
                skipped_total += skip
                chunk = []

            if i % PROGRESS_EVERY == 0:
                print(
                    f"  [{i:,}/{total:,}] processed — "
                    f"{upserted_total:,} upserted, {skipped_total} skipped"
                )

        if chunk:
            ups, skip = _upsert_chunk(session, chunk, db_run_id, run_ts)
            upserted_total += ups
            skipped_total += skip

        session.execute(
            sqlalchemy.text(
                "UPDATE scrape_runs SET listings_found=:found, listings_upserted=:ups "
                "WHERE id=:id"
            ),
            {"found": total, "ups": upserted_total, "id": str(db_run_id)},
        )
        session.commit()

        print(
            f"\n  Done: {upserted_total:,} upserted, "
            f"{skipped_total} skipped out of {total:,} total."
        )
        return upserted_total

    except Exception as exc:
        session.rollback()
        print(f"  FATAL ERROR: {exc}")
        raise
    finally:
        session.close()


def main() -> None:
    print("Historical Import Script")
    print(f"Database: {settings.database_url.split('@')[-1]}")

    dirs = _find_parquet_dirs(settings.scrape_runs_dir.replace("scrape_runs", "parsed_sales_runs"))
    alt_dirs = _find_parquet_dirs(
        str(Path(__file__).resolve().parent.parent / "data" / "parsed_sales_runs")
    )
    seen: set[Path] = set()
    all_dirs: list[Path] = []
    for d in dirs + alt_dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            all_dirs.append(d)

    if not all_dirs:
        print("\nNo parsed_listings.parquet files found.")
        return

    # Sync taxonomy CSVs to DB first so property type slugs and geo paths are available.
    taxonomy_dir = str(Path(__file__).resolve().parent.parent / "data" / "taxonomy")
    print(f"\nSyncing taxonomy from {taxonomy_dir} ...")
    sync_taxonomy_to_db(taxonomy_dir)
    print("  Taxonomy synced.")

    print(f"\nFound {len(all_dirs)} director(y/ies) to import:")
    for d in all_dirs:
        print(f"  {d.name}")

    grand_total = 0
    for d in all_dirs:
        parquet_path = d / "parsed_listings.parquet"
        run_ts = _parse_dir_timestamp(d.name)
        grand_total += _import_directory(parquet_path, run_ts)

    print(f"\n{'='*60}")
    print(f"Import complete. Total upserted: {grand_total:,}")
    print("\nVerification:")
    print(
        "  SELECT deal_type_normalized, count(*), "
        "round(avg(price_per_sqm_model)::numeric,0) "
        "FROM listings GROUP BY 1 ORDER BY 1;"
    )


if __name__ == "__main__":
    main()
