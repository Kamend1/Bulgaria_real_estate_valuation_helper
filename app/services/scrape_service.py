"""
Scrape orchestration service.

run_scrape_background() is called from a daemon thread (not FastAPI BackgroundTasks).
It:
  1. Creates a ScrapeRun in the DB
  2. Captures stdout from existing scraper utilities via ProgressCapture
  3. Calls collect_listing_urls_for_routes_parallel_streaming
  4. Calls download_and_parse_listing_batch_streaming
  5. Ingests the resulting CSV into the DB via _ingest_csv_to_db
"""

import contextlib
import io
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

import csv as _csv_mod
from pathlib import Path as _Path

from app.config import settings
from app.db.models import Listing, ListingSnapshot, ScrapeRun, TaxonomyPropertyType, TaxonomyGeoPath
from app.db.session import db_session
from app.progress.store import ProgressRun, progress_store
from utils.feature_engineering import engineer_features, PROPERTY_TYPE_DISPLAY

_ROUTE_RE = re.compile(r"\[(\d+)/(\d+)\]\s+routes?\s+completed", re.IGNORECASE)
_LISTING_RE = re.compile(r"\[(\d+)/(\d+)\]\s+listings?\s+processed", re.IGNORECASE)


class ProgressCapture(io.StringIO):
    """
    Intercepts scraper print() output and forwards parsed progress events to SSE.
    Mirrors the real stdout so the console still receives output.
    """

    def __init__(self, run: ProgressRun, real_stdout: io.TextIOBase) -> None:
        super().__init__()
        self._run = run
        self._real = real_stdout
        self._buf = ""

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return

        self._run.last_message = line

        m = _ROUTE_RE.search(line)
        if m:
            self._run.routes_done = int(m.group(1))
            self._run.routes_total = int(m.group(2))
            self._run.push("progress", self._run.to_dict())
            return

        m = _LISTING_RE.search(line)
        if m:
            self._run.listings_done = int(m.group(1))
            self._run.listings_total = int(m.group(2))
            self._run.push("progress", self._run.to_dict())
            return

        self._run.push("log", {"message": line})


def _safe(value: Any, cast=None):
    """Convert NaN/None/empty string to None, optionally cast."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("nan", "none", "nat", ""):
        return None
    if cast is not None:
        try:
            return cast(s)
        except (ValueError, TypeError):
            return None
    return s


# Sanity caps: values above these are clearly parse errors (e.g. price ranges merged)
_MAX_TOTAL_PRICE = 99_999_999.0   # 100M EUR — anything above is implausible
_MAX_PPSQM = 9_999_999.0         # 10M EUR/sqm — impossible; NUMERIC(10,2) fits up to ~99M
_MAX_AREA = 9_999_999.0          # 10M sqm — impossible for a single listing


def _clamp_price(value: Any, max_val: float) -> float | None:
    """Return float if within range, else None (treats outlier as bad parse)."""
    f = _safe(value, float)
    if f is None:
        return None
    if f <= 0 or f > max_val:
        return None
    return f


def _build_listing_values(row: dict, run_id) -> dict:
    """Build the dict for INSERT/UPDATE into the listings table from an engineered row dict."""

    def _int(k):
        return _safe(row.get(k), int)

    def _float(k):
        return _safe(row.get(k), float)

    def _bool_flag(k):
        v = row.get(k)
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "1", "yes")

    return {
        "ad_url": str(row["ad_url"]).strip(),
        "ad_id": _safe(row.get("ad_id")),
        "last_scrape_run_id": run_id,
        # Raw parsed
        "listing_type": _safe(row.get("listing_type")),
        "deal_raw": _safe(row.get("deal_raw")),
        "property_type_raw": _safe(row.get("property_type_raw")),
        "title": _safe(row.get("title")),
        "title_city_raw": _safe(row.get("title_city_raw")),
        "title_geo_2_raw": _safe(row.get("title_geo_2_raw")),
        "location_raw": _safe(row.get("location_raw")),
        "total_price": _clamp_price(row.get("total_price"), _MAX_TOTAL_PRICE),
        "currency": _safe(row.get("currency")),
        "price_raw": _safe(row.get("price_raw")),
        "vat_status": _safe(row.get("vat_status")),
        "area_sqm": _clamp_price(row.get("area_sqm"), _MAX_AREA),
        "price_per_sqm": _clamp_price(row.get("price_per_sqm"), _MAX_PPSQM),
        "floor": _int("floor"),
        "total_floors": _int("total_floors"),
        "construction_type": _safe(row.get("construction_type")),
        "construction_year": _int("construction_year"),
        "description_clean": _safe(row.get("description_clean")),
        "features_pipe": _safe(row.get("features_pipe")),
        "features_count": _int("features_count"),
        "views": _int("views"),
        "published_raw": _safe(row.get("published_raw")),
        "training_eligible": _bool_flag("training_eligible"),
        "html_path": _safe(row.get("html_path")),
        "parse_error": _safe(row.get("parse_error")),
        "property_type_slug": _safe(row.get("property_type_slug")),
        # Engineered
        "deal_type_normalized": _safe(row.get("deal_type_normalized")),
        "published_date": row.get("published_date"),
        "price_per_sqm_model": _clamp_price(row.get("price_per_sqm_model"), _MAX_PPSQM),
        "area_sqm_model": _clamp_price(row.get("area_sqm_model"), _MAX_AREA),
        "title_city_model": _safe(row.get("title_city_model")),
        "title_geo_2_model": _safe(row.get("title_geo_2_model")),
        "location_level_1": _safe(row.get("location_level_1")),
        "location_level_2": _safe(row.get("location_level_2")),
        "location_level_3": _safe(row.get("location_level_3")),
        "location_level_1_model": _safe(row.get("location_level_1_model")),
        "location_level_2_model": _safe(row.get("location_level_2_model")),
        "location_level_3_model": _safe(row.get("location_level_3_model")),
        "geo_category": _safe(row.get("geo_category")),
        "exclude_foreign": _bool_flag("exclude_foreign"),
        "construction_type_model": _safe(row.get("construction_type_model")),
        "construction_year_model": _int("construction_year_model"),
        "floor_model": _int("floor_model"),
        "total_floors_model": _int("total_floors_model"),
        "floor_applicability": _safe(row.get("floor_applicability")),
    }


def sync_taxonomy_to_db(taxonomy_dir: str) -> None:
    """
    Reads the three taxonomy CSVs and upserts into taxonomy_property_types
    and taxonomy_geo_paths. Called once at the start of each scrape run or import.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    tax_dir = _Path(taxonomy_dir)

    prop_path = tax_dir / "valid_property_types.csv"
    geo_path  = tax_dir / "valid_geo_paths.csv"

    with db_session() as session:
        # Property types
        if prop_path.exists():
            with open(prop_path, encoding="utf-8") as f:
                for row in _csv_mod.DictReader(f):
                    slug = (row.get("property_type") or "").strip()
                    if not slug:
                        continue
                    stmt = (
                        pg_insert(TaxonomyPropertyType)
                        .values(
                            slug=slug,
                            display_name_bg=PROPERTY_TYPE_DISPLAY.get(slug, slug),
                            route_count=int(row.get("route_count") or 0),
                        )
                        .on_conflict_do_update(
                            index_elements=["slug"],
                            set_={"display_name_bg": PROPERTY_TYPE_DISPLAY.get(slug, slug),
                                  "route_count": int(row.get("route_count") or 0)},
                        )
                    )
                    session.execute(stmt)

        # Geo paths
        if geo_path.exists():
            with open(geo_path, encoding="utf-8") as f:
                for row in _csv_mod.DictReader(f):
                    deal = (row.get("deal_type") or "").strip()
                    path = (row.get("geo_path") or "").strip()
                    if not deal or not path:
                        continue
                    stmt = (
                        pg_insert(TaxonomyGeoPath)
                        .values(
                            deal_type=deal,
                            geo_path=path,
                            geo_level_count=int(row.get("geo_level_count") or 0),
                            geo_1=row.get("geo_1") or None,
                            geo_2=row.get("geo_2") or None,
                            geo_3=row.get("geo_3") or None,
                            route_count=int(row.get("route_count") or 0),
                        )
                        .on_conflict_do_update(
                            index_elements=["deal_type", "geo_path"],
                            set_={"route_count": int(row.get("route_count") or 0)},
                        )
                    )
                    session.execute(stmt)


def _ingest_rows_to_db(
    rows: list[dict],
    run_id,
    cumulative_offset: int = 0,
    progress_run: ProgressRun | None = None,
    progress_callback=None,
) -> int:
    """
    Upsert a batch of engineered row dicts into listings + insert snapshots.
    Uses savepoints (begin_nested) so one bad row never rolls back the whole batch.
    Returns number of rows upserted in this batch.
    """
    upserted = 0

    with db_session() as session:
        for raw_row in rows:
            try:
                row = engineer_features(raw_row)
            except Exception:
                continue

            if not row.get("ad_url"):
                continue

            listing_vals = _build_listing_values(row, run_id)

            try:
                with session.begin_nested():  # SAVEPOINT — isolates each row
                    stmt = (
                        pg_insert(Listing)
                        .values(**listing_vals)
                        .on_conflict_do_update(
                            index_elements=["ad_url"],
                            set_={
                                "last_seen_at": datetime.now(timezone.utc),
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
                        continue

                    listing_id, pub_date = row_result

                    dom = None
                    if pub_date:
                        dom = (datetime.now(timezone.utc).date() - pub_date).days

                    parsed_data_json = {
                        k: str(v) if v is not None else None
                        for k, v in row.items()
                        if k not in ("description_clean",)
                    }
                    session.add(ListingSnapshot(
                        listing_id=listing_id,
                        scrape_run_id=run_id,
                        total_price=listing_vals["total_price"],
                        currency=listing_vals["currency"],
                        price_per_sqm_model=listing_vals["price_per_sqm_model"],
                        area_sqm_model=listing_vals["area_sqm_model"],
                        vat_status=listing_vals["vat_status"],
                        views=listing_vals["views"],
                        days_on_market=dom,
                        parsed_data=parsed_data_json,
                    ))
                    upserted += 1

            except Exception:
                pass  # savepoint rolled back; session continues

    total_so_far = cumulative_offset + upserted
    if progress_run:
        progress_run.listings_upserted = total_so_far
    if progress_callback:
        progress_callback(total_so_far)

    return upserted


def _ingest_csv_to_db(
    csv_path: str,
    run_id,
    progress_run: ProgressRun | None = None,
    progress_callback=None,
) -> int:
    """Read a parsed_listings.csv and ingest via _ingest_rows_to_db in chunks."""
    import csv

    upserted_total = 0
    chunk: list[dict] = []
    chunk_size = 500

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            chunk.append(dict(raw_row))
            if len(chunk) >= chunk_size:
                upserted_total += _ingest_rows_to_db(
                    chunk, run_id,
                    cumulative_offset=upserted_total,
                    progress_run=progress_run,
                    progress_callback=progress_callback,
                )
                chunk = []

    if chunk:
        upserted_total += _ingest_rows_to_db(
            chunk, run_id,
            cumulative_offset=upserted_total,
            progress_run=progress_run,
            progress_callback=progress_callback,
        )

    return upserted_total


def run_scrape_background(
    run_id: str,
    deal_types: list[str],
    geo_paths: list[str],
    property_types: list[str],
) -> None:
    """
    Main background function — runs in daemon thread.
    All stdout from the existing scraper utils is captured via ProgressCapture.
    """
    progress_run = progress_store.get(run_id)
    if progress_run is None:
        return

    import uuid

    # Ensure run_id is a UUID object for the DB
    try:
        db_run_id = uuid.UUID(run_id)
    except ValueError:
        db_run_id = uuid.uuid4()

    try:
        # 1. Create scrape_run row in DB
        with db_session() as session:
            db_run = ScrapeRun(
                id=db_run_id,
                status="running",
                deal_types=deal_types,
                geo_paths=geo_paths,
                property_types=property_types,
            )
            session.add(db_run)

        # 2. Build output directory
        run_dir = Path(settings.scrape_runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        capture = ProgressCapture(progress_run, sys.stdout)

        with contextlib.redirect_stdout(capture):
            # 3. Refresh taxonomy from imot.bg sitemap (Notebook 01 logic)
            from utils.fetch_data.fetch_data_utils import (
                ScrapeSelection,
                collect_listing_urls_for_routes_parallel_streaming,
                download_and_parse_listing_batch_streaming,
                load_valid_deal_types,
                load_valid_geo_paths,
                load_valid_property_types,
                refresh_taxonomy,
            )
            import csv as _csv

            refresh_taxonomy(settings.taxonomy_dir)

            # 4. Load refreshed taxonomy and sync to DB
            tax = settings.taxonomy_dir
            valid_deal_types = load_valid_deal_types(f"{tax}/valid_deal_types.csv")
            valid_geo_paths = load_valid_geo_paths(f"{tax}/valid_geo_paths.csv")
            valid_property_types = load_valid_property_types(f"{tax}/valid_property_types.csv")

            sync_taxonomy_to_db(settings.taxonomy_dir)

            selection = ScrapeSelection(
                deal_types=deal_types,
                geo_paths=geo_paths,
                property_types=property_types or [],
            )

            route_urls = selection.build_urls(
                valid_deal_types=valid_deal_types,
                valid_geo_paths=valid_geo_paths,
                valid_property_types=valid_property_types,
            )
            progress_run.routes_total = len(route_urls)
            progress_run.push("progress", progress_run.to_dict())

            # 4. Collect listing URLs from all route pages
            route_result = collect_listing_urls_for_routes_parallel_streaming(
                route_urls=route_urls,
                checkpoint_dir=str(run_dir),
                max_workers=settings.scrape_max_workers_routes,
                delay_seconds=settings.scrape_delay_seconds,
            )

            # Read deduplicated listing URLs from the checkpoint CSV
            listing_urls_path = route_result["listing_urls_path"]
            seen_urls: set[str] = set()
            listing_urls: list[str] = []
            if Path(listing_urls_path).exists():
                with open(listing_urls_path, encoding="utf-8", errors="replace") as f:
                    for row in _csv.DictReader(f):
                        url = row.get("listing_url", "").strip()
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            listing_urls.append(url)

            progress_run.listings_total = len(listing_urls)
            progress_run.push("progress", progress_run.to_dict())

            # 5. Download and parse listings
            download_and_parse_listing_batch_streaming(
                listing_urls=listing_urls,
                output_dir=str(run_dir),
                max_workers=settings.scrape_max_workers_listings,
            )

        # 6. Ingest CSV to DB
        csv_path = run_dir / "parsed_listings.csv"
        if csv_path.exists():
            progress_run.push("log", {"message": f"Ingesting {csv_path.name} to DB…"})
            upserted = _ingest_csv_to_db(str(csv_path), db_run_id, progress_run)
        else:
            upserted = 0
            progress_run.push("log", {"message": "Warning: parsed_listings.csv not found"})

        # 7. Mark run complete in DB
        with db_session() as session:
            run_row = session.get(ScrapeRun, db_run_id)
            if run_row:
                run_row.status = "completed"
                run_row.finished_at = datetime.now(timezone.utc)
                run_row.routes_done = progress_run.routes_done
                run_row.routes_total = progress_run.routes_total
                run_row.listings_found = progress_run.listings_total
                run_row.listings_upserted = upserted

        # 8. Analytics: detect price events + refresh materialized view
        try:
            from app.services.analytics_service import compute_price_events, refresh_mv
            progress_run.push("log", {"message": "Изчисляване на ценови промени…"})
            n_events = compute_price_events(db_run_id)
            progress_run.push("log", {"message": f"Ценови събития: {n_events}. Обновяване на аналитичен изглед…"})
            refresh_mv()
            progress_run.push("log", {"message": "Аналитичният изглед е обновен."})
        except Exception as _ae:
            progress_run.push("log", {"message": f"Предупреждение: аналитиката не се обнови ({_ae})"})

        # 9. Embeddings: re-embed listings touched by this run (new or changed
        # since last embedded). Scoped to db_run_id, not the whole corpus --
        # see app/services/llm/embed_backfill.py. Non-fatal: missing
        # OPENAI_API_KEY or a transient API error should not fail the scrape.
        try:
            from app.services.llm.embed_backfill import backfill_embeddings

            progress_run.push("log", {"message": "Обновяване на семантични вектори (embeddings) за нови/променени обяви…"})
            with db_session() as session:
                n_embedded = backfill_embeddings(session, run_id=db_run_id)
            progress_run.push("log", {"message": f"Embeddings: {n_embedded} обяви ембед-нати/обновени."})
        except Exception as _ee:
            progress_run.push("log", {"message": f"Предупреждение: embeddings не се обновиха ({_ee})"})

        # 10. AVM: retrain segments whose training_eligible row count grew
        # enough since their active model was trained (Phase 14 Tier 2.1).
        # No-op on any machine without R2_MAINTAINER_* -- see
        # avm_retrain_service's own docstring for why that's the right gate.
        # Non-fatal like steps 8/9: a failed retrain never fails the scrape.
        try:
            from app.services.avm_retrain_service import maybe_retrain_avm_models

            with db_session() as session:
                retrain_results = maybe_retrain_avm_models(
                    session, on_progress=lambda msg: progress_run.push("log", {"message": msg})
                )
            retrained = [r["segment"] for r in retrain_results if r["action"] == "retrained"]
            if retrained:
                progress_run.push("log", {"message": f"AVM пре-трениране: {', '.join(retrained)}."})
        except Exception as _re:
            progress_run.push("log", {"message": f"Предупреждение: AVM пре-трениране не се изпълни ({_re})"})

        progress_run.status = "completed"
        progress_run.finished_at = datetime.utcnow()
        progress_run.push("done", progress_run.to_dict())

    except Exception as exc:
        tb = traceback.format_exc()
        progress_run.status = "failed"
        progress_run.error = str(exc)
        progress_run.push("error", {"message": str(exc), "traceback": tb})

        with db_session() as session:
            run_row = session.get(ScrapeRun, db_run_id)
            if run_row:
                run_row.status = "failed"
                run_row.finished_at = datetime.now(timezone.utc)
                run_row.error_message = str(exc)[:2000]


def get_scrape_status(db) -> dict:
    """Return data freshness info for the Scrape page status card."""
    from sqlalchemy import text as sa_text

    last_run = (
        db.query(ScrapeRun)
        .filter(ScrapeRun.status == "completed")
        .order_by(ScrapeRun.finished_at.desc())
        .first()
    )

    if last_run is None:
        return {
            "last_run": None,
            "age_days": None,
            "active_listings": 0,
            "recommendation": "no_data",
            "recommendation_label": "Няма данни — стартирайте засичане",
            "recommendation_class": "error",
        }

    age_days = (
        (datetime.now(timezone.utc) - last_run.finished_at).days
        if last_run.finished_at else None
    )

    active_listings = db.execute(
        sa_text("SELECT COUNT(*) FROM listings WHERE status = 'active'")
    ).scalar() or 0

    if age_days is None or age_days > 30:
        recommendation = "outdated"
        recommendation_label = f"Данните са остарели ({age_days or '?'} дни) — препоръчва се ново засичане"
        recommendation_class = "error"
    elif age_days > 14:
        recommendation = "stale"
        recommendation_label = f"Данните са на {age_days} дни — препоръчва се актуализация"
        recommendation_class = "warn"
    else:
        recommendation = "fresh"
        recommendation_label = f"Данните са актуални ({age_days} дни)"
        recommendation_class = "ok"

    return {
        "last_run": last_run,
        "age_days": age_days,
        "active_listings": active_listings,
        "recommendation": recommendation,
        "recommendation_label": recommendation_label,
        "recommendation_class": recommendation_class,
    }
