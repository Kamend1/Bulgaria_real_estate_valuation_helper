"""
Scrape orchestration service -- shared helpers used by scripts/run_scrape.py,
the ACTUAL scrape pipeline (launched via subprocess.Popen from
app/routers/scrape.py). Everything below is a plain function library, not an
orchestrator: taxonomy sync, CSV-to-DB ingestion, and the freshness status
card's data.

Historical note (2026-09-11): this file used to also define
run_scrape_background(), a daemon-thread orchestrator with its own
per-step pipeline (taxonomy -> routes -> download -> ingest -> analytics ->
embeddings -> AVM retrain). It was never called from anywhere -- real
scrapes have always run through scripts/run_scrape.py's own, separately
maintained copy of that pipeline instead. That gap silently ate two real
features that were only ever added here: automatic embeddings refresh
(caught 2026-08-28, fixed by copying the step into run_scrape.py) and
automatic AVM retraining (caught 2026-09-11, same fix). Removed for good
this time, rather than leaving a second well-intentioned future step to
land in the one place that never runs -- see run_scrape.py for the real
pipeline.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

import csv as _csv_mod
from pathlib import Path as _Path

from app.db.models import Listing, ListingSnapshot, ScrapeRun, TaxonomyPropertyType, TaxonomyGeoPath
from app.db.session import db_session
from utils.feature_engineering import engineer_features, PROPERTY_TYPE_DISPLAY


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
                                # Being re-ingested in a real scrape is definitional proof the
                                # listing is live -- revive it even if a prior run (wrongly, e.g.
                                # due to a transient outage) archived it. See 2026-09-11 incident:
                                # without this, listings archived-then-recovered stayed stuck.
                                "status": "active",
                                "archived_at": None,
                                "archived_by_run_id": None,
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
                    # on_conflict_do_nothing (not do_update): re-ingesting the same
                    # (listing, run) pair -- e.g. a recovery script re-processing an
                    # already-ingested CSV -- must be a true no-op here. See the
                    # 2026-09-11 incident: a bare session.add() with no idempotency
                    # guard created 129,197 duplicate snapshot rows for one run,
                    # inflating every mv_analytics_flat-derived count.
                    snap_stmt = (
                        pg_insert(ListingSnapshot)
                        .values(
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
                        )
                        .on_conflict_do_nothing(
                            index_elements=["listing_id", "scrape_run_id"],
                        )
                    )
                    session.execute(snap_stmt)
                    upserted += 1

            except Exception:
                pass  # savepoint rolled back; session continues

    total_so_far = cumulative_offset + upserted
    if progress_callback:
        progress_callback(total_so_far)

    return upserted


def _ingest_csv_to_db(
    csv_path: str,
    run_id,
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
                    progress_callback=progress_callback,
                )
                chunk = []

    if chunk:
        upserted_total += _ingest_rows_to_db(
            chunk, run_id,
            cumulative_offset=upserted_total,
            progress_callback=progress_callback,
        )

    return upserted_total


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
