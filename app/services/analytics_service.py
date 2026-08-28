"""
Analytics service: market trend computation, price event detection, MV refresh.

All heavy aggregation stays in SQL (no pandas in request handlers).
"""

import logging
from typing import Optional

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.db.session import db_session

logger = logging.getLogger(__name__)

# ── Materialized view ─────────────────────────────────────────────────────────

def refresh_mv() -> None:
    """Refresh mv_analytics_flat concurrently. Non-fatal if MV doesn't exist yet."""
    try:
        with db_session() as session:
            session.execute(text(
                "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_analytics_flat"
            ))
        logger.info("mv_analytics_flat refreshed")
    except Exception as exc:
        logger.warning("MV refresh failed (non-fatal): %s", exc)


# ── Price event detection ─────────────────────────────────────────────────────

def compute_price_events(run_id) -> int:
    """
    For each listing updated in `run_id`, compare its snapshot with the immediately
    preceding snapshot from a different run. Upsert into listing_price_events.
    Returns number of events written.
    """
    sql = text("""
        INSERT INTO listing_price_events
            (listing_id, from_run_id, to_run_id, observed_at,
             price_before, price_after, ppsqm_before, ppsqm_after,
             change_pct, direction, days_between)
        WITH current_snaps AS (
            SELECT ls.listing_id,
                   ls.scrape_run_id,
                   ls.scraped_at,
                   ls.total_price,
                   ls.price_per_sqm_model
            FROM listing_snapshots ls
            WHERE ls.scrape_run_id = :run_id
              AND ls.price_per_sqm_model IS NOT NULL
        ),
        prev_snaps AS (
            SELECT DISTINCT ON (ls.listing_id)
                   ls.listing_id,
                   ls.scrape_run_id         AS prev_run_id,
                   ls.scraped_at            AS prev_scraped_at,
                   ls.total_price           AS prev_price,
                   ls.price_per_sqm_model   AS prev_ppsqm
            FROM listing_snapshots ls
            JOIN current_snaps cs ON cs.listing_id = ls.listing_id
            WHERE ls.scrape_run_id != :run_id
              AND ls.price_per_sqm_model IS NOT NULL
              AND ls.scraped_at < cs.scraped_at
            ORDER BY ls.listing_id, ls.scraped_at DESC
        ),
        events AS (
            SELECT
                cs.listing_id,
                ps.prev_run_id,
                cs.scrape_run_id,
                now()                                               AS observed_at,
                ps.prev_price,
                cs.total_price,
                ps.prev_ppsqm,
                cs.price_per_sqm_model,
                ROUND(
                    ((cs.price_per_sqm_model - ps.prev_ppsqm)
                     / NULLIF(ps.prev_ppsqm, 0) * 100)::numeric, 4
                )                                                   AS chg_pct,
                EXTRACT(DAY FROM (cs.scraped_at - ps.prev_scraped_at))::integer
                                                                    AS days_btw
            FROM current_snaps cs
            JOIN prev_snaps ps ON ps.listing_id = cs.listing_id
            -- Exclude pairs where price changed > 500% — almost certainly data quality issues
            WHERE ABS(
                (cs.price_per_sqm_model - ps.prev_ppsqm)
                / NULLIF(ps.prev_ppsqm, 0) * 100
            ) <= 500
        )
        SELECT
            listing_id,
            prev_run_id,
            scrape_run_id,
            observed_at,
            prev_price,
            total_price,
            prev_ppsqm,
            price_per_sqm_model,
            chg_pct,
            CASE
                WHEN chg_pct < -0.5  THEN 'reduced'
                WHEN chg_pct >  0.5  THEN 'increased'
                ELSE 'stable'
            END,
            days_btw
        FROM events
        ON CONFLICT (listing_id, to_run_id) DO UPDATE SET
            from_run_id   = EXCLUDED.from_run_id,
            price_before  = EXCLUDED.price_before,
            price_after   = EXCLUDED.price_after,
            ppsqm_before  = EXCLUDED.ppsqm_before,
            ppsqm_after   = EXCLUDED.ppsqm_after,
            change_pct    = EXCLUDED.change_pct,
            direction     = EXCLUDED.direction,
            days_between  = EXCLUDED.days_between,
            observed_at   = EXCLUDED.observed_at
    """)
    try:
        with db_session() as session:
            result = session.execute(sql, {"run_id": run_id})
            count = result.rowcount
        logger.info("compute_price_events: %d events for run %s", count, run_id)
        return count
    except Exception as exc:
        logger.warning("compute_price_events failed (non-fatal): %s", exc)
        return 0


# ── Filter options ────────────────────────────────────────────────────────────

def get_filter_options(db: Session) -> dict:
    """Lists for all analytics filter dropdowns, drawn from actual data."""
    cities = db.execute(text("""
        SELECT DISTINCT title_city_model
        FROM listings
        WHERE title_city_model IS NOT NULL
          AND training_eligible = TRUE
        ORDER BY title_city_model
    """)).scalars().all()

    prop_types = db.execute(text("""
        SELECT pt.slug, pt.display_name_bg
        FROM taxonomy_property_types pt
        WHERE EXISTS (
            SELECT 1 FROM listings l
            WHERE l.property_type_slug = pt.slug AND l.training_eligible = TRUE
        )
        ORDER BY pt.display_name_bg
    """)).all()

    return {
        "cities": list(cities),
        "property_types": [(r[0], r[1] or r[0]) for r in prop_types],
    }


def get_neighborhoods_for_city(db: Session, city: str) -> list[str]:
    rows = db.execute(text("""
        SELECT DISTINCT title_geo_2_model
        FROM listings
        WHERE title_city_model = :city
          AND title_geo_2_model IS NOT NULL
          AND training_eligible = TRUE
        ORDER BY title_geo_2_model
    """), {"city": city}).scalars().all()
    return list(rows)


# ── Market trend ──────────────────────────────────────────────────────────────

def get_market_trend(
    db: Session,
    deal_type: Optional[str] = None,
    geo_category: Optional[str] = None,
    city: Optional[str] = None,
    neighborhood: Optional[str] = None,
    property_type_slug: Optional[str] = None,
    n_runs: int = 12,
) -> list[dict]:
    """
    Time series of price stats per scrape run (chronological order).
    Uses mv_analytics_flat. Falls back to empty list if MV not yet populated.
    """
    sql = text("""
        WITH trend AS (
            SELECT
                mv.scrape_run_id,
                mv.run_started_at,
                COUNT(*)                                                            AS n_listings,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY mv.price_per_sqm_model)::numeric, 2)                  AS median_ppsqm,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                    ORDER BY mv.price_per_sqm_model)::numeric, 2)                  AS p25_ppsqm,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                    ORDER BY mv.price_per_sqm_model)::numeric, 2)                  AS p75_ppsqm,
                ROUND(AVG(mv.price_per_sqm_model)::numeric, 2)                    AS mean_ppsqm,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY mv.days_on_market
                ) FILTER (WHERE mv.days_on_market IS NOT NULL)::numeric, 0)        AS median_dom
            FROM mv_analytics_flat mv
            WHERE (:deal_type     IS NULL OR mv.deal_type_normalized = :deal_type)
              AND (:geo_category  IS NULL OR mv.geo_category         = :geo_category)
              AND (:city          IS NULL OR mv.title_city_model     = :city)
              AND (:neighborhood  IS NULL OR mv.title_geo_2_model    = :neighborhood)
              AND (:prop_type     IS NULL OR mv.property_type_slug   = :prop_type)
            GROUP BY mv.scrape_run_id, mv.run_started_at
            ORDER BY mv.run_started_at DESC
            LIMIT :n_runs
        )
        SELECT * FROM trend ORDER BY run_started_at ASC
    """)
    try:
        rows = db.execute(sql, {
            "deal_type":    deal_type or None,
            "geo_category": geo_category or None,
            "city":         city or None,
            "neighborhood": neighborhood or None,
            "prop_type":    property_type_slug or None,
            "n_runs":       n_runs,
        }).fetchall()
    except Exception as exc:
        logger.warning("get_market_trend query failed: %s", exc)
        return []

    return [
        {
            "run_id":       str(r[0]),
            "run_date":     r[1].strftime("%d.%m.%Y") if r[1] else None,
            "n_listings":   int(r[2]),
            "median_ppsqm": float(r[3]) if r[3] is not None else None,
            "p25_ppsqm":    float(r[4]) if r[4] is not None else None,
            "p75_ppsqm":    float(r[5]) if r[5] is not None else None,
            "mean_ppsqm":   float(r[6]) if r[6] is not None else None,
            "median_dom":   int(r[7])   if r[7] is not None else None,
        }
        for r in rows
    ]


_STATS_AGG_COLUMNS = """
    count(*)                                                                          AS n_listings,
    round(percentile_cont(0.5) WITHIN GROUP (
        ORDER BY price_per_sqm_model)::numeric, 2)                                    AS median_ppsqm,
    round(percentile_cont(0.25) WITHIN GROUP (
        ORDER BY price_per_sqm_model)::numeric, 2)                                    AS p25_ppsqm,
    round(percentile_cont(0.75) WITHIN GROUP (
        ORDER BY price_per_sqm_model)::numeric, 2)                                    AS p75_ppsqm,
    round(AVG(price_per_sqm_model)::numeric, 2)                                       AS mean_ppsqm,
    round(percentile_cont(0.5) WITHIN GROUP (
        ORDER BY days_on_market
    ) FILTER (WHERE days_on_market IS NOT NULL)::numeric, 0)                          AS median_dom
"""

# group_by -> the mv_analytics_flat column it aggregates by, for
# get_market_stats_flexible(). Validated against this fixed whitelist
# before being interpolated into the SQL string -- never raw user/model
# text (SECURITY INVARIANT, same discipline as listing_service.search_listings
# and retriever.py's condition-fragment style).
_GROUP_BY_COLUMNS = {
    "geo_category": "geo_category",
    "city": "title_city_model",
    "neighborhood": "title_geo_2_model",
    "construction_type": "construction_type_model",
    "property_type_slug": "property_type_slug",
}


def get_market_stats_flexible(
    db: Session,
    deal_type: str,
    property_type_slugs: Optional[list[str]] = None,
    geo_categories: Optional[list[str]] = None,
    cities: Optional[list[str]] = None,
    neighborhoods: Optional[list[str]] = None,
    construction_types: Optional[list[str]] = None,
    area_min: Optional[float] = None,
    area_max: Optional[float] = None,
    group_by: str = "run",
    n_runs: int = 6,
) -> dict:
    """Flexible cross-section over mv_analytics_flat for the market analyst
    agent (Phase 10) -- generalizes get_market_trend() above from
    single-value filters to lists (so "двустаен AND тристаен" or several
    neighborhoods at once work in one call), adds an area band and a
    construction_type filter, and adds group_by so the SAME query answers
    both "time series for one segment" (group_by="run", mirrors
    get_market_trend's own shape) and "side-by-side comparison across a
    dimension" (group_by="neighborhood"/"construction_type"/etc,
    aggregated over the last n_runs pooled together -- a more robust
    snapshot than any single run alone, especially for smaller segments).

    Every filter is built as a condition fragment only when the caller
    actually supplies it (retriever.py's established style) -- list-valued
    filters use IN with an expanding bindparam, never string-formatted
    values. group_by is checked against _GROUP_BY_COLUMNS above before use.
    Falls back to an empty result (not an exception) if mv_analytics_flat
    isn't populated yet."""
    if group_by != "run" and group_by not in _GROUP_BY_COLUMNS:
        raise ValueError(f"Invalid group_by {group_by!r}. Valid: 'run' or one of {sorted(_GROUP_BY_COLUMNS)}")

    conditions = ["mv.deal_type_normalized = :deal_type"]
    params: dict = {"deal_type": deal_type, "n_runs": n_runs}
    expanding: list[str] = []

    if property_type_slugs:
        conditions.append("mv.property_type_slug IN :ptypes")
        params["ptypes"] = tuple(property_type_slugs)
        expanding.append("ptypes")
    if geo_categories:
        conditions.append("mv.geo_category IN :geo_cats")
        params["geo_cats"] = tuple(geo_categories)
        expanding.append("geo_cats")
    if cities:
        conditions.append("mv.title_city_model IN :cities")
        params["cities"] = tuple(cities)
        expanding.append("cities")
    if neighborhoods:
        conditions.append("mv.title_geo_2_model ILIKE ANY(:neigh_patterns)")
        params["neigh_patterns"] = [f"%{n}%" for n in neighborhoods]
    if construction_types:
        conditions.append("mv.construction_type_model IN :constr_types")
        params["constr_types"] = tuple(construction_types)
        expanding.append("constr_types")
    if area_min is not None:
        conditions.append("mv.area_sqm_model >= :area_min")
        params["area_min"] = area_min
    if area_max is not None:
        conditions.append("mv.area_sqm_model <= :area_max")
        params["area_max"] = area_max

    where_sql = " AND ".join(conditions)

    if group_by == "run":
        sql = text(f"""
            WITH recent_runs AS (
                SELECT DISTINCT scrape_run_id, run_started_at
                FROM mv_analytics_flat
                ORDER BY run_started_at DESC
                LIMIT :n_runs
            ),
            filtered AS (
                SELECT mv.* FROM mv_analytics_flat mv
                JOIN recent_runs rr ON rr.scrape_run_id = mv.scrape_run_id
                WHERE {where_sql}
            )
            SELECT scrape_run_id, run_started_at, {_STATS_AGG_COLUMNS}
            FROM filtered
            GROUP BY scrape_run_id, run_started_at
            ORDER BY run_started_at ASC
        """)
    else:
        group_col = _GROUP_BY_COLUMNS[group_by]
        sql = text(f"""
            WITH recent_runs AS (
                SELECT DISTINCT scrape_run_id, run_started_at
                FROM mv_analytics_flat
                ORDER BY run_started_at DESC
                LIMIT :n_runs
            ),
            filtered AS (
                SELECT mv.* FROM mv_analytics_flat mv
                JOIN recent_runs rr ON rr.scrape_run_id = mv.scrape_run_id
                WHERE {where_sql}
            )
            SELECT {group_col} AS group_value, {_STATS_AGG_COLUMNS}
            FROM filtered
            WHERE {group_col} IS NOT NULL
            GROUP BY {group_col}
            ORDER BY n_listings DESC
        """)

    if expanding:
        sql = sql.bindparams(*(bindparam(name, expanding=True) for name in expanding))

    try:
        rows = db.execute(sql, params).mappings().all()
    except Exception as exc:
        logger.warning("get_market_stats_flexible query failed: %s", exc)
        return {"group_by": group_by, "rows": []}

    if group_by == "run":
        out_rows = [
            {
                "run_id": str(r["scrape_run_id"]),
                "run_date": r["run_started_at"].strftime("%d.%m.%Y") if r["run_started_at"] else None,
                "n_listings": int(r["n_listings"]),
                "median_ppsqm": float(r["median_ppsqm"]) if r["median_ppsqm"] is not None else None,
                "p25_ppsqm": float(r["p25_ppsqm"]) if r["p25_ppsqm"] is not None else None,
                "p75_ppsqm": float(r["p75_ppsqm"]) if r["p75_ppsqm"] is not None else None,
                "mean_ppsqm": float(r["mean_ppsqm"]) if r["mean_ppsqm"] is not None else None,
                "median_dom": int(r["median_dom"]) if r["median_dom"] is not None else None,
            }
            for r in rows
        ]
    else:
        out_rows = [
            {
                "group_value": r["group_value"],
                "n_listings": int(r["n_listings"]),
                "median_ppsqm": float(r["median_ppsqm"]) if r["median_ppsqm"] is not None else None,
                "p25_ppsqm": float(r["p25_ppsqm"]) if r["p25_ppsqm"] is not None else None,
                "p75_ppsqm": float(r["p75_ppsqm"]) if r["p75_ppsqm"] is not None else None,
                "mean_ppsqm": float(r["mean_ppsqm"]) if r["mean_ppsqm"] is not None else None,
                "median_dom": int(r["median_dom"]) if r["median_dom"] is not None else None,
            }
            for r in rows
        ]

    return {"group_by": group_by, "n_runs_included": n_runs, "rows": out_rows}


def get_market_filter_values(db: Session, city: Optional[str] = None) -> dict:
    """Real, currently-queryable values for the market analyst's filter
    parameters -- prevents the model from guessing a city/neighborhood
    spelling that doesn't literally match what's stored (e.g. "Sofia"
    instead of "град софия"). geo_category/property-type segments are a
    fixed, known vocabulary (no query needed); cities/neighborhoods/
    construction types come from mv_analytics_flat itself, ranked by
    listing count so the most useful values sort first."""
    from utils.ml.avm_features import GEO_CATEGORIES, SEGMENT_DISPLAY_NAMES, SEGMENT_PROPERTY_TYPES

    cities = db.execute(text("""
        SELECT title_city_model, count(*) AS n FROM mv_analytics_flat
        WHERE title_city_model IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 30
    """)).all()
    construction_types = db.execute(text("""
        SELECT construction_type_model, count(*) AS n FROM mv_analytics_flat
        WHERE construction_type_model IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 20
    """)).all()
    neighborhoods = []
    if city:
        neighborhoods = db.execute(text("""
            SELECT title_geo_2_model, count(*) AS n FROM mv_analytics_flat
            WHERE title_city_model = :city AND title_geo_2_model IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT 40
        """), {"city": city}).all()

    return {
        "geo_categories": GEO_CATEGORIES,
        "segments": {seg: {"display_name": SEGMENT_DISPLAY_NAMES[seg], "property_type_slugs": slugs} for seg, slugs in SEGMENT_PROPERTY_TYPES.items()},
        "top_cities": [{"city": r[0], "n_listings": int(r[1])} for r in cities],
        "construction_types": [{"construction_type": r[0], "n_listings": int(r[1])} for r in construction_types],
        "neighborhoods_for_city": [{"neighborhood": r[0], "n_listings": int(r[1])} for r in neighborhoods] if city else None,
    }


def get_repeat_stats(
    db: Session,
    deal_type: Optional[str] = None,
    geo_category: Optional[str] = None,
    city: Optional[str] = None,
    property_type_slug: Optional[str] = None,
    n_runs: int = 12,
) -> list[dict]:
    """Per-run breakdown of repeat listings: reduced / stable / increased."""
    sql = text("""
        WITH run_events AS (
            SELECT
                pe.to_run_id,
                sr.started_at                                                   AS run_started_at,
                COUNT(*)                                                        AS repeat_n,
                COUNT(*) FILTER (WHERE pe.direction = 'reduced')               AS n_reduced,
                COUNT(*) FILTER (WHERE pe.direction = 'increased')             AS n_increased,
                COUNT(*) FILTER (WHERE pe.direction = 'stable')                AS n_stable,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY pe.change_pct
                ) FILTER (WHERE pe.direction = 'reduced')::numeric, 2)         AS median_reduction_pct
            FROM listing_price_events pe
            JOIN scrape_runs sr ON sr.id = pe.to_run_id
            JOIN listings l     ON l.id  = pe.listing_id
            WHERE sr.status IN ('completed', 'imported')
              AND (:deal_type    IS NULL OR l.deal_type_normalized = :deal_type)
              AND (:geo_category IS NULL OR l.geo_category         = :geo_category)
              AND (:city         IS NULL OR l.title_city_model     = :city)
              AND (:prop_type    IS NULL OR l.property_type_slug   = :prop_type)
            GROUP BY pe.to_run_id, sr.started_at
            ORDER BY sr.started_at DESC
            LIMIT :n_runs
        )
        SELECT * FROM run_events ORDER BY run_started_at ASC
    """)
    try:
        rows = db.execute(sql, {
            "deal_type":    deal_type or None,
            "geo_category": geo_category or None,
            "city":         city or None,
            "prop_type":    property_type_slug or None,
            "n_runs":       n_runs,
        }).fetchall()
    except Exception as exc:
        logger.warning("get_repeat_stats query failed: %s", exc)
        return []

    return [
        {
            "run_date":             r[1].strftime("%d.%m.%Y") if r[1] else None,
            "repeat_n":             int(r[2]),
            "n_reduced":            int(r[3]),
            "n_increased":          int(r[4]),
            "n_stable":             int(r[5]),
            "median_reduction_pct": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]
