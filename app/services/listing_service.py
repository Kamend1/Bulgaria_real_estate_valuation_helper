from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

GEO_CATEGORIES = [
    ("sofia_center", "София — Център"),
    ("sofia_other", "София — Друго"),
    ("large_regional_city", "Голям областен град"),
    ("regional_city", "Областен град"),
    ("small_city", "Малък град"),
    ("sea_resort", "Морски курорт"),
    ("mountain_resort", "Планински курорт"),
    ("foreign", "Чужбина"),
    ("other_unknown", "Друго / Неизвестно"),
]

SORT_OPTIONS = [
    ("last_seen", "Последно виждан"),
    ("price_asc", "Цена ↑"),
    ("price_desc", "Цена ↓"),
    ("ppsqm_asc", "Цена/кв.м ↑"),
    ("ppsqm_desc", "Цена/кв.м ↓"),
    ("area_asc", "Площ ↑"),
    ("area_desc", "Площ ↓"),
]

_SORT_SQL = {
    "last_seen": "l.last_seen_at DESC",
    "price_asc": "l.total_price ASC NULLS LAST",
    "price_desc": "l.total_price DESC NULLS LAST",
    "ppsqm_asc": "l.price_per_sqm_model ASC NULLS LAST",
    "ppsqm_desc": "l.price_per_sqm_model DESC NULLS LAST",
    "area_asc": "l.area_sqm_model ASC NULLS LAST",
    "area_desc": "l.area_sqm_model DESC NULLS LAST",
}

RESULTS_PER_PAGE = 50


@dataclass
class SearchFilters:
    deal_type: str = ""
    city: list[str] = field(default_factory=list)   # exact title_city_model values
    quarter: str = ""
    property_types: list[str] = field(default_factory=list)
    geo_categories: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    min_area: float | None = None
    max_area: float | None = None
    min_ppsqm: float | None = None
    max_ppsqm: float | None = None
    sort: str = "last_seen"

    def to_query_string(self, exclude: set[str] | None = None) -> str:
        parts = []
        for k, v in self.__dict__.items():
            if exclude and k in exclude:
                continue
            if isinstance(v, list):
                for item in v:
                    parts.append(f"{k}={item}")
            elif v is None or v == "":
                continue
            else:
                parts.append(f"{k}={v}")
        return "&".join(parts)


def search_listings(
    db: Session,
    filters: SearchFilters,
    page: int = 1,
    per_page: int = RESULTS_PER_PAGE,
) -> tuple[list[dict[str, Any]], int]:
    """Returns (rows, total_count). Each row has price_trend and pct_change.

    SECURITY INVARIANT: `conditions` below gets joined and f-string'd
    directly into the SQL text (see `where_sql` / `main_sql`). That's only
    safe because every entry is a *literal, hardcoded* fragment
    (`"l.x = :name"`) — the actual user-supplied values always travel
    through `params` as bound placeholders, never through this list. If
    you add a new filter, follow that same split; do not f-string a
    filter *value* into `conditions` directly, even for something that
    looks harmless like a sort column name (see `_SORT_SQL`'s dict-lookup
    pattern for how to keep that safe too).
    """
    conditions = [
        "l.total_price IS NOT NULL",
        "l.price_per_sqm_model IS NOT NULL",
        "l.status = 'active'",
    ]
    params: dict[str, Any] = {}

    if filters.deal_type:
        conditions.append("l.deal_type_normalized = :deal_type")
        params["deal_type"] = filters.deal_type

    if filters.property_types:
        conditions.append("l.property_type_slug = ANY(:ptypes)")
        params["ptypes"] = filters.property_types

    if filters.geo_categories:
        conditions.append("l.geo_category = ANY(:geo_cats)")
        params["geo_cats"] = filters.geo_categories

    if filters.city:
        conditions.append(
            "(l.title_city_model = ANY(:cities) OR l.location_level_1_model = ANY(:cities))"
        )
        params["cities"] = filters.city

    if filters.quarter:
        conditions.append(
            "(l.location_level_2_model ILIKE :quarter OR l.title_geo_2_model ILIKE :quarter)"
        )
        params["quarter"] = f"%{filters.quarter}%"

    if filters.min_price is not None:
        conditions.append("l.total_price >= :min_price")
        params["min_price"] = filters.min_price
    if filters.max_price is not None:
        conditions.append("l.total_price <= :max_price")
        params["max_price"] = filters.max_price

    if filters.min_area is not None:
        conditions.append("l.area_sqm_model >= :min_area")
        params["min_area"] = filters.min_area
    if filters.max_area is not None:
        conditions.append("l.area_sqm_model <= :max_area")
        params["max_area"] = filters.max_area

    if filters.min_ppsqm is not None:
        conditions.append("l.price_per_sqm_model >= :min_ppsqm")
        params["min_ppsqm"] = filters.min_ppsqm
    if filters.max_ppsqm is not None:
        conditions.append("l.price_per_sqm_model <= :max_ppsqm")
        params["max_ppsqm"] = filters.max_ppsqm

    where_sql = " AND ".join(conditions)
    order_sql = _SORT_SQL.get(filters.sort, _SORT_SQL["last_seen"])

    total: int = db.execute(
        text(f"SELECT count(*) FROM listings l WHERE {where_sql}"),
        params,
    ).scalar_one()

    main_sql = f"""
    SELECT
        l.id, l.ad_url, l.deal_type_normalized, l.property_type_raw,
        l.title, l.title_city_raw, l.title_city_model,
        l.title_geo_2_raw, l.title_geo_2_model,
        l.location_raw, l.geo_category,
        l.total_price, l.currency, l.area_sqm_model, l.price_per_sqm_model,
        l.floor_model, l.total_floors_model, l.floor_applicability,
        l.construction_type_model, l.construction_year_model,
        l.published_date, l.last_seen_at, l.first_seen_at,
        l.features_count, l.views,
        prev.price_per_sqm_model AS prev_ppsqm,
        CASE
            WHEN prev.price_per_sqm_model IS NULL THEN 'new'
            WHEN l.price_per_sqm_model < prev.price_per_sqm_model THEN 'down'
            WHEN l.price_per_sqm_model > prev.price_per_sqm_model THEN 'up'
            ELSE 'same'
        END AS price_trend,
        CASE
            WHEN prev.price_per_sqm_model IS NULL THEN NULL
            ELSE round(
                (l.price_per_sqm_model - prev.price_per_sqm_model)
                / nullif(prev.price_per_sqm_model, 0) * 100,
                1
            )
        END AS pct_change
    FROM listings l
    LEFT JOIN LATERAL (
        SELECT price_per_sqm_model
        FROM listing_snapshots
        WHERE listing_id = l.id
        ORDER BY scraped_at DESC
        OFFSET 1 LIMIT 1
    ) prev ON true
    WHERE {where_sql}
    ORDER BY {order_sql}
    LIMIT :limit OFFSET :offset
    """

    params["limit"] = per_page
    params["offset"] = (page - 1) * per_page

    rows = db.execute(text(main_sql), params).mappings().all()
    return [dict(r) for r in rows], total


def get_cities_for_filter(db: Session) -> list[tuple[str, int]]:
    """Returns (city_model, count) ordered by count desc — only non-null, non-foreign."""
    rows = db.execute(text("""
        SELECT title_city_model, count(*) AS n
        FROM listings
        WHERE title_city_model IS NOT NULL
          AND (geo_category IS NULL OR geo_category != 'foreign')
        GROUP BY title_city_model
        ORDER BY n DESC
    """)).fetchall()
    return [(r[0], int(r[1])) for r in rows]


def get_quarters_for_filter(db: Session, cities: list[str]) -> list[str]:
    """Returns distinct location_level_2_model values for the given exact city values."""
    if not cities:
        return []
    rows = db.execute(
        text("""
            SELECT DISTINCT location_level_2_model
            FROM listings
            WHERE location_level_2_model IS NOT NULL
              AND location_level_2_model != ''
              AND title_city_model = ANY(:cities)
            ORDER BY 1
        """),
        {"cities": cities},
    ).fetchall()
    return [r[0] for r in rows]


def get_property_types_for_filter(db: Session) -> list[tuple[str, str, int]]:
    """Returns (slug, display_name_bg, count) ordered by listing count desc."""
    rows = db.execute(text("""
        SELECT t.slug, t.display_name_bg, count(l.id) AS n
        FROM taxonomy_property_types t
        LEFT JOIN listings l ON l.property_type_slug = t.slug
        GROUP BY t.slug, t.display_name_bg
        ORDER BY n DESC, t.display_name_bg
    """)).fetchall()
    return [(r[0], r[1] or r[0], int(r[2])) for r in rows]


def get_all_listing_ids(db: Session, filters: SearchFilters) -> list[int]:
    """Returns all matching listing IDs for the given filters (used for select-all)."""
    conditions = [
        "l.total_price IS NOT NULL",
        "l.price_per_sqm_model IS NOT NULL",
        "l.status = 'active'",
    ]
    params: dict = {}

    if filters.deal_type:
        conditions.append("l.deal_type_normalized = :deal_type")
        params["deal_type"] = filters.deal_type
    if filters.property_types:
        conditions.append("l.property_type_slug = ANY(:ptypes)")
        params["ptypes"] = filters.property_types
    if filters.geo_categories:
        conditions.append("l.geo_category = ANY(:geo_cats)")
        params["geo_cats"] = filters.geo_categories
    if filters.city:
        conditions.append("(l.title_city_model = ANY(:cities) OR l.location_level_1_model = ANY(:cities))")
        params["cities"] = filters.city
    if filters.quarter:
        conditions.append("(l.location_level_2_model ILIKE :quarter OR l.title_geo_2_model ILIKE :quarter)")
        params["quarter"] = f"%{filters.quarter}%"
    if filters.min_price is not None:
        conditions.append("l.total_price >= :min_price")
        params["min_price"] = filters.min_price
    if filters.max_price is not None:
        conditions.append("l.total_price <= :max_price")
        params["max_price"] = filters.max_price
    if filters.min_area is not None:
        conditions.append("l.area_sqm_model >= :min_area")
        params["min_area"] = filters.min_area
    if filters.max_area is not None:
        conditions.append("l.area_sqm_model <= :max_area")
        params["max_area"] = filters.max_area
    if filters.min_ppsqm is not None:
        conditions.append("l.price_per_sqm_model >= :min_ppsqm")
        params["min_ppsqm"] = filters.min_ppsqm
    if filters.max_ppsqm is not None:
        conditions.append("l.price_per_sqm_model <= :max_ppsqm")
        params["max_ppsqm"] = filters.max_ppsqm

    where_sql = " AND ".join(conditions)
    rows = db.execute(
        text(f"SELECT l.id FROM listings l WHERE {where_sql}"),
        params,
    ).fetchall()
    return [r[0] for r in rows]


def get_listing_detail(db: Session, listing_id: int) -> dict[str, Any] | None:
    row = db.execute(
        text("SELECT l.* FROM listings l WHERE l.id = :id"),
        {"id": listing_id},
    ).mappings().first()
    return dict(row) if row else None


def get_listing_snapshots(db: Session, listing_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        text("""
        SELECT s.id, s.scraped_at, s.total_price, s.currency,
               s.price_per_sqm_model, s.area_sqm_model, s.views, s.days_on_market
        FROM listing_snapshots s
        WHERE s.listing_id = :id
        ORDER BY s.scraped_at DESC
        """),
        {"id": listing_id},
    ).mappings().all()
    return [dict(r) for r in rows]
