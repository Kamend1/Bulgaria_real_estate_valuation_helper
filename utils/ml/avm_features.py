"""
Shared feature-column definitions for the segment-aware AVM (automated
valuation model). Used by both scripts/train_avm_model.py (training) and
app/services/avm_service.py (inference) so the two can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# ── Segment definitions ──────────────────────────────────────────────────────
# Taxonomy slugs (data/taxonomy/valid_property_types.csv) grouped into asset
# classes with similar price drivers. Land (partsel, zemedelska-zemya, myasto)
# and garazh-parkomyasto are intentionally excluded — priced per decare/plot
# or negligible variance, not a price/sqm regression fit.

SEGMENT_PROPERTY_TYPES: dict[str, list[str]] = {
    "residential": [
        "ednostaen", "dvustaen", "tristaen", "chetiristaen", "mnogostaen",
        "mezonet", "atelie-tavan", "etazh-ot-kashta", "kashta", "vila", "staya",
    ],
    "office": ["ofis"],
    "retail": ["magazin", "zavedenie", "biznes-imot"],
    "industrial": ["sklad", "promishleno-pomeshtenie"],
    "hospitality": ["hotel"],
}

SEGMENT_DISPLAY_NAMES: dict[str, str] = {
    "residential": "Жилищни имоти",
    "office": "Офиси",
    "retail": "Търговски имоти",
    "industrial": "Индустриални имоти",
    "hospitality": "Хотелиерски имоти",
}

_SLUG_TO_SEGMENT: dict[str, str] = {
    slug: segment
    for segment, slugs in SEGMENT_PROPERTY_TYPES.items()
    for slug in slugs
}


def segment_for_property_type(slug: str | None) -> str | None:
    return _SLUG_TO_SEGMENT.get(slug) if slug else None


# ── Feature columns (same shape for every segment) ───────────────────────────
# Ported verbatim from notebooks/04_residential_real_estate_regression_ML_analysis.ipynb.
# Binary feature_* amenity columns are segment-specific and recomputed at
# training time (see scripts/train_avm_model.py) — they live in each
# avm_models.feature_columns row, not here.

NUMERIC_COLS = [
    "area_sqm", "views", "features_count",
    "construction_year_model", "floor_model", "total_floors_model",
]

CATEGORICAL_COLS = [
    "property_type_raw", "title_city_model", "title_geo_2_model",
    "geo_category", "construction_type_model",
]

# The 9 buckets produced by map_geo_category() in
# utils/feature_engineering/feature_engineering_utils.py.
GEO_CATEGORIES = [
    "sofia_center", "sofia_other", "large_regional_city", "regional_city",
    "small_city", "sea_resort", "mountain_resort", "other_unknown", "foreign",
]

REQUIRED_SUBJECT_FIELDS = [
    "subject_area_sqm", "subject_city", "subject_property_type", "subject_geo_category",
]


def missing_subject_fields(report) -> list[str]:
    """Required AppraisalReport fields that are empty. Empty list = ready to predict."""
    return [f for f in REQUIRED_SUBJECT_FIELDS if getattr(report, f, None) in (None, "")]


def get_property_type_raw_for_slug(db: Session, slug: str) -> str | None:
    """
    Most common raw scraped `property_type_raw` value for a taxonomy slug
    (e.g. "dvustaen" -> "2-СТАЕН"). The model was trained on the raw scraped
    label, not the taxonomy slug, so a manually-chosen subject property type
    needs to be translated to the label the categorical encoder actually saw.
    """
    row = db.execute(text("""
        SELECT property_type_raw, count(*) AS n
        FROM listings
        WHERE property_type_slug = :slug AND property_type_raw IS NOT NULL
        GROUP BY property_type_raw
        ORDER BY n DESC
        LIMIT 1
    """), {"slug": slug}).first()
    return row[0] if row else None


def build_feature_row(db: Session, report, feature_columns: list[str]) -> pd.DataFrame:
    """
    Builds a single-row DataFrame matching the exact column layout an
    avm_models pipeline was fit on. Raises ValueError if required subject
    fields are missing (caller should check missing_subject_fields() first
    to surface a friendlier message).
    """
    missing = missing_subject_fields(report)
    if missing:
        raise ValueError(f"missing subject fields: {', '.join(missing)}")

    raw_label = get_property_type_raw_for_slug(db, report.subject_property_type)

    def _num(v):
        return float(v) if v is not None else np.nan

    row = {
        "area_sqm": _num(report.subject_area_sqm),
        "views": np.nan,                 # unknown for an unpublished/subject property
        "features_count": 0,             # v1 collects no subject amenity checklist
        "construction_year_model": _num(report.subject_year),
        "floor_model": _num(report.subject_floor),
        "total_floors_model": _num(report.subject_total_floors),
        "property_type_raw": raw_label,
        "title_city_model": report.subject_city or None,
        "title_geo_2_model": report.subject_neighborhood or None,
        "geo_category": report.subject_geo_category,
        "construction_type_model": report.subject_construction or None,
    }
    for col in feature_columns:
        if col.startswith("feature_"):
            row[col] = 0

    return pd.DataFrame([row])[list(feature_columns)]


def prep_for_catboost(row: pd.DataFrame, categorical_cols: list[str] = CATEGORICAL_COLS) -> pd.DataFrame:
    """
    CatBoost needs categorical columns as non-null strings (it handles NaN
    numerics natively, but not NaN categoricals passed via cat_features).
    build_feature_row()'s output is otherwise already CatBoost-ready — it
    never one-hot-encodes, that only happens inside the LightGBM sklearn
    Pipeline's ColumnTransformer, not in this module.
    """
    row = row.copy()
    for c in categorical_cols:
        row[c] = row[c].fillna("missing").astype(str)
    return row
