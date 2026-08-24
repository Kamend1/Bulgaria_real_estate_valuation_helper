"""Retrieves the k semantically-nearest comparable listings for a report's
subject property (Phase 7).

Hybrid retrieval: a structured SQL pre-filter (same condition-fragment
style and bound-params discipline as listing_service.search_listings --
see its SECURITY INVARIANT docstring) narrows to *eligible* candidates,
then pgvector's cosine-distance operator ranks *within* that set by
similarity to the subject's own serialized text. Exact facts gate
eligibility; the embedding only adjudicates ranking. See the plan doc's
"representation / embedding / retrieval" section for the full rationale.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport
from app.services.llm.embeddings import get_embeddings_model, resolve_embedding_model
from app.services.llm.listing_doc import subject_to_text

MIN_CANDIDATES = 3
AREA_BAND_LOW = 0.6   # subject_area * this .. subject_area * AREA_BAND_HIGH
AREA_BAND_HIGH = 1.4

_SELECT_COLUMNS = """
    l.id, l.ad_url, l.title_city_model, l.title_geo_2_model, l.location_raw,
    l.area_sqm_model, l.total_price, l.currency, l.price_per_sqm_model,
    l.construction_type_model, l.construction_year_model,
    l.floor_model, l.total_floors_model,
    e.embedding <=> CAST(:subject_vec AS vector) AS distance
"""
# NOTE: `:subject_vec::vector` (Postgres's `::` cast shorthand touching the
# bind param name) silently confuses SQLAlchemy's text() bind-parameter
# parser -- every other :name param above gets substituted correctly, but
# that one is left as a literal ":subject_vec::vector" and Postgres then
# fails on the bare ":". CAST(:subject_vec AS vector) avoids the ambiguity.


def _filter_stages(has_ptype: bool, has_geo: bool, has_area: bool) -> list[list[str]]:
    """Progressively relaxed structured pre-filters, tried in order until
    one yields >= MIN_CANDIDATES rows (the last stage is used regardless of
    how few rows it returns -- an empty result is a valid, meaningful
    answer, not an error).

    property_type_slug is a HARD constraint whenever the subject has one --
    it is never relaxed away. An apartment is never a valid comparable for
    an office (or vice versa) regardless of how numerically close its
    embedding lands; only geo_category and the area band are progressively
    loosened. deal_type_normalized ("sale"/"rent") is likewise never
    relaxed -- a rent listing is never a valid comparable for a sale
    approach or vice versa."""
    base = ["l.status = 'active'", "l.deal_type_normalized = :ctype", "l.training_eligible IS TRUE"]
    if has_ptype:
        base = base + ["l.property_type_slug = :ptype"]
    area_cond = ["l.area_sqm_model BETWEEN :area_lo AND :area_hi"] if has_area else []
    geo_cond = ["l.geo_category = :geo_cat"] if has_geo else []

    stages = [base + geo_cond + area_cond]
    if geo_cond:
        stages.append(base + area_cond)
    if area_cond:
        stages.append(base)
    # De-duplicate consecutive identical stages (e.g. when the subject has
    # no geo_category/area to begin with, several stages above collapse).
    deduped: list[list[str]] = []
    for s in stages:
        if not deduped or s != deduped[-1]:
            deduped.append(s)
    return deduped


def retrieve_comparables(
    db: Session, report: AppraisalReport, k: int = 6, comparable_type: str = "sale",
) -> list[dict]:
    """Returns up to k dicts (listing fields + `distance`, nearest first).
    Empty list if the subject has no embeddable listings yet, or nothing
    matches even the most relaxed filter stage.

    comparable_type: "sale" (default) or "rent" -- selects which side of
    the market to retrieve from (Phase 7, Tier 5). subject_to_text() is
    unchanged either way -- it describes the physical property, not the
    deal type, so the same subject embedding is compared against whichever
    corpus slice comparable_type selects."""
    provider, model = resolve_embedding_model()
    embeddings_model = get_embeddings_model(provider, model)
    subject_vec = embeddings_model.embed_query(subject_to_text(report))

    has_ptype = bool(report.subject_property_type)
    has_geo = bool(report.subject_geo_category)
    has_area = report.subject_area_sqm is not None

    params: dict = {
        "provider": provider,
        "model": model,
        "ctype": comparable_type,
        "k": k,
        "subject_vec": "[" + ",".join(repr(float(x)) for x in subject_vec) + "]",
    }
    if has_ptype:
        params["ptype"] = report.subject_property_type
    if has_geo:
        params["geo_cat"] = report.subject_geo_category
    if has_area:
        area = float(report.subject_area_sqm)
        params["area_lo"] = area * AREA_BAND_LOW
        params["area_hi"] = area * AREA_BAND_HIGH

    stages = _filter_stages(has_ptype, has_geo, has_area)
    rows: list[dict] = []
    for i, stage in enumerate(stages):
        where_sql = " AND ".join(stage)
        result = db.execute(text(f"""
            SELECT {_SELECT_COLUMNS}
            FROM listings l
            JOIN listing_embeddings e
              ON e.listing_id = l.id AND e.provider = :provider AND e.model = :model
            WHERE {where_sql}
            ORDER BY distance ASC
            LIMIT :k
        """), params).mappings().all()
        rows = [dict(r) for r in result]
        if len(rows) >= MIN_CANDIDATES or i == len(stages) - 1:
            break
    return rows
