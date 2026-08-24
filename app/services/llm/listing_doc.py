"""Serializes a Listing (or an AppraisalReport's subject fields) into the
natural-language text that gets embedded (Phase 7).

Structured/filterable facts (property type, geo category, area band, deal
type, status) stay in real DB columns and drive retriever.py's SQL
pre-filter -- they are NOT what this text is for. This text captures the
qualitative, "reads like the same kind of property" signal an embedding
model is actually good at: condition/quality language, feature overlap,
neighborhood phrasing, overall gestalt. Every column is still represented
here (as prose, not a raw concatenation) so nothing from the listing is
invisible to the embedding -- see the plan doc's "representation /
embedding / retrieval" section for the full rationale.

subject_to_text() must produce the same *shape* of text as listing_to_text()
for retrieval to be meaningful (retriever.py embeds the subject with the
same model as the corpus and compares by cosine distance).
"""
from __future__ import annotations

from app.db.models import AppraisalReport, Listing


def _join_sentence(parts: list[str | None]) -> str | None:
    """Join non-empty parts into one sentence, dropping any part whose text
    is already substring-contained (case-insensitively) in another part --
    title_city_model/title_geo_2_model often repeat verbatim inside the raw
    location_raw string -- and stripping any trailing period each part
    might already carry (e.g. "г." for a construction year) so the sentence
    ends with exactly one."""
    bits: list[str] = []
    for p in parts:
        if not p or not p.strip():
            continue
        candidate = p.strip().rstrip(".")
        c_lower = candidate.lower()
        if any(c_lower in b.lower() for b in bits):
            continue  # fully covered by an already-accepted, longer bit
        bits = [b for b in bits if b.lower() not in c_lower]  # this bit subsumes a shorter one
        bits.append(candidate)
    return ", ".join(bits) + "." if bits else None


def listing_to_text(listing: Listing) -> str:
    """Build the semantic text blob for one Listing row."""
    lines: list[str] = []

    deal = {"sale": "Продажба", "rent": "Дава под наем"}.get(listing.deal_type_normalized or "", "")
    prop_type = listing.property_type_slug or ""
    header = f"{deal}, {prop_type}.".strip(", .").strip()
    if header:
        lines.append(header + ".")

    loc = _join_sentence([listing.title_city_model, listing.title_geo_2_model, listing.location_raw])
    if loc:
        lines.append(loc)

    construction = _join_sentence([
        listing.construction_type_model,
        f"строена {listing.construction_year_model} г." if listing.construction_year_model else None,
    ])
    if construction:
        lines.append(construction)

    if listing.floor_model is not None:
        floor_txt = f"Етаж {listing.floor_model}"
        if listing.total_floors_model:
            floor_txt += f" от {listing.total_floors_model}"
        lines.append(floor_txt + ".")

    if listing.area_sqm_model:
        lines.append(f"Площ {listing.area_sqm_model:g} кв.м.")

    if listing.total_price and listing.price_per_sqm_model:
        currency = listing.currency or "EUR"
        lines.append(
            f"Цена {listing.total_price:.0f} {currency} "
            f"({listing.price_per_sqm_model:.0f} {currency}/кв.м)."
        )

    if listing.features_pipe:
        lines.append(f"Особености: {listing.features_pipe.replace('|', ', ')}.")

    if listing.description_clean:
        lines.append(listing.description_clean.strip())

    return "\n".join(lines)


def subject_to_text(report: AppraisalReport) -> str:
    """Build the same-shape semantic text blob for an AppraisalReport's
    subject property (used to embed the query side of retrieval)."""
    lines: list[str] = []

    prop_type = report.subject_property_type or ""
    lines.append(f"Продажба, {prop_type}.".strip(", .") + ".")

    loc = _join_sentence([report.subject_city, report.subject_neighborhood, report.subject_address])
    if loc:
        lines.append(loc)

    construction = _join_sentence([
        report.subject_construction,
        f"строена {report.subject_year} г." if report.subject_year else None,
    ])
    if construction:
        lines.append(construction)

    if report.subject_floor is not None:
        floor_txt = f"Етаж {report.subject_floor}"
        if report.subject_total_floors:
            floor_txt += f" от {report.subject_total_floors}"
        lines.append(floor_txt + ".")

    if report.subject_area_sqm:
        lines.append(f"Площ {report.subject_area_sqm:g} кв.м.")

    if report.subject_description:
        lines.append(report.subject_description.strip())

    return "\n".join(lines)
