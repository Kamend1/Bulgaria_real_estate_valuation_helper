"""
FastAPI-facing wrapper around utils.gis — resolves an AppraisalReport's
subject_cadastral_id into parcel/building/legal-description data for the
comparables page's "Кадастър" panel.

No DB access needed here (all data comes from AGKK's live service) — this
module only exists to translate between AppraisalReport and utils.gis's
own models/exceptions, the same role avm_service plays for the AVM.
"""
from __future__ import annotations

import logging

from pydantic import ValidationError

from app.db.models import AppraisalReport
from utils.gis.cache.sqlite_cache import get_shared_cache
from utils.gis.connectors.agkk_client import AgkkClientError
from utils.gis.connectors.isofmap_client import IsofmapError, get_zoning_at_point
from utils.gis.connectors.nag_sofia_client import NagSofiaError, search_development_plans
from utils.gis.engines import legal_description_engine, parcel_engine
from utils.gis.models.schemas import CadastralIdentifier, ZoningInfo
from utils.gis.spatial_engine.sketch_svg import render_parcel_svg

logger = logging.getLogger(__name__)


def get_cadastre_panel_data(report: AppraisalReport) -> dict:
    """Returns a dict always containing "ok". On success: parcel, buildings,
    legal (may be None if the legal-description step itself fails after the
    parcel resolved — treated as a soft failure, not a hard one, since the
    parcel/building data is still useful on its own). On failure: "reason"
    is "missing_cadastral_id" | "invalid_format" | "agkk_error" (+ "detail").

    Accepts either a 3-segment parcel id (EKATTE.masiv.imot) or a
    4-segment building id (EKATTE.masiv.imot.building_no) — AGKK's own
    convention is that a building's id is literally its parent parcel's id
    plus a suffix, confirmed live (e.g. building "15285.13.286.2" sits on
    parcel "15285.13.286"). A 4-segment input is therefore resolved to its
    *parent parcel* for the parcel/legal-description lookups; the
    originally-entered building id is returned separately so the UI can
    point out which of the parcel's buildings the user actually meant."""
    if not report.subject_cadastral_id:
        return {"ok": False, "reason": "missing_cadastral_id"}

    try:
        cid = CadastralIdentifier(raw=report.subject_cadastral_id)
    except ValidationError:
        return {"ok": False, "reason": "invalid_format"}

    highlighted_building_id = report.subject_cadastral_id if len(cid.raw.split(".")) == 4 else None

    cache = get_shared_cache()
    try:
        parcel, buildings = parcel_engine.get_parcel_with_buildings(cadastral_id=cid.parcel_id, cache=cache)
    except AgkkClientError as exc:
        logger.warning("AGKK parcel lookup failed for report %s: %s", report.id, exc)
        return {"ok": False, "reason": "agkk_error", "detail": str(exc)}

    settlement_name = report.subject_neighborhood or report.subject_city or None
    legal = None
    try:
        if highlighted_building_id:
            # A specific building was pointed at — the legally relevant
            # subject is the building sitting on the plot, not the bare
            # plot, so lead with the СГРАДА template instead.
            legal = legal_description_engine.generate_building_description(
                highlighted_building_id, cid.parcel_id, settlement_name=settlement_name, cache=cache
            )
        else:
            legal = legal_description_engine.generate_legal_description(
                cid.parcel_id, settlement_name=settlement_name, cache=cache
            )
    except AgkkClientError as exc:
        logger.warning("Legal description generation failed for report %s: %s", report.id, exc)

    total_building_area_sqm = sum(b.area_sqm or 0.0 for b in buildings)
    sketch_svg = render_parcel_svg(parcel.geometry_geojson, parcel.cadastral_id, parcel.area_sqm)

    # NAG Sofia's case registry only ever has data for Sofia parcels, but
    # it's not restricted to a single EKATTE code (Sofia spans several —
    # confirmed live across at least "68134" and "15285" during this
    # project), so there's no cheap reliable prefix check to skip it by.
    # Cheap enough (2 GETs, cached after) to just always try; failure here
    # is soft — the rest of the panel doesn't depend on it.
    development_plans = None
    try:
        development_plans = search_development_plans(cid.parcel_id, cache=cache)
    except NagSofiaError as exc:
        logger.warning("NAG Sofia development-plan search failed for report %s: %s", report.id, exc)

    # isofmap.bg (GIS-Sofia's own public viewer) is the real source of ОУП
    # zoning parameters (Kint/density/landscaping) — see
    # connectors/isofmap_client.py. Like the NAG Sofia search above, this
    # only ever has data for Sofia, but the cheap failure mode (empty
    # result / soft error) means there's no need to gate the call.
    zoning = ZoningInfo(confidence="unavailable")
    try:
        zoning = get_zoning_at_point(parcel.centroid.lat, parcel.centroid.lon, src_crs=parcel.centroid.crs, cache=cache)
    except IsofmapError as exc:
        logger.warning("isofmap.bg zoning lookup failed for report %s: %s", report.id, exc)

    return {
        "ok": True,
        "parcel": parcel,
        "buildings": buildings,
        "highlighted_building_id": highlighted_building_id,
        "total_building_area_sqm": total_building_area_sqm,
        "legal": legal,
        "sketch_svg": sketch_svg,
        "development_plans": development_plans,
        "zoning": zoning,
    }
