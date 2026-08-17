"""
Building-level orchestration. Thin today (the connector already does most
of the work — metric area derivation lives in spatial_engine since other
callers need it too), but kept as its own engine module — mirroring
parcel_engine — as the natural home for building-specific logic that
doesn't belong in the raw HTTP connector: e.g. flagging when a building's
footprint area looks implausible relative to its parent parcel's area, or
aggregating multiple buildings on one parcel into a single "total built-up
area" figure for an appraisal.
"""
from __future__ import annotations

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.connectors.agkk_client import (
    AgkkClientError,
    fetch_building_by_cadastral_id,
    fetch_buildings_on_parcel,
)
from utils.gis.models.schemas import BuildingInfo

__all__ = ["AgkkClientError", "get_building_profile", "total_built_up_area_sqm"]


def get_building_profile(building_cadastral_id: str, cache: GisCache | None = None) -> BuildingInfo:
    """Single-building lookup by its own (4-segment) cadastral id."""
    return fetch_building_by_cadastral_id(building_cadastral_id, cache=cache)


def total_built_up_area_sqm(parcel_cadastral_id: str, cache: GisCache | None = None) -> float:
    """Sum of every building footprint AGKK has registered on a parcel —
    useful as a sanity cross-check against a subject property's declared
    area, or as a basis for a coverage-ratio calc against the parcel's own
    `areavalue`. Returns 0.0 for an undeveloped plot, not None — a
    genuinely absent parcel is a `fetch_parcel_by_cadastral_id` concern,
    not this function's."""
    buildings = fetch_buildings_on_parcel(parcel_cadastral_id, cache=cache)
    return sum(b.area_sqm or 0.0 for b in buildings)
