"""
Parcel-level orchestration: combines the AGKK parcel connector with the
building connector to answer "what's on this plot" as a single call,
instead of making callers stitch fetch_parcel_by_cadastral_id +
fetch_buildings_on_parcel together themselves.
"""
from __future__ import annotations

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.connectors.agkk_client import (
    AgkkClientError,
    fetch_buildings_on_parcel,
    fetch_neighbouring_parcels,
    fetch_parcel_by_cadastral_id,
    fetch_parcel_by_coordinates,
)
from utils.gis.models.schemas import BuildingInfo, NeighbourParcel, ParcelGeometry

__all__ = ["AgkkClientError", "get_parcel_with_buildings", "get_parcel_with_neighbours"]


def get_parcel_with_buildings(
    cadastral_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    cache: GisCache | None = None,
) -> tuple[ParcelGeometry, list[BuildingInfo]]:
    """Resolves a parcel (by ID or by point) and every building AGKK has
    registered on it. Raises AgkkClientError if the parcel itself can't be
    resolved; an empty buildings list is a legitimate answer (undeveloped
    plot), not an error."""
    if cadastral_id:
        parcel = fetch_parcel_by_cadastral_id(cadastral_id, cache=cache)
    elif lat is not None and lon is not None:
        parcel = fetch_parcel_by_coordinates(lat, lon, cache=cache)
    else:
        raise ValueError("Provide either cadastral_id or both lat and lon")

    buildings = fetch_buildings_on_parcel(parcel.cadastral_id, cache=cache)
    return parcel, buildings


def get_parcel_with_neighbours(
    cadastral_id: str, cache: GisCache | None = None
) -> tuple[ParcelGeometry, list[NeighbourParcel]]:
    """Resolves a parcel plus every parcel that shares a boundary with
    it — the raw material for a legal land-plot description's "при
    съседи: ..." clause. See connectors.agkk_client.fetch_neighbouring_parcels
    for the spatial predicate used (esriSpatialRelTouches)."""
    parcel = fetch_parcel_by_cadastral_id(cadastral_id, cache=cache)
    neighbours = fetch_neighbouring_parcels(cadastral_id, cache=cache)
    return parcel, neighbours
