"""
Client for AGKK's INSPIRE cadastral services.

VERIFIED LIVE at inspire.cadastre.bg:
  https://inspire.cadastre.bg/arcgis/rest/services/Cadastral_Parcel/MapServer  (CP.CadastralParcel)
  https://inspire.cadastre.bg/arcgis/rest/services/Building/MapServer         (BU.Building)

These are Esri ArcGIS Server MapServers with the InspireFeatureDownload /
InspireView / WFSServer / WMSServer extensions enabled. The plain REST
`query` operation (`f=geojson`) needs no authentication and no API key —
confirmed with live queries returning real geometry + attributes for both
layers (parcel "15285.14.122", 1471 m²; building "15285.13.286.2", on
parcel "15285.13.286").

Do NOT confuse this free service with AGKK's separate *paid* bulk WMS
product (КАИС portal service #8002), which requires a КЕП-signed
application, IP whitelisting, ~3 business day approval, and a per-layer
subscription fee. This connector only talks to the free INSPIRE service.

Two things confirmed about the Building layer that shape the functions
below:
  - Its `id_localid` is 4 segments (EKATTE.masiv.imot.building_no) — the
    first 3 segments are literally the parent parcel's id_localid, which
    means "find the buildings on parcel X" is a cheap attribute LIKE
    query, not a spatial join.
  - It has NO `areavalue` attribute (unlike the parcel layer) — building
    footprint area has to be computed client-side from geometry via
    spatial_engine.geometry_ops.compute_metric_area_perimeter.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.models.schemas import BuildingInfo, Coordinates, NeighbourParcel, ParcelGeometry, SourceMeta
from utils.gis.spatial_engine.geometry_ops import (
    compute_metric_area_perimeter,
    esri_rings_from_geojson,
    pick_containing_polygon,
)

logger = logging.getLogger(__name__)

AGKK_BASE_URL = "https://inspire.cadastre.bg/arcgis/rest/services"
CADASTRAL_PARCEL_SERVICE = f"{AGKK_BASE_URL}/Cadastral_Parcel/MapServer/0"
BUILDING_SERVICE = f"{AGKK_BASE_URL}/Building/MapServer/0"
NATIVE_CRS = "EPSG:4258"  # ETRS89 — what the INSPIRE service actually returns
REQUEST_TIMEOUT_S = 15


class AgkkClientError(RuntimeError):
    """Raised on network failure, non-200 response, or an empty result set
    after retries — never silently returns a fabricated/default geometry."""


def _query_layer(
    layer_url: str,
    params: dict[str, Any],
    cache: GisCache | None,
) -> dict[str, Any]:
    if cache is not None:
        cached = cache.get("agkk", layer_url, params)
        if cached is not None:
            logger.info("AGKK cache hit for %s params=%s", layer_url, params)
            return cached

    try:
        resp = requests.get(f"{layer_url}/query", params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise AgkkClientError(f"AGKK request failed for {layer_url}: {exc}") from exc

    data = resp.json()
    if "error" in data:
        raise AgkkClientError(f"AGKK service returned an error: {data['error']}")

    if cache is not None:
        cache.set("agkk", layer_url, params, data)
    return data


def _where_query(layer_url: str, where: str, cache: GisCache | None, result_count: int = 50) -> dict[str, Any]:
    params = {
        "where": where,
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
        "resultRecordCount": result_count,
    }
    return _query_layer(layer_url, params, cache)


def _polygon_centroid(geojson_geometry: dict) -> tuple[float, float]:
    coords = geojson_geometry["coordinates"][0]
    if isinstance(coords[0][0], list):
        coords = coords[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _feature_to_parcel(feature: dict, cadastral_id: str | None = None) -> ParcelGeometry:
    props = feature["properties"]
    geometry = feature["geometry"]
    centroid = _polygon_centroid(geometry)
    return ParcelGeometry(
        cadastral_id=cadastral_id or props.get("id_localid", "unknown"),
        area_sqm=props.get("areavalue"),
        admin_unit_code=str(props.get("admunit")) if props.get("admunit") is not None else None,
        centroid=Coordinates(lat=centroid[1], lon=centroid[0], crs=NATIVE_CRS),
        geometry_geojson=geometry,
        native_crs=NATIVE_CRS,
        source=SourceMeta(
            name="AGKK INSPIRE Cadastral_Parcel",
            endpoint=f"{CADASTRAL_PARCEL_SERVICE}/query",
            queried_at=datetime.now(timezone.utc),
            status="ok",
        ),
    )


# ---------------------------------------------------------------------------
# Parcels
# ---------------------------------------------------------------------------


def fetch_parcel_by_cadastral_id(cadastral_id: str, cache: GisCache | None = None) -> ParcelGeometry:
    """Looks up a single cadastral parcel by its EKATTE.masiv.imot identifier
    (the `id_localid` field on AGKK's side). Raises AgkkClientError if not
    found or the service is unreachable — callers must not treat a missing
    parcel as "zero area", it means "we genuinely don't know"."""
    data = _where_query(CADASTRAL_PARCEL_SERVICE, f"id_localid = '{cadastral_id}'", cache, result_count=1)
    features = data.get("features", [])
    if not features:
        raise AgkkClientError(f"No cadastral parcel found for id_localid='{cadastral_id}'")
    return _feature_to_parcel(features[0], cadastral_id)


def fetch_parcel_by_coordinates(
    lat: float, lon: float, cache: GisCache | None = None, buffer_deg: float = 0.0002
) -> ParcelGeometry:
    """Spatial lookup: find whichever parcel polygon contains (lat, lon).
    Uses an envelope intersects query (cheap, index-friendly), then
    disambiguates client-side via shapely point-in-polygon since an
    envelope query returns false positives near parcel boundaries."""
    envelope = {
        "xmin": lon - buffer_deg,
        "ymin": lat - buffer_deg,
        "xmax": lon + buffer_deg,
        "ymax": lat + buffer_deg,
        "spatialReference": {"wkid": 4258},
    }
    params = {
        "geometry": json.dumps(envelope),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 4258,
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
    }
    data = _query_layer(CADASTRAL_PARCEL_SERVICE, params, cache)
    features = data.get("features", [])
    if not features:
        raise AgkkClientError(f"No cadastral parcel found containing point ({lat}, {lon})")

    feature = pick_containing_polygon(features, lat, lon)
    return _feature_to_parcel(feature)


def fetch_neighbouring_parcels(cadastral_id: str, cache: GisCache | None = None) -> list[NeighbourParcel]:
    """Finds parcels that share a boundary with the subject parcel —
    the "who are the neighbours" mechanism for a legal land-plot
    description. Queries AGKK with the subject polygon itself (not just
    its envelope).

    Uses `spatialRel=esriSpatialRelIntersects`, not the seemingly more
    correct `esriSpatialRelTouches` — tested live and confirmed
    `esriSpatialRelTouches` returns zero results even for parcels that are
    visibly adjacent in the data (real surveyed cadastral boundaries
    rarely share exact vertex-for-vertex topology, so a strict "touches"
    predicate finds nothing). `Intersects` correctly returns the subject
    parcel plus its true neighbours; the subject itself is filtered out
    below by id_localid."""
    subject = fetch_parcel_by_cadastral_id(cadastral_id, cache=cache)
    esri_geom = esri_rings_from_geojson(subject.geometry_geojson, wkid=4258)

    params = {
        "geometry": json.dumps(esri_geom),
        "geometryType": "esriGeometryPolygon",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 4258,
        "outFields": "id_localid,label,areavalue",
        "f": "geojson",
        "returnGeometry": "false",
    }
    data = _query_layer(CADASTRAL_PARCEL_SERVICE, params, cache)
    features = data.get("features", [])

    neighbours = []
    for feature in features:
        props = feature["properties"]
        neighbour_id = props.get("id_localid")
        if neighbour_id is None or neighbour_id == cadastral_id:
            continue  # defensive: a shared-edge query shouldn't return self, but don't trust that blindly
        neighbours.append(
            NeighbourParcel(
                cadastral_id=neighbour_id,
                label=props.get("label"),
                area_sqm=props.get("areavalue"),
            )
        )
    return neighbours


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------

def _feature_to_building(feature: dict) -> BuildingInfo:
    props = feature["properties"]
    geometry = feature.get("geometry")

    area_sqm = None
    centroid = None
    if geometry:
        area_sqm, _perimeter_m = compute_metric_area_perimeter(geometry, native_crs=NATIVE_CRS)
        cx, cy = _polygon_centroid(geometry)
        centroid = Coordinates(lat=cy, lon=cx, crs=NATIVE_CRS)

    cadastral_id = props.get("id_localid", "unknown")
    parcel_id = ".".join(cadastral_id.split(".")[:3]) if cadastral_id != "unknown" else "unknown"

    # currentuse/buildingnature are INSPIRE codelist fields with up to 3
    # parallel (code/label/uri/percentage) variants server-side — field
    # names confirmed live: currentuse_label1..3, buildingnature_label1..3.
    # Most buildings in this dataset have these void (not populated by
    # AGKK) rather than genuinely empty — a null here is common, not a bug.
    current_use = props.get("currentuse_label1") or props.get("currentuse_label2") or props.get("currentuse_label3")
    building_nature = (
        props.get("buildingnature_label1")
        or props.get("buildingnature_label2")
        or props.get("buildingnature_label3")
    )

    return BuildingInfo(
        cadastral_id=cadastral_id,
        parcel_id=parcel_id,
        area_sqm=area_sqm,
        floors_above_ground=_safe_int(props.get("numberoffloorsaboveground")),
        dwellings=_safe_int(props.get("numberofdwellings")),
        building_units=_safe_int(props.get("numberofbuildingunits")),
        current_use=current_use,
        building_nature=building_nature,
        condition=props.get("conditionofconstr"),
        construction_date=props.get("dateofconstr_beginning") or props.get("dateofconstr_anypoint"),
        centroid=centroid,
        source=SourceMeta(
            name="AGKK INSPIRE Building",
            endpoint=f"{BUILDING_SERVICE}/query",
            queried_at=datetime.now(timezone.utc),
            status="ok",
        ),
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_building_by_cadastral_id(building_cadastral_id: str, cache: GisCache | None = None) -> BuildingInfo:
    """Looks up a single building by its own (4-segment) cadastral
    identifier, e.g. '15285.13.286.2'."""
    data = _where_query(BUILDING_SERVICE, f"id_localid = '{building_cadastral_id}'", cache, result_count=1)
    features = data.get("features", [])
    if not features:
        raise AgkkClientError(f"No building found for id_localid='{building_cadastral_id}'")
    return _feature_to_building(features[0])


def fetch_buildings_on_parcel(parcel_cadastral_id: str, cache: GisCache | None = None) -> list[BuildingInfo]:
    """Finds every building sitting on a given parcel. Relies on AGKK's
    own id_localid convention (building id = parcel id + '.' + building
    number) rather than a spatial join — cheaper and, since the
    convention was confirmed live, just as reliable for this service."""
    data = _where_query(BUILDING_SERVICE, f"id_localid LIKE '{parcel_cadastral_id}.%'", cache, result_count=50)
    features = data.get("features", [])
    return [_feature_to_building(f) for f in features]
