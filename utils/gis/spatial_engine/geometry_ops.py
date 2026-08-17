"""
CRS transforms and spatial predicates shared by the connectors/engines.

Bulgaria-relevant CRSes used across this pipeline:
  EPSG:4258  ETRS89 geographic — what AGKK's INSPIRE service returns natively
  EPSG:4326  WGS84 geographic — what GPS/consumer map inputs use (~1m offset
             from 4258 in practice; the two are frequently conflated but a
             correct pipeline should still transform explicitly, not assume
             they're interchangeable)
  EPSG:7801  BGS2005 / CCS2005 — Bulgaria's official national projected CRS,
             what municipal cadastral/zoning layers are commonly digitized in
  EPSG:32635 WGS84 / UTM zone 35N — metric CRS used here for area/perimeter
             math (AGKK's own geometries are in decimal degrees, so any
             "area in m^2" for a layer that doesn't ship its own area
             attribute — e.g. Building — has to be computed after
             reprojecting into something metric first)
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyproj is required for CRS transforms — see requirements.txt. "
        "On Windows, prefer conda-forge over pip for reliable PROJ data "
        "bundling if plain pip install gives you PROJ errors."
    ) from exc

_TRANSFORMER_CACHE: dict[tuple[str, str], Transformer] = {}


def get_transformer(src_epsg: str, dst_epsg: str) -> Transformer:
    key = (src_epsg, dst_epsg)
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return _TRANSFORMER_CACHE[key]


def reproject_geometry(geom: BaseGeometry, src_epsg: str, dst_epsg: str) -> BaseGeometry:
    if src_epsg == dst_epsg:
        return geom
    transformer = get_transformer(src_epsg, dst_epsg)
    from shapely.ops import transform as shapely_transform

    return shapely_transform(transformer.transform, geom)


def reproject_point(lat: float, lon: float, src_epsg: str, dst_epsg: str) -> tuple[float, float]:
    transformer = get_transformer(src_epsg, dst_epsg)
    x, y = transformer.transform(lon, lat)  # always_xy=True -> (lon, lat) order in
    return x, y  # (lon, lat) order out too


def pick_containing_polygon(geojson_features: list[dict], lat: float, lon: float) -> dict:
    """Given several candidate GeoJSON features from an envelope/bbox
    query, return the one whose polygon actually contains the point —
    disambiguates the false positives an envelope query returns near
    parcel boundaries. Falls back to nearest centroid if none contain it
    exactly (coordinate-precision edge case, e.g. point on a shared edge)."""
    point = Point(lon, lat)
    for feature in geojson_features:
        polygon = shape(feature["geometry"])
        if polygon.contains(point):
            return feature

    nearest = min(
        geojson_features,
        key=lambda f: shape(f["geometry"]).centroid.distance(point),
    )
    return nearest


def compute_metric_area_perimeter(
    geojson_geometry: dict, native_crs: str = "EPSG:4258", metric_crs: str = "EPSG:32635"
) -> tuple[float, float]:
    """Reprojects a geometry into a metric CRS and returns (area_sqm,
    perimeter_m). Use this for anything that doesn't ship AGKK's own
    `areavalue` attribute — the Building layer, for instance."""
    geom = shape(geojson_geometry)
    metric_geom = reproject_geometry(geom, native_crs, metric_crs)
    return metric_geom.area, metric_geom.length


def esri_rings_from_geojson(geojson_geometry: dict, wkid: int = 4258) -> dict[str, Any]:
    """Converts a GeoJSON Polygon/MultiPolygon geometry into Esri JSON
    geometry form (`{"rings": [...], "spatialReference": {...}}`) for use
    as a `geometry` query parameter against an ArcGIS REST `query`
    operation. GeoJSON and Esri JSON both represent a polygon as a list of
    coordinate rings, so this is a near-direct passthrough — the one real
    difference (ring winding order convention) doesn't matter for a
    *query* geometry, only for a geometry you intend to persist/edit
    server-side, so it's left as-is here."""
    coords = geojson_geometry["coordinates"]
    if geojson_geometry["type"] == "Polygon":
        rings = coords
    elif geojson_geometry["type"] == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
    else:
        raise ValueError(f"Unsupported geometry type for Esri rings conversion: {geojson_geometry['type']}")
    return {"rings": rings, "spatialReference": {"wkid": wkid}}
