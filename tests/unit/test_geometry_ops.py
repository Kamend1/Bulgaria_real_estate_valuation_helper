"""
Unit tests for utils/gis/spatial_engine/geometry_ops.py — pure
computation, no network I/O. The reproject_point test anchors on a real
(lat, lon) -> (x, y) pair already verified live against AGKK during this
project's GIS development (parcel 68134.905.1462's centroid), not an
independently-derived value.
"""
import pytest

from utils.gis.spatial_engine.geometry_ops import (
    esri_rings_from_geojson,
    pick_containing_polygon,
    reproject_point,
)

_SQUARE_A = {
    "type": "Feature",
    "properties": {"id": "A"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]],
    },
}
_SQUARE_B = {
    "type": "Feature",
    "properties": {"id": "B"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[10, 0], [10, 10], [20, 10], [20, 0], [10, 0]]],
    },
}


def test_pick_containing_polygon_returns_the_true_containing_feature():
    # Both squares' bounding boxes could plausibly show up in a sloppy
    # envelope query near x=10, but only B actually contains (15, 5).
    result = pick_containing_polygon([_SQUARE_A, _SQUARE_B], lat=5, lon=15)
    assert result["properties"]["id"] == "B"


def test_pick_containing_polygon_falls_back_to_nearest_centroid_on_edge():
    # (10, 5) sits exactly on the shared edge — contained by neither
    # polygon under shapely's strict `.contains()` (boundary doesn't
    # count). Should fall back to *a* result, not raise.
    result = pick_containing_polygon([_SQUARE_A, _SQUARE_B], lat=5, lon=10)
    assert result["properties"]["id"] in ("A", "B")


def test_esri_rings_from_geojson_polygon():
    ring = [[23.0, 42.0], [23.0, 42.1], [23.1, 42.1], [23.0, 42.0]]
    result = esri_rings_from_geojson({"type": "Polygon", "coordinates": [ring]}, wkid=4258)
    assert result == {"rings": [ring], "spatialReference": {"wkid": 4258}}


def test_esri_rings_from_geojson_multipolygon_flattens_rings():
    ring1 = [[0, 0], [0, 1], [1, 1], [0, 0]]
    ring2 = [[5, 5], [5, 6], [6, 6], [5, 5]]
    geom = {"type": "MultiPolygon", "coordinates": [[ring1], [ring2]]}
    result = esri_rings_from_geojson(geom)
    assert result["rings"] == [ring1, ring2]


def test_esri_rings_from_geojson_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        esri_rings_from_geojson({"type": "Point", "coordinates": [0, 0]})


def test_reproject_point_epsg4258_to_epsg7801_matches_known_agkk_value():
    # Ground truth captured live during this project's isofmap.bg zoning
    # connector development (parcel 68134.905.1462's AGKK centroid).
    lat, lon = 42.63590506694878, 23.333411800111165
    x, y = reproject_point(lat, lon, "EPSG:4258", "EPSG:7801")
    assert x == pytest.approx(322326.17782368726, abs=0.01)
    assert y == pytest.approx(4724549.985394472, abs=0.01)
