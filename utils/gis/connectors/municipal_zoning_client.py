"""
Generic Esri ArcGIS REST client for municipal zoning (ОУП/УПИ) layers,
driven by config.municipalities.MUNICIPALITY_REGISTRY.

Deliberately does NOT hardcode a working, fully-populated example for any
one city — see the module docstring in config/municipalities.py for why.
What this module gives you instead:

  1. `discover_services(base_url)` — point it at any ArcGIS Server root
     and it lists every folder/service found, so you can locate the right
     zoning layer without guessing.
  2. `inspect_layer_schema(service_root, layer_id)` — dumps the real field
     names for a layer once you've found it, so you can fill in
     `field_mapping` in the registry correctly.
  3. `query_zone_at_point(config, lat, lon)` — once a registry entry is
     verified + mapped, does the actual point-in-polygon zoning lookup.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.config.municipalities import MunicipalZoningConfig
from utils.gis.models.schemas import SourceMeta, ZoningInfo

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_S = 15


class MunicipalZoningError(RuntimeError):
    pass


def discover_services(base_url: str) -> dict[str, Any]:
    """GET <base_url>?f=pjson — lists folders + services. Use this
    interactively (e.g. in a notebook) to find the right service name
    before wiring up a MunicipalZoningConfig."""
    resp = requests.get(base_url, params={"f": "pjson"}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def inspect_layer_schema(service_root: str, layer_id: int) -> dict[str, Any]:
    """GET <service_root>/<layer_id>?f=pjson — returns the field list
    (names + types + aliases) for a specific layer."""
    url = f"{service_root}/{layer_id}"
    resp = requests.get(url, params={"f": "pjson"}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def query_zone_at_point(
    config: MunicipalZoningConfig, lat: float, lon: float, cache: GisCache | None = None
) -> ZoningInfo:
    if not config.verified_live:
        logger.warning(
            "Querying '%s' zoning service, which is marked verified_live=False "
            "(%s). Treat the result as unreliable until you confirm the "
            "endpoint and field_mapping yourself.",
            config.municipality,
            config.notes,
        )
    if not config.field_mapping:
        raise MunicipalZoningError(
            f"No field_mapping configured for '{config.municipality}' — run "
            "inspect_layer_schema() and populate config/municipalities.py "
            "before calling this."
        )

    url = f"{config.service_root}/{config.layer_id}/query"
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 4326,
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "false",
    }

    if cache is not None:
        cached = cache.get("municipal_zoning", url, params)
        data = cached if cached is not None else _fetch(url, params, cache)
    else:
        data = _fetch(url, params, None)

    features = data.get("features", [])
    source = SourceMeta(
        name=f"{config.municipality} municipal zoning",
        endpoint=url,
        queried_at=datetime.now(timezone.utc),
        status="ok" if features else "not_found",
    )

    if not features:
        return ZoningInfo(confidence="not_found", source=source)

    props = features[0]["properties"]
    fm = config.field_mapping
    return ZoningInfo(
        zone_code=props.get(fm.get("zone_code", "")),
        zone_description=props.get(fm.get("zone_description", "")),
        max_density_pct=_safe_float(props.get(fm.get("max_density_pct", ""))),
        max_kint=_safe_float(props.get(fm.get("max_kint", ""))),
        max_height_m=_safe_float(props.get(fm.get("max_height_m", ""))),
        min_landscaping_pct=_safe_float(props.get(fm.get("min_landscaping_pct", ""))),
        plan_name=props.get(fm.get("plan_name", "")),
        confidence="exact_match" if len(features) == 1 else "nearest_zone",
        source=source,
    )


def _fetch(url: str, params: dict, cache: GisCache | None) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise MunicipalZoningError(f"Municipal zoning request failed for {url}: {exc}") from exc
    data = resp.json()
    if cache is not None:
        cache.set("municipal_zoning", url, params, data)
    return data


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
