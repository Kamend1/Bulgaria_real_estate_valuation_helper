"""
Client for data.egov.bg — Bulgaria's national open data portal.

VERIFIED: data.egov.bg is a genuine CKAN deployment (standard facets:
organizations/tags/formats/licenses; a documented "API спецификация"
page). This connector targets CKAN's standard Action API
(`/api/3/action/<function>`), publicly readable with no key required for
the read-only actions used here.

What this connector is realistically useful for: discovering *which*
municipal open-data datasets exist for a given topic (e.g. "ОУП Пловдив",
"кадастър Варна") and pulling their resource URLs (often CSV/Shapefile/
GeoJSON downloads, occasionally a WFS/ArcGIS link in the resource
description). It is a *catalog*, not a live zoning-lookup API — feed any
WFS/ArcGIS resource URLs it turns up into municipal_zoning_client.py.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from utils.gis.cache.sqlite_cache import GisCache

logger = logging.getLogger(__name__)

CKAN_BASE_URL = "https://data.egov.bg/api/3/action"
REQUEST_TIMEOUT_S = 15


class CkanClientError(RuntimeError):
    pass


def search_datasets(query: str, rows: int = 20, cache: GisCache | None = None) -> list[dict[str, Any]]:
    """Free-text search over data.egov.bg datasets. Returns CKAN's raw
    package dicts (title, organization, resources[] with url/format)."""
    params = {"q": query, "rows": rows}

    if cache is not None:
        cached = cache.get("ckan", "package_search", params)
        if cached is not None:
            return cached["result"]["results"]

    try:
        resp = requests.get(f"{CKAN_BASE_URL}/package_search", params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise CkanClientError(f"data.egov.bg request failed: {exc}") from exc

    data = resp.json()
    if not data.get("success", False):
        raise CkanClientError(f"data.egov.bg returned success=false: {data.get('error')}")

    if cache is not None:
        cache.set("ckan", "package_search", params, data)
    return data["result"]["results"]


def get_resource_urls(dataset: dict[str, Any], format_filter: str | None = None) -> list[str]:
    """Extracts downloadable resource URLs from a CKAN package dict,
    optionally filtered by format (e.g. 'WFS', 'GeoJSON', 'SHP')."""
    resources = dataset.get("resources", [])
    urls = []
    for r in resources:
        if format_filter and r.get("format", "").upper() != format_filter.upper():
            continue
        if r.get("url"):
            urls.append(r["url"])
    return urls
