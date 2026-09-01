"""
Lightweight reachability check for the GIS/cadastre connectors actually wired
into gis_service.py (AGKK/inspire.cadastre.bg, isofmap.bg, nag.sofia.bg) --
Phase 14 Tier 1.2. municipal_zoning_client/ckan_client are research-only
tools (see docs/gis_cadastral/PLAN.md), not called from the live app, so
they're intentionally not part of this check.

This never raises -- a connector being down is a normal, expected runtime
condition (these are third-party government sites with no uptime guarantee),
not an application error. Each check is a single short-timeout GET, not a
full functional probe -- it answers "is the endpoint reachable at all right
now", not "would a real query succeed."
"""
from __future__ import annotations

import time

import requests

from utils.gis.connectors.agkk_client import CADASTRAL_PARCEL_SERVICE
from utils.gis.connectors.isofmap_client import ISOFMAP_BASE_URL
from utils.gis.connectors.nag_sofia_client import NAG_BASE_URL

HEALTHCHECK_TIMEOUT_S = 5

_CONNECTORS = [
    {"name": "АГКК (кадастър)", "url": CADASTRAL_PARCEL_SERVICE, "params": {"f": "pjson"}},
    {"name": "isofmap.bg (устройствена зона)", "url": ISOFMAP_BASE_URL, "params": None},
    {"name": "НАГ София (устройствени планове)", "url": NAG_BASE_URL, "params": None},
]


def check_gis_connectors() -> list[dict]:
    results = []
    for connector in _CONNECTORS:
        started = time.monotonic()
        try:
            resp = requests.get(connector["url"], params=connector["params"], timeout=HEALTHCHECK_TIMEOUT_S)
            latency_ms = round((time.monotonic() - started) * 1000)
            ok = resp.status_code < 500
            results.append({
                "name": connector["name"], "url": connector["url"], "ok": ok,
                "status_code": resp.status_code, "latency_ms": latency_ms, "error": None,
            })
        except requests.RequestException as exc:
            latency_ms = round((time.monotonic() - started) * 1000)
            results.append({
                "name": connector["name"], "url": connector["url"], "ok": False,
                "status_code": None, "latency_ms": latency_ms, "error": str(exc),
            })
    return results
