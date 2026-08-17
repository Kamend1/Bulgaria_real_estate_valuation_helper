"""
Per-municipality zoning-service configuration registry.

THIS IS THE PART OF THE PIPELINE WITH NO NATIONAL STANDARD. Bulgaria has
~265 municipalities; each publishes (or doesn't publish) its ОУП/ПУП
zoning layers on its own GIS stack, with its own field names, its own
zone-code vocabulary, and its own uptime. There is no CKAN-style single
catalog for this the way there is for cadastral parcels at AGKK. Every
entry below is therefore a *plugin config*, not a guarantee.

Sofia — every endpoint tried so far (gis.sofiaplan.bg/arcgis/rest/services,
nag.sofia.bg/arcgis/rest/services, nag.sofia.bg/arcgis/rest/services/OUP,
gis.sofia.bg) either 404'd or failed DNS resolution during two separate
research passes (2026-08-11 and 2026-08-17). The real public viewer,
nag.sofia.bg/OpenMap/Zones, is a JavaScript single-page app whose backing
API could not be identified by static fetching (its network calls happen
client-side, invisible to a plain HTTP GET of the page). The entry below
is left as a documented template with `verified_live=False` — the
concrete next step to unblock it is opening that map in a real browser,
opening DevTools' Network tab, clicking a zone, and reading the resulting
request URL, since that's the one piece of information static tools in
this environment can't extract.

Plovdiv / Varna: no public ArcGIS/WFS endpoint was located during
research. Left unconfigured rather than guessing a URL.

FIELD_MAPPING translates each municipality's actual attribute field
names to the canonical ZoningInfo schema. Populate this only after you've
inspected the real service schema (MapServer/<layer>?f=pjson) — never
carry over another municipality's field names by assumption, since this
research did not find two Bulgarian municipal GIS stacks with matching
schemas.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MunicipalZoningConfig:
    municipality: str
    service_root: str  # ArcGIS REST MapServer/FeatureServer base URL
    layer_id: int
    native_crs: str
    field_mapping: dict[str, str]  # canonical name -> actual service field name
    verified_live: bool
    notes: str = ""


MUNICIPALITY_REGISTRY: dict[str, MunicipalZoningConfig] = {
    "sofia": MunicipalZoningConfig(
        municipality="Sofia",
        service_root="https://gis.sofiaplan.bg/arcgis/rest/services/oup_2009/MapServer",
        layer_id=0,  # UNVERIFIED — confirm via discover_services() before use
        native_crs="EPSG:7801",  # BGS2005, the CRS Bulgarian municipal GIS commonly uses natively
        field_mapping={
            # left blank deliberately — populate after inspecting the live
            # schema; a fabricated mapping would silently produce wrong
            # zoning numbers, which is worse than no zoning number.
        },
        verified_live=False,
        notes=(
            "No endpoint variant tried across two research passes "
            "(gis.sofiaplan.bg, nag.sofia.bg/arcgis, gis.sofia.bg) resolved "
            "live. The real viewer at nag.sofia.bg/OpenMap/Zones is a JS "
            "SPA — its API needs browser DevTools network inspection to "
            "identify, not static fetching."
        ),
    ),
}
