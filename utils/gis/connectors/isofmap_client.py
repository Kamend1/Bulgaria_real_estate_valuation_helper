"""
Client for isofmap.bg — GIS-Sofia's free public property/zoning viewer.

VERIFIED LIVE — this is the real source of Sofia's ОУП (General Master
Plan) zoning parameters (Устройствена зона / Плътност на застрояване /
КИНТ / Мин. озеленена площ), reverse-engineered by walking the site's own
client-side JS (no API docs exist for this):

  1. Backend is UMN MapServer 8.0, exposed as WMS at `isofmap.bg/owsmap`
     — `mapUrl()` in the site's `js/mapUrl.js`.
  2. The map's own "info" click control (`Info.prototype.getFeatureInfo`
     in `js/olCustomControls.js`) does nothing more exotic than a
     standard OGC `GetFeatureInfo` request per visible+queryable layer —
     `INFO_FORMAT=text/html`, small pixel-radius query window around the
     click point, native CRS **EPSG:7801** (BGS2005).
  3. Two WMS quirks specific to this MapServer instance (found by
     iterating on ServiceExceptions, not documented anywhere):
       - `STYLES=` is a *required* parameter even when empty — omitting
         it fails with `MissingParameterValue`.
       - The HTML response is genuine UTF-8 bytes, but `requests` (and
         evidently browsers relying on the HTTP header) can't tell,
         since the server sends no charset in `Content-Type` — must
         decode `response.content` as UTF-8 explicitly, not `response.text`.
  4. The specific zoning-parameters layer is **`gdp_close_2010`**
     ("Урбанизирани територии ... (ОУП)") — a *child* of the "Общ
     устройствен план" > "ОУП на СО (Решение №960 от 16.12.2009г МС)"
     layer group. The group's own top-level WMS name is NOT itself
     queryable (MapServer error: "Nothing specified in DATA statement" —
     it's a pure UI grouping node); GetFeatureInfo has to target the
     specific leaf layer. Verified against a real Sofia parcel
     (68134.905.1462, Лозенец): returned zone code "Смф", density 60%,
     Kint 3.5, min. landscaping 40% — an exact field-for-field match
     against a table the user obtained by hand via the real map UI.

Coverage caveat: `gdp_close_2010` only covers "urbanized territory"
within the 2009 ОУП's mapped extent. A point outside that (confirmed:
returns an empty GetFeatureInfo response, not an error) genuinely has no
ОУП zoning parameters in this system — not a bug, a real "not covered"
answer (e.g. some parcels sit on landscaping/green-belt sub-zones with
0% density and blank Kint, which is itself the correct legal answer for
that specific area, not missing data).
"""
from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.models.schemas import SourceMeta, ZoningInfo
from utils.gis.spatial_engine.geometry_ops import reproject_point

logger = logging.getLogger(__name__)

ISOFMAP_BASE_URL = "http://isofmap.bg"
OWSMAP_URL = f"{ISOFMAP_BASE_URL}/owsmap"
ZONING_LAYER = "gdp_close_2010"
NATIVE_CRS = "EPSG:7801"  # BGS2005 — what isofmap.bg's WMS actually serves
REQUEST_TIMEOUT_S = 20

# Query window: a small pixel radius around the point, matching what the
# site's own click-to-identify control uses (RADIUS=10, feature_count=30).
_QUERY_RESOLUTION_M = 0.5  # map units (metres) per pixel — a tight, "zoomed in" query
_QUERY_IMAGE_SIZE = 101  # odd, so the click point lands exactly on the centre pixel


class IsofmapError(RuntimeError):
    """Raised on network failure or an unexpected response shape. An empty
    (no feature found) result is NOT an error — see module docstring's
    coverage caveat — it's returned as ZoningInfo(confidence="not_found")."""


def _decimal(value: str) -> float | None:
    """isofmap.bg writes decimals with a comma (e.g. "3,5") — Bulgarian
    numeric convention, not a formatting bug."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _query_layer_at_point(lat: float, lon: float, src_crs: str, cache: GisCache | None) -> str:
    x, y = reproject_point(lat, lon, src_crs, NATIVE_CRS)
    half = (_QUERY_IMAGE_SIZE // 2) * _QUERY_RESOLUTION_M
    bbox = f"{x - half},{y - half},{x + half},{y + half}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": ZONING_LAYER,
        "STYLES": "",  # required by this MapServer instance even when empty
        "QUERY_LAYERS": ZONING_LAYER,
        "SRS": NATIVE_CRS,
        "BBOX": bbox,
        "WIDTH": _QUERY_IMAGE_SIZE,
        "HEIGHT": _QUERY_IMAGE_SIZE,
        "X": _QUERY_IMAGE_SIZE // 2,
        "Y": _QUERY_IMAGE_SIZE // 2,
        "INFO_FORMAT": "text/html",
        "RADIUS": 10,
        "FEATURE_COUNT": 1,
    }

    if cache is not None:
        cached = cache.get("isofmap", OWSMAP_URL, params)
        if cached is not None:
            return cached["html"]

    try:
        resp = requests.get(OWSMAP_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise IsofmapError(f"isofmap.bg WMS request failed: {exc}") from exc

    if "vnd.ogc.se_xml" in resp.headers.get("Content-Type", ""):
        raise IsofmapError(f"isofmap.bg returned a ServiceException: {resp.content.decode('utf-8', errors='replace')[:300]}")

    html = resp.content.decode("utf-8", errors="replace")  # see module docstring — mislabeled charset
    if cache is not None:
        cache.set("isofmap", OWSMAP_URL, params, {"html": html})
    return html


def _parse_zoning_html(html: str) -> dict[str, str]:
    """Extracts the Атрибут/Стойност rows into a plain dict. Rows whose
    visible cell is just an expand-button (Описание на .../Група
    устройствени зони's long-text popups) are read from the button's
    `data-*` attribute instead, since that's where the real text lives.
    One row (the zone's full name, e.g. "Смесена многофункционална
    зона") has no label of its own — it's stored as
    "_categoryFullName", not merged into "Устройствена категория", so
    that field stays a clean short code (e.g. "Смф")."""
    soup = BeautifulSoup(html, "html.parser")
    fields: dict[str, str] = {}
    current_label = None

    for row in soup.select("table tbody tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue
        label = cells[0].get_text(strip=True)
        value_cell = cells[1]

        button = value_cell.find(attrs={"data-territory": True}) or value_cell.find(attrs={"data-groupzone": True})
        if button is not None:
            value = button.get("data-territory") or button.get("data-groupzone") or ""
        else:
            value = value_cell.get_text(strip=True)

        if label:
            fields[label] = value
            current_label = label
        elif current_label == "Устройствена категория" and value:
            fields["_categoryFullName"] = value
        elif current_label and value:
            fields[current_label] = f"{fields.get(current_label, '')} — {value}".strip(" —")

    return fields


def get_zoning_at_point(
    lat: float, lon: float, src_crs: str = "EPSG:4258", cache: GisCache | None = None
) -> ZoningInfo:
    """Looks up ОУП zoning parameters for a point. `src_crs` defaults to
    EPSG:4258 since that's what AGKK centroids arrive in — pass the
    correct one if calling with coordinates from elsewhere."""
    html = _query_layer_at_point(lat, lon, src_crs, cache)
    source = SourceMeta(
        name="isofmap.bg ОУП (gdp_close_2010)",
        endpoint=OWSMAP_URL,
        status="ok" if html.strip() else "not_found",
    )

    if not html.strip():
        return ZoningInfo(confidence="not_found", source=source)

    fields = _parse_zoning_html(html)
    if not fields:
        return ZoningInfo(confidence="not_found", source=source)

    zone_code = fields.get("Устройствена категория") or None
    category_full_name = fields.get("_categoryFullName")
    purpose = fields.get("Предназначение")
    zone_description = " — ".join(p for p in [category_full_name, purpose] if p) or fields.get(
        "Група устройствена зона"
    )
    plan_name_parts = [p for p in [fields.get("Административен район"), "ОУП на София (2009)"] if p]

    return ZoningInfo(
        zone_code=zone_code,
        zone_description=zone_description,
        max_density_pct=_decimal(fields.get("Плътност на застрояване", "")),
        max_kint=_decimal(fields.get("КИНТ", "")),
        max_height_m=_decimal(re.sub(r"[^\d,\.]", "", fields.get("Кота корниз", ""))),
        min_landscaping_pct=_decimal(fields.get("Мин. озеленена площ", "")),
        plan_name=", ".join(plan_name_parts) or None,
        confidence="exact_match",
        source=source,
    )
