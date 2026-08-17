"""
Client for Sofia municipality's НАГ (Направление "Архитектура и
градоустройство") development-plan case registry.

VERIFIED LIVE at nag.sofia.bg — but NOT a simple REST endpoint, and NOT
the "give me Kint/density/height at this point" zoning-parameters lookup
this project originally went looking for on nag.sofia.bg (see
docs/gis_cadastral/PLAN.md §1.3 — every straightforward endpoint guess
there 404'd or resolved to the wrong thing). What's here instead: a
searchable registry of ПУП (detailed development plan) applications and
administrative decisions, filterable by cadastral identifier, with real
downloadable case documents (Скица – предложение, Заявление, ИПР, ИПРЗ,
РУП, ...).

Reverse-engineered from the real search page's rendered form/Kendo
config (docs/gis_cadastral/PLAN.md has the walkthrough) — none of it is
documented publicly:

  1. GET  /SearchDevelopmentPlans           — mints a session-bound
     `searchQueryId` GUID, embedded in the page's own markup (the form's
     `action` attribute). Also sets an ASP.NET_SessionId cookie the
     second request must reuse.
  2. GET  /SearchDevelopmentPlans/Search?searchQueryId=<that>&Identifier=<id>
     — despite the HTML `<form method="post">`, the actual submission is
     a GET (`data-ajax-method="GET"` on the form). The filter field is
     named `Identifier`, not `CadastralIdentifier` — the latter silently
     matches nothing and falls back to an unfiltered "10 most recent"
     list, which looks like a working result if you don't check the
     content (confirmed by testing this exact trap).
  3. Document links: each result row's raw "Attachments" cell is a
     `;`-separated list of `name&path` pairs (client-side Kendo template
     splits it that way). The real download URL is
     `/SearchDevelopmentPlans/ViewAttachment?filename=<path>` — confirmed
     live, returns a genuine `application/pdf`.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.models.schemas import DevelopmentPlanDocument, DevelopmentPlanRecord

logger = logging.getLogger(__name__)

NAG_BASE_URL = "https://nag.sofia.bg"
SEARCH_PAGE_URL = f"{NAG_BASE_URL}/SearchDevelopmentPlans"
SEARCH_ACTION_URL = f"{NAG_BASE_URL}/SearchDevelopmentPlans/Search"
ATTACHMENT_URL = f"{NAG_BASE_URL}/SearchDevelopmentPlans/ViewAttachment"
REQUEST_TIMEOUT_S = 20

_SEARCH_QUERY_ID_RE = re.compile(r"/SearchDevelopmentPlans/Search\?searchQueryId=([0-9a-f-]+)")

# Positional column indices confirmed against a live response — the grid
# has more <td> cells than visible header labels (several hidden
# id/office-id columns interleaved for Kendo's own bookkeeping). Verified
# against real rows during development; re-check against a live sample
# if nag.sofia.bg ever changes this grid's column layout.
_COL_DESCRIPTION = 2
_COL_OFFICE = 4
_COL_SCOPE = 5
_COL_PROCEDURE = 6
_COL_ATTACHMENTS = 7
_MIN_EXPECTED_COLS = 8


class NagSofiaError(RuntimeError):
    """Raised on network failure or an unexpected page shape (e.g. the
    search form's structure changed) — never silently returns an empty
    result for a real error, only for a genuine "no matching cases"."""


def _get_search_query_id(session: requests.Session) -> str:
    try:
        resp = session.get(SEARCH_PAGE_URL, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NagSofiaError(f"Failed to load {SEARCH_PAGE_URL}: {exc}") from exc

    match = _SEARCH_QUERY_ID_RE.search(resp.text)
    if not match:
        raise NagSofiaError(
            "Could not find a searchQueryId in the NAG Sofia search page — "
            "the page structure may have changed."
        )
    return match.group(1)


def _parse_results(html: str) -> list[DevelopmentPlanRecord]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[DevelopmentPlanRecord] = []

    for row in soup.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < _MIN_EXPECTED_COLS:
            continue  # e.g. the "k-no-data" empty-result row

        def cell_text(idx: int) -> str:
            return cells[idx].get_text(strip=True)

        documents = []
        attachments_raw = cell_text(_COL_ATTACHMENTS)
        for entry in attachments_raw.split(";"):
            entry = entry.strip()
            if not entry or "&" not in entry:
                continue
            name, path = entry.split("&", 1)
            name, path = name.strip(), path.strip()
            if not path:
                continue
            documents.append(
                DevelopmentPlanDocument(name=name, url=f"{ATTACHMENT_URL}?filename={quote(path)}")
            )

        records.append(
            DevelopmentPlanRecord(
                reference=cell_text(_COL_DESCRIPTION),
                office=cell_text(_COL_OFFICE) or None,
                scope_text=cell_text(_COL_SCOPE),
                procedure_type=cell_text(_COL_PROCEDURE) or None,
                documents=documents,
            )
        )

    return records


def search_development_plans(
    cadastral_id: str, cache: GisCache | None = None
) -> list[DevelopmentPlanRecord]:
    """Searches NAG Sofia's development-plan case registry for records
    referencing `cadastral_id`. Returns an empty list for "no matching
    cases found" (a legitimate, common answer for most parcels — this
    registry only covers parcels with recent planning activity), and
    raises NagSofiaError only on an actual failure to reach/parse the
    site."""
    cache_params = {"Identifier": cadastral_id}
    if cache is not None:
        cached = cache.get("nag_sofia", SEARCH_ACTION_URL, cache_params)
        if cached is not None:
            logger.info("NAG Sofia cache hit for Identifier=%s", cadastral_id)
            return [DevelopmentPlanRecord(**r) for r in cached["records"]]

    session = requests.Session()
    query_id = _get_search_query_id(session)

    try:
        resp = session.get(
            SEARCH_ACTION_URL,
            params={"searchQueryId": query_id, "Identifier": cadastral_id},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NagSofiaError(f"NAG Sofia search request failed: {exc}") from exc

    records = _parse_results(resp.text)

    if cache is not None:
        cache.set(
            "nag_sofia", SEARCH_ACTION_URL, cache_params,
            {"records": [r.model_dump() for r in records]},
        )
    return records
