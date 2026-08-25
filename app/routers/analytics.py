import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies import require_auth as get_current_user
from app.db.models import User
from app.templating import templates
from app.services.analytics_service import (
    get_filter_options,
    get_market_trend,
    get_neighborhoods_for_city,
    get_repeat_stats,
    refresh_mv,
)
from app.services.listing_service import GEO_CATEGORIES

router = APIRouter(prefix="/analytics", tags=["analytics"])

_N_RUNS_OPTIONS = [
    ("6",   "6 засичания"),
    ("12",  "12 засичания"),
    ("24",  "24 засичания"),
    ("999", "Всички"),
]


def _pct_delta(cur: float | int | None, prev: float | int | None) -> float | None:
    """% change from `prev` to `cur`, or None if either side is missing/zero."""
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / prev * 100, 1)


@router.get("/", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    deal_type:          str = Query(""),
    geo_category:       str = Query(""),
    city:               str = Query(""),
    neighborhood:       str = Query(""),
    property_type_slug: str = Query(""),
    n_runs:             int = Query(12),
):
    filter_opts = get_filter_options(db)

    trend_data   = get_market_trend(
        db,
        deal_type=deal_type or None,
        geo_category=geo_category or None,
        city=city or None,
        neighborhood=neighborhood or None,
        property_type_slug=property_type_slug or None,
        n_runs=n_runs,
    )
    repeat_data = get_repeat_stats(
        db,
        deal_type=deal_type or None,
        geo_category=geo_category or None,
        city=city or None,
        property_type_slug=property_type_slug or None,
        n_runs=n_runs,
    )

    # Neighborhood options for currently selected city
    neighborhoods = get_neighborhoods_for_city(db, city) if city else []

    # Latest-run KPIs (last entry in trend, which is sorted ASC); the entry
    # before it is the prior run, used for the period-over-period % deltas.
    latest = trend_data[-1] if trend_data else {}
    previous = trend_data[-2] if len(trend_data) > 1 else {}
    kpi_deltas = {
        "median_ppsqm": _pct_delta(latest.get("median_ppsqm"), previous.get("median_ppsqm")),
        "n_listings":   _pct_delta(latest.get("n_listings"),   previous.get("n_listings")),
        "median_dom":   _pct_delta(latest.get("median_dom"),   previous.get("median_dom")),
    }

    return templates.TemplateResponse(
        request,
        "analytics.html",
        {
            "trend_data":       trend_data,
            "trend_json":       json.dumps(trend_data, default=str),
            "repeat_data":      repeat_data,
            "repeat_json":      json.dumps(repeat_data, default=str),
            "latest":           latest,
            "kpi_deltas":       kpi_deltas,
            "filter_opts":      filter_opts,
            "geo_categories":   GEO_CATEGORIES,
            "neighborhoods":    neighborhoods,
            "n_runs_options":   _N_RUNS_OPTIONS,
            # Active filter values (for re-populating the form)
            "f_deal_type":          deal_type,
            "f_geo_category":       geo_category,
            "f_city":               city,
            "f_neighborhood":       neighborhood,
            "f_property_type_slug": property_type_slug,
            "f_n_runs":             n_runs,
        },
    )


@router.get("/filters/neighborhoods", response_class=HTMLResponse)
async def neighborhoods_options(
    city: str = Query(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """HTMX endpoint: returns <option> elements for the neighborhood dropdown."""
    neighborhoods = get_neighborhoods_for_city(db, city) if city else []
    html = '<option value="">— Всички квартали —</option>'
    for n in neighborhoods:
        html += f'<option value="{n}">{n}</option>'
    return HTMLResponse(html)


@router.post("/refresh-mv", response_class=HTMLResponse)
async def trigger_mv_refresh(
    request: Request,
    user: User = Depends(get_current_user),
):
    """Admin action: manually refresh the analytics materialized view."""
    if user.role != "admin":
        return HTMLResponse("Само администратори могат да извършват тази операция.", status_code=403)
    refresh_mv()
    return HTMLResponse(
        '<div class="flash flash-ok">Материализираният изглед е обновен успешно.</div>'
    )
