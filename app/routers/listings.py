from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.templating import templates
from app.services.listing_service import (
    GEO_CATEGORIES,
    RESULTS_PER_PAGE,
    SORT_OPTIONS,
    SearchFilters,
    get_cities_for_filter,
    get_listing_detail,
    get_listing_snapshots,
    get_property_types_for_filter,
    get_quarters_for_filter,
    search_listings,
)

router = APIRouter(prefix="/listings", tags=["listings"])


def _parse_filters(
    deal_type: str = Query(""),
    city: list[str] = Query(default=[]),
    quarter: str = Query(""),
    property_types: list[str] = Query(default=[]),
    geo_categories: list[str] = Query(default=[]),
    min_price: str = Query(""),
    max_price: str = Query(""),
    min_area: str = Query(""),
    max_area: str = Query(""),
    min_ppsqm: str = Query(""),
    max_ppsqm: str = Query(""),
    sort: str = Query("last_seen"),
) -> SearchFilters:
    def _f(v: str) -> float | None:
        try:
            return float(v) if v.strip() else None
        except ValueError:
            return None

    return SearchFilters(
        deal_type=deal_type,
        city=city,
        quarter=quarter,
        property_types=property_types,
        geo_categories=geo_categories,
        min_price=_f(min_price),
        max_price=_f(max_price),
        min_area=_f(min_area),
        max_area=_f(max_area),
        min_ppsqm=_f(min_ppsqm),
        max_ppsqm=_f(max_ppsqm),
        sort=sort,
    )


@router.get("/", response_class=HTMLResponse)
async def listings_page(
    request: Request,
    page: int = Query(1, ge=1),
    filters: SearchFilters = Depends(_parse_filters),
    db: Session = Depends(get_db),
):
    rows, total = search_listings(db, filters, page=page)
    pages = max(1, (total + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
    property_types = get_property_types_for_filter(db)
    cities = get_cities_for_filter(db)
    return templates.TemplateResponse(
        request,
        "listings/search.html",
        {
            "rows": rows,
            "total": total,
            "page": page,
            "pages": pages,
            "filters": filters,
            "geo_categories": GEO_CATEGORIES,
            "sort_options": SORT_OPTIONS,
            "property_types": property_types,
            "cities": cities,
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def listings_search_partial(
    request: Request,
    page: int = Query(1, ge=1),
    filters: SearchFilters = Depends(_parse_filters),
    db: Session = Depends(get_db),
):
    """HTMX partial — returns only the results section."""
    rows, total = search_listings(db, filters, page=page)
    pages = max(1, (total + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
    return templates.TemplateResponse(
        request,
        "listings/_results.html",
        {
            "rows": rows,
            "total": total,
            "page": page,
            "pages": pages,
            "filters": filters,
        },
    )


@router.get("/ids")
async def listing_ids(
    filters: SearchFilters = Depends(_parse_filters),
    db: Session = Depends(get_db),
):
    """Returns all listing IDs matching current filters (max 5000), for select-all."""
    from app.services.listing_service import get_all_listing_ids
    ids = get_all_listing_ids(db, filters)
    return {"ids": ids, "total": len(ids)}


@router.get("/quarters", response_class=HTMLResponse)
async def quarters_datalist(
    city: list[str] = Query(default=[]),
    db: Session = Depends(get_db),
):
    """Returns <option> tags for the quarter <datalist> filtered by selected cities."""
    quarters = get_quarters_for_filter(db, city)
    options = "".join(f'<option value="{q}">' for q in quarters)
    return HTMLResponse(content=options)


@router.get("/{listing_id}", response_class=HTMLResponse)
async def listing_detail(
    listing_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    listing = get_listing_detail(db, listing_id)
    if listing is None:
        return HTMLResponse("<h2>Обявата не е намерена.</h2>", status_code=404)
    snapshots = get_listing_snapshots(db, listing_id)
    features = [
        f.strip()
        for f in (listing.get("features_pipe") or "").split("|")
        if f.strip()
    ]
    return templates.TemplateResponse(
        request,
        "listings/detail.html",
        {
            "l": listing,
            "snapshots": snapshots,
            "features": features,
        },
    )
