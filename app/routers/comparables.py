import uuid
from datetime import date

from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, ComparablePool, User
from app.db.session import get_db
from app.dependencies import require_auth as get_current_user
from app.templating import templates
from app.services import avm_service, gis_service
from app.services.comparable_service import (
    ADJUSTMENT_FACTOR_LABELS,
    MAX_PINNED,
    add_to_pool,
    clear_pool,
    delete_user_report,
    export_excel,
    finalize_user_report,
    generate_docx,
    get_or_create_draft,
    get_purpose_options,
    get_pool_with_stats,
    get_report_for_user,
    get_user_reports,
    new_draft,
    remove_from_pool,
    toggle_pin,
    update_conclusion,
    update_income_approach,
    update_legal_description,
    update_pool_adjustment,
    update_residual_approach,
    update_sales_approach,
    update_subject,
    update_submarket_rationale,
)
from utils.feature_engineering import PROPERTY_TYPE_DISPLAY
from utils.ml.avm_features import GEO_CATEGORIES, SEGMENT_DISPLAY_NAMES, SEGMENT_PROPERTY_TYPES

router = APIRouter(prefix="/comparables", tags=["comparables"])

_VALID_REPORT_PURPOSES = {slug for slug, _ in get_purpose_options()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_report(request: Request, db: Session, user: User) -> AppraisalReport:
    """Return the report stored in session, or find/create a draft for the user."""
    rid_str = request.session.get("active_report_id")
    if rid_str:
        try:
            rid = uuid.UUID(rid_str)
            report = get_report_for_user(db, rid, user.id)
            if report:
                return report
        except Exception:
            pass
    report = get_or_create_draft(db, user.id)
    request.session["active_report_id"] = str(report.id)
    return report


def _panel_response(request: Request, db: Session, ctype: str, report_id: uuid.UUID):
    pool = get_pool_with_stats(db, ctype, report_id)
    return templates.TemplateResponse(
        request,
        "comparables/_pool_panel.html",
        {
            "pool": pool,
            "ctype": ctype,
            "report_id": str(report_id),
            "ppsqm_label": "EUR/кв.м" if ctype == "sale" else "EUR/кв.м/мес",
            "MAX_PINNED": MAX_PINNED,
            "adjustment_factor_labels": ADJUSTMENT_FACTOR_LABELS,
        },
    )


def _htmx_or_redirect(
    request: Request, db: Session, ctype: str, report_id: uuid.UUID
):
    if request.headers.get("HX-Request"):
        return _panel_response(request, db, ctype, report_id)
    return RedirectResponse(url="/comparables/", status_code=303)


def _pool_item_guard(
    db: Session, pool_id: int, user: User
) -> ComparablePool:
    """Load pool item and verify ownership. Raises 403 on mismatch."""
    item = db.get(ComparablePool, pool_id)
    if not item or item.user_id != user.id:
        raise HTTPException(status_code=403)
    return item


# ── Main page ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def comparables_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    pool_sale = get_pool_with_stats(db, "sale", report.id)
    pool_rent = get_pool_with_stats(db, "rent", report.id)
    # Both of these make blocking network calls (AGKK/isofmap.bg/NAG Sofia,
    # cold-start joblib.load) — offloaded to a worker thread so a slow
    # external response doesn't stall the single asyncio event loop for
    # every other concurrent request.
    avm = await run_in_threadpool(avm_service.predict_sales_value, db, report)
    cadastre = await run_in_threadpool(gis_service.get_cadastre_panel_data, report)
    property_type_groups = [
        (SEGMENT_DISPLAY_NAMES[segment], [(slug, PROPERTY_TYPE_DISPLAY.get(slug, slug)) for slug in slugs])
        for segment, slugs in SEGMENT_PROPERTY_TYPES.items()
    ]
    return templates.TemplateResponse(
        request,
        "comparables.html",
        {
            "report": report,
            "pool_sale": pool_sale,
            "pool_rent": pool_rent,
            "avm": avm,
            "cadastre": cadastre,
            "property_type_groups": property_type_groups,
            "geo_categories": GEO_CATEGORIES,
            "segment_display_names": SEGMENT_DISPLAY_NAMES,
            "report_purpose_options": get_purpose_options(),
            "MAX_PINNED": MAX_PINNED,
            "adjustment_factor_labels": ADJUSTMENT_FACTOR_LABELS,
        },
    )


# ── Pool mutations ────────────────────────────────────────────────────────────

@router.post("/add")
async def add_comparables(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    listing_ids: list[int] = Form(default=[]),
    comparable_type: str = Form("sale"),
):
    report = _active_report(request, db, user)
    add_to_pool(db, listing_ids, comparable_type, report.id, user.id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/remove/{pool_id}")
async def remove_comparable(
    pool_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = _pool_item_guard(db, pool_id, user)
    ctype, report_id = item.comparable_type, item.report_id
    remove_from_pool(db, pool_id)
    return _htmx_or_redirect(request, db, ctype, report_id)


@router.post("/pin/{pool_id}")
async def pin_comparable(
    pool_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = _pool_item_guard(db, pool_id, user)
    ctype, report_id = item.comparable_type, item.report_id
    toggle_pin(db, pool_id)
    return _htmx_or_redirect(request, db, ctype, report_id)


@router.post("/clear")
async def clear_comparables(
    request: Request,
    comparable_type: str = Form(""),
    report_id: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    try:
        rid = uuid.UUID(report_id) if report_id else report.id
    except ValueError:
        rid = report.id
    clear_pool(db, rid, comparable_type or None)
    return _htmx_or_redirect(request, db, comparable_type or "sale", rid)


@router.post("/adjustment/{pool_id}")
async def save_adjustment(
    pool_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    adjustment_pct: str = Form("0"),
    analyst_note: str = Form(""),
    mode: str = Form("simple"),
    adj_market: str = Form("0"),
    adj_location: str = Form("0"),
    adj_size: str = Form("0"),
    adj_floor: str = Form("0"),
    adj_condition: str = Form("0"),
):
    item = _pool_item_guard(db, pool_id, user)
    ctype, report_id = item.comparable_type, item.report_id

    def _f(v: str) -> float:
        try:
            return float(v) if v.strip() else 0.0
        except ValueError:
            return 0.0

    if mode == "factors":
        factors = {
            "market": _f(adj_market),
            "location": _f(adj_location),
            "size": _f(adj_size),
            "floor": _f(adj_floor),
            "condition": _f(adj_condition),
        }
        update_pool_adjustment(db, pool_id, None, analyst_note, adjustment_factors=factors)
    else:
        adj = None
        try:
            adj = float(adjustment_pct) if adjustment_pct.strip() else None
        except ValueError:
            pass
        update_pool_adjustment(db, pool_id, adj, analyst_note)
    return _htmx_or_redirect(request, db, ctype, report_id)


# ── Subject & report management ───────────────────────────────────────────────

@router.post("/subject")
async def save_subject(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    title: str = Form(""),
    subject_address: str = Form(""),
    subject_city: str = Form(""),
    subject_area_sqm: str = Form(""),
    subject_floor: str = Form(""),
    subject_total_floors: str = Form(""),
    subject_construction: str = Form(""),
    subject_year: str = Form(""),
    subject_description: str = Form(""),
    valuation_date: str = Form(""),
    subject_property_type: str = Form(""),
    subject_geo_category: str = Form(""),
    subject_neighborhood: str = Form(""),
    subject_cadastral_id: str = Form(""),
    report_purpose: str = Form(""),
):
    def _int(v): return int(v) if v.strip() else None
    def _float(v): return float(v) if v.strip() else None
    def _date(v):
        try: return date.fromisoformat(v) if v.strip() else None
        except ValueError: return None

    report = _active_report(request, db, user)
    update_subject(db, report.id, {
        "title": title or "Нов доклад",
        "subject_address": subject_address,
        "subject_city": subject_city,
        "subject_area_sqm": _float(subject_area_sqm),
        "subject_floor": _int(subject_floor),
        "subject_total_floors": _int(subject_total_floors),
        "subject_construction": subject_construction,
        "subject_year": _int(subject_year),
        "subject_description": subject_description,
        "valuation_date": _date(valuation_date),
        "subject_property_type": subject_property_type,
        "subject_geo_category": subject_geo_category,
        "subject_neighborhood": subject_neighborhood,
        "subject_cadastral_id": subject_cadastral_id.strip(),
        "report_purpose": report_purpose if report_purpose in _VALID_REPORT_PURPOSES else "",
    })
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/new-report")
async def new_report_endpoint(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = new_draft(db, user.id)
    request.session["active_report_id"] = str(report.id)
    return RedirectResponse(url="/comparables/", status_code=303)


# ── Export ────────────────────────────────────────────────────────────────────

@router.post("/save-income")
async def save_income_approach(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    rent_per_sqm_month: str = Form(""),
    cap_rate_pct: str = Form(""),
    concluded_per_sqm: str = Form(""),
    subject_area_sqm: str = Form(""),
):
    def _f(v): return float(v) if v.strip() else None
    report = _active_report(request, db, user)
    update_income_approach(
        db, report.id,
        rent_per_sqm_month=_f(rent_per_sqm_month),
        cap_rate_pct=_f(cap_rate_pct),
        concluded_per_sqm=_f(concluded_per_sqm),
        subject_area_sqm=_f(subject_area_sqm),
    )
    return RedirectResponse(url="/comparables/#income-panel", status_code=303)


@router.post("/save-sales")
async def save_sales_approach(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    concluded_value_sales: str = Form(""),
):
    # source is always "manual" here, deliberately -- the AVM prediction is a
    # sanity-check tool only and must never reach a value the final report
    # can cite as the sales-approach conclusion. There used to be a
    # source="avm" pathway wired to the AVM panel's own predicted number;
    # it was removed at this boundary (not just hidden in the template) so
    # it can't come back via a direct POST either.
    def _f(v): return float(v) if v.strip() else None
    report = _active_report(request, db, user)
    update_sales_approach(
        db, report.id,
        concluded_value_sales=_f(concluded_value_sales),
        source="manual",
    )
    return RedirectResponse(url="/comparables/#avm-panel", status_code=303)


@router.post("/save-submarket-rationale")
async def save_submarket_rationale(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    submarket_rationale: str = Form(""),
):
    report = _active_report(request, db, user)
    update_submarket_rationale(db, report.id, submarket_rationale.strip())
    return RedirectResponse(url="/comparables/#tab-content-sale", status_code=303)


@router.post("/save-legal-description")
async def save_legal_description(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    legal_description: str = Form(""),
    source: str = Form("agkk"),
):
    report = _active_report(request, db, user)
    update_legal_description(
        db, report.id,
        text=legal_description.strip(),
        source=source if source in ("agkk", "manual") else "manual",
    )
    return RedirectResponse(url="/comparables/#cadastre-panel", status_code=303)


@router.post("/save-residual")
async def save_residual_approach(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    concluded_value_residual: str = Form(""),
):
    def _f(v): return float(v) if v.strip() else None
    report = _active_report(request, db, user)
    update_residual_approach(db, report.id, concluded_value_residual=_f(concluded_value_residual))
    return RedirectResponse(url="/comparables/#residual-panel", status_code=303)


@router.post("/save-conclusion")
async def save_conclusion(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    weight_sales_pct: str = Form(""),
    weight_income_pct: str = Form(""),
    weight_residual_pct: str = Form(""),
    weighting_rationale: str = Form(""),
):
    def _f(v): return float(v) if v.strip() else None
    report = _active_report(request, db, user)
    update_conclusion(
        db, report.id,
        weight_sales_pct=_f(weight_sales_pct),
        weight_income_pct=_f(weight_income_pct),
        weight_residual_pct=_f(weight_residual_pct),
        weighting_rationale=weighting_rationale.strip(),
    )
    return RedirectResponse(url="/comparables/#conclusion-panel", status_code=303)


@router.get("/export/excel")
async def export_excel_download(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    buf = export_excel(db, report)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename*=UTF-8''sravnimi.xlsx"},
    )


@router.get("/export/docx")
async def export_docx_download(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    buf = generate_docx(db, report)
    safe_title = (report.title or "ocenka").replace(" ", "_")[:40]
    encoded = quote(safe_title, safe="")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}.docx"},
    )
