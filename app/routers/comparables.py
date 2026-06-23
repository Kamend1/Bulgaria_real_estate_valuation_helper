import uuid
from datetime import date

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, ComparablePool, User
from app.db.session import get_db
from app.dependencies import require_auth as get_current_user
from app.templating import templates
from app.services.comparable_service import (
    MAX_PINNED,
    add_to_pool,
    clear_pool,
    delete_user_report,
    export_excel,
    finalize_user_report,
    get_or_create_draft,
    get_pool_with_stats,
    get_report_for_user,
    get_user_reports,
    new_draft,
    remove_from_pool,
    toggle_pin,
    update_pool_adjustment,
    update_subject,
)

router = APIRouter(prefix="/comparables", tags=["comparables"])


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
    return templates.TemplateResponse(
        request,
        "comparables.html",
        {
            "report": report,
            "pool_sale": pool_sale,
            "pool_rent": pool_rent,
            "MAX_PINNED": MAX_PINNED,
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
):
    item = _pool_item_guard(db, pool_id, user)
    ctype, report_id = item.comparable_type, item.report_id
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
