from datetime import date

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.templating import templates
from app.services.comparable_service import (
    MAX_PINNED,
    add_to_pool,
    clear_pool,
    export_excel,
    get_or_create_draft,
    get_pool_with_stats,
    new_draft,
    remove_from_pool,
    toggle_pin,
    update_pool_adjustment,
    update_subject,
)

router = APIRouter(prefix="/comparables", tags=["comparables"])


@router.get("/", response_class=HTMLResponse)
async def comparables_page(request: Request, db: Session = Depends(get_db)):
    report = get_or_create_draft(db)
    pool_sale = get_pool_with_stats(db, "sale")
    pool_rent = get_pool_with_stats(db, "rent")
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


@router.post("/add")
async def add_comparables(
    db: Session = Depends(get_db),
    listing_ids: list[int] = Form(default=[]),
    comparable_type: str = Form("sale"),
):
    add_to_pool(db, listing_ids, comparable_type)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/remove/{pool_id}")
async def remove_comparable(pool_id: int, db: Session = Depends(get_db)):
    remove_from_pool(db, pool_id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/pin/{pool_id}")
async def pin_comparable(pool_id: int, db: Session = Depends(get_db)):
    toggle_pin(db, pool_id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/clear")
async def clear_comparables(
    comparable_type: str = Form(""),
    db: Session = Depends(get_db),
):
    clear_pool(db, comparable_type or None)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/adjustment/{pool_id}")
async def save_adjustment(
    pool_id: int,
    db: Session = Depends(get_db),
    adjustment_pct: str = Form("0"),
    analyst_note: str = Form(""),
):
    adj = None
    try:
        adj = float(adjustment_pct) if adjustment_pct.strip() else None
    except ValueError:
        pass
    update_pool_adjustment(db, pool_id, adj, analyst_note)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/subject")
async def save_subject(
    db: Session = Depends(get_db),
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

    report = get_or_create_draft(db)
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
async def new_report(db: Session = Depends(get_db)):
    new_draft(db)
    clear_pool(db)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.get("/export/excel")
async def export_excel_download(db: Session = Depends(get_db)):
    report = get_or_create_draft(db)
    buf = export_excel(db, report)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename*=UTF-8''sravnimi.xlsx"},
    )
