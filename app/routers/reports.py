import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User
from app.dependencies import require_auth as get_current_user
from app.templating import templates
from app.services.comparable_service import (
    delete_user_report,
    finalize_user_report,
    get_report_for_user,
    get_user_reports,
    new_draft,
    new_scratch_draft,
    promote_scratch_report,
)

router = APIRouter(prefix="/reports", tags=["reports"])

# Whitelisted redirect targets for /reports/{id}/open's `next` field --
# never accepts an arbitrary path (would be an open redirect); the switcher
# partial (_report_switcher.html) is the only current caller that sends a
# non-default value, so a user could only ever land back on the page they
# switched from.
_OPEN_REDIRECT_TARGETS = {"/comparables/", "/assistant/"}


@router.get("/", response_class=HTMLResponse)
async def reports_list(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    show_scratch: bool = False,
):
    reports = get_user_reports(db, user.id, include_scratch=show_scratch)
    active_rid = request.session.get("active_report_id")
    return templates.TemplateResponse(
        request,
        "reports.html",
        {"reports": reports, "active_report_id": active_rid, "show_scratch": show_scratch},
    )


@router.post("/{report_id}/open")
async def open_report(
    report_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    next: str = Form("/comparables/"),
):
    report = get_report_for_user(db, report_id, user.id)
    if not report:
        raise HTTPException(status_code=404)
    request.session["active_report_id"] = str(report.id)
    # `next` lets the quick switcher (_report_switcher.html) return you to
    # the page you switched from (e.g. /assistant/) instead of always
    # bouncing to /comparables/ -- whitelisted, never an open redirect.
    target = next if next in _OPEN_REDIRECT_TARGETS else "/comparables/"
    return RedirectResponse(url=target, status_code=303)


@router.post("/{report_id}/finalize")
async def finalize_report(
    report_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    finalize_user_report(db, report_id, user.id)
    return RedirectResponse(url="/reports/", status_code=303)


@router.post("/{report_id}/reopen")
async def reopen_report(
    report_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Set a finalized report back to draft so it can be edited again."""
    from app.db.models import AppraisalReport
    report = (
        db.query(AppraisalReport)
        .filter(AppraisalReport.id == report_id, AppraisalReport.user_id == user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404)
    report.status = "draft"
    db.commit()
    request.session["active_report_id"] = str(report.id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/{report_id}/delete")
async def delete_report(
    report_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = get_report_for_user(db, report_id, user.id)
    if not report:
        raise HTTPException(status_code=404)
    # If this was the active report, clear session
    if request.session.get("active_report_id") == str(report_id):
        request.session.pop("active_report_id", None)
    delete_user_report(db, report_id, user.id)
    return RedirectResponse(url="/reports/", status_code=303)


@router.post("/new")
async def new_report(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = new_draft(db, user.id)
    request.session["active_report_id"] = str(report.id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/new-scratch")
async def new_scratch_report(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """"Нов хипотетичен сценарий" (Phase 11, 2026-08-28) -- a normal draft
    report, just hidden from the default /reports/ list (see
    new_scratch_draft's own docstring). Lands on /comparables/ exactly like
    a real new report -- every panel (subject, AVM, GIS, income, AI
    Assistant) works on it unchanged."""
    report = new_scratch_draft(db, user.id)
    request.session["active_report_id"] = str(report.id)
    return RedirectResponse(url="/comparables/", status_code=303)


@router.post("/{report_id}/promote")
async def promote_report(
    report_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """"Направи истински доклад" -- flips is_scratch off so the scenario
    now shows up in the normal /reports/ list. Nothing else changes."""
    report = get_report_for_user(db, report_id, user.id)
    if not report:
        raise HTTPException(status_code=404)
    promote_scratch_report(db, report_id)
    return RedirectResponse(url="/comparables/", status_code=303)
