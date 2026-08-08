import subprocess
import sys
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db.models import AvmModel, User, UserConsent
from app.db.session import get_db
from app.services import auth_service
from app.templating import templates
from utils.ml.avm_features import SEGMENT_DISPLAY_NAMES, SEGMENT_PROPERTY_TYPES

router = APIRouter(prefix="/admin", tags=["admin"])

_avm_training_active = False


def _run_avm_training(segment: str | None) -> None:
    global _avm_training_active
    try:
        cmd = [sys.executable, "-m", "scripts.train_avm_model"]
        if segment:
            cmd += ["--segment", segment]
        subprocess.run(cmd, check=False)
    finally:
        _avm_training_active = False


def _require_admin(request: Request):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login?next=/admin/", status_code=302)
    if user.role != "admin":
        return RedirectResponse(url="/", status_code=302)
    return None  # OK


# ── User list ──────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def admin_home(request: Request, db: Session = Depends(get_db), q: str = ""):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    query = db.query(User)
    if q:
        like = f"%{q}%"
        query = query.filter(
            (User.email.ilike(like)) | (User.username.ilike(like)) | (User.full_name.ilike(like))
        )
    users = query.order_by(User.created_at.desc()).all()
    return templates.TemplateResponse(
        request, "admin/users.html", {"users": users, "q": q}
    )


# ── User detail ────────────────────────────────────────────────────────────────

@router.get("/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail(request: Request, user_id: int, db: Session = Depends(get_db)):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)

    consents = (
        db.query(UserConsent)
        .filter_by(user_id=user_id)
        .order_by(UserConsent.accepted_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        request, "admin/user_detail.html",
        {
            "target": target,
            "consents": consents,
            "saved": request.query_params.get("saved"),
        },
    )


# ── Toggle active ──────────────────────────────────────────────────────────────

@router.post("/users/{user_id}/toggle-active")
async def toggle_active(request: Request, user_id: int, db: Session = Depends(get_db)):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)
    if target.id == request.state.user.id:
        return RedirectResponse(url=f"/admin/users/{user_id}?saved=0", status_code=302)

    target.is_active = not target.is_active
    db.commit()
    return RedirectResponse(url=f"/admin/users/{user_id}?saved=1", status_code=302)


# ── Change role ────────────────────────────────────────────────────────────────

@router.post("/users/{user_id}/set-role")
async def set_role(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    role: str = Form(...),
):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    if role not in ("user", "admin"):
        return HTMLResponse("Невалидна роля.", status_code=400)

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)
    if target.id == request.state.user.id:
        return RedirectResponse(url=f"/admin/users/{user_id}?saved=0", status_code=302)

    target.role = role
    db.commit()
    return RedirectResponse(url=f"/admin/users/{user_id}?saved=1", status_code=302)


# ── Update profile (admin edit) ────────────────────────────────────────────────

@router.post("/users/{user_id}/update")
async def admin_update_user(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    email: str = Form(...),
    username: str = Form(...),
    full_name: str = Form(default=""),
):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)

    errors = []
    existing = auth_service.get_user_by_email(db, email)
    if existing and existing.id != user_id:
        errors.append("Имейлът вече се използва.")
    existing_u = auth_service.get_user_by_username(db, username)
    if existing_u and existing_u.id != user_id:
        errors.append("Потребителското ime вече е заето.")

    if errors:
        consents = db.query(UserConsent).filter_by(user_id=user_id).order_by(UserConsent.accepted_at.desc()).all()
        return templates.TemplateResponse(
            request, "admin/user_detail.html",
            {"target": target, "consents": consents, "errors": errors},
            status_code=422,
        )

    target.email = email.lower().strip()
    target.username = username.strip()
    target.full_name = full_name.strip() or None
    db.commit()
    return RedirectResponse(url=f"/admin/users/{user_id}?saved=1", status_code=302)


# ── Reset password (admin) ─────────────────────────────────────────────────────

@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    new_password: str = Form(...),
    new_password2: str = Form(...),
):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)

    errors = []
    if len(new_password) < 8:
        errors.append("Паролата трябва да е поне 8 символа.")
    if new_password != new_password2:
        errors.append("Двете пароли не съвпадат.")

    if errors:
        consents = db.query(UserConsent).filter_by(user_id=user_id).order_by(UserConsent.accepted_at.desc()).all()
        return templates.TemplateResponse(
            request, "admin/user_detail.html",
            {"target": target, "consents": consents, "pw_errors": errors},
            status_code=422,
        )

    target.hashed_password = auth_service.hash_password(new_password)
    db.commit()
    return RedirectResponse(url=f"/admin/users/{user_id}?saved=1", status_code=302)


# ── Create user (admin) ────────────────────────────────────────────────────────

@router.get("/users/new/form", response_class=HTMLResponse)
async def admin_new_user_form(request: Request):
    redirect = _require_admin(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request, "admin/user_new.html", {})


@router.post("/users/new/create")
async def admin_create_user(
    request: Request,
    db: Session = Depends(get_db),
    email: str = Form(...),
    username: str = Form(...),
    full_name: str = Form(default=""),
    password: str = Form(...),
    password2: str = Form(...),
    role: str = Form(default="user"),
):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    errors = []
    if len(password) < 8:
        errors.append("Паролата трябва да е поне 8 символа.")
    if password != password2:
        errors.append("Двете пароли не съвпадат.")
    if not errors and auth_service.get_user_by_email(db, email):
        errors.append("Имейлът вече е регистриран.")
    if not errors and auth_service.get_user_by_username(db, username):
        errors.append("Потребителското ime вече е заето.")
    if role not in ("user", "admin"):
        errors.append("Невалидна роля.")

    if errors:
        return templates.TemplateResponse(
            request, "admin/user_new.html",
            {"errors": errors, "form": {"email": email, "username": username, "full_name": full_name, "role": role}},
            status_code=422,
        )

    user = auth_service.create_user(db, email=email, username=username, password=password, full_name=full_name, role=role)
    db.commit()
    return RedirectResponse(url=f"/admin/users/{user.id}?saved=1", status_code=302)


# ── Delete user ────────────────────────────────────────────────────────────────

@router.post("/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: int, db: Session = Depends(get_db)):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    target = db.get(User, user_id)
    if not target:
        return HTMLResponse("Потребителят не е намерен.", status_code=404)
    if target.id == request.state.user.id:
        return RedirectResponse(url=f"/admin/users/{user_id}?saved=0", status_code=302)

    db.delete(target)
    db.commit()
    return RedirectResponse(url="/admin/?deleted=1", status_code=302)


# ── AVM model registry ───────────────────────────────────────────────────────

@router.get("/avm", response_class=HTMLResponse)
async def admin_avm(request: Request, db: Session = Depends(get_db)):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    models = (
        db.query(AvmModel)
        .order_by(AvmModel.segment, AvmModel.trained_at.desc())
        .all()
    )
    by_segment: dict[str, list[AvmModel]] = {seg: [] for seg in SEGMENT_PROPERTY_TYPES}
    for m in models:
        by_segment.setdefault(m.segment, []).append(m)

    return templates.TemplateResponse(
        request, "admin/avm.html",
        {
            "by_segment": by_segment,
            "segment_display_names": SEGMENT_DISPLAY_NAMES,
            "training_active": _avm_training_active,
            "started": request.query_params.get("started"),
        },
    )


@router.post("/avm/retrain")
async def admin_avm_retrain(request: Request, segment: str = Form("")):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    global _avm_training_active
    if not _avm_training_active:
        _avm_training_active = True
        threading.Thread(
            target=_run_avm_training,
            args=(segment or None,),
            daemon=True,
        ).start()
    return RedirectResponse(url="/admin/avm?started=1", status_code=302)
