from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import EmailStr
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import User, UserConsent
from app.db.session import get_db
from app.rate_limit import limiter
from app.services import auth_service
from app.services.auth_service import PRIVACY_VERSION, TERMS_VERSION
from app.templating import templates

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


# ── Login ──────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.state.user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(
        request, "auth/login.html",
        {"next": request.query_params.get("next", "/")},
    )


@router.post("/login", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def login_post(
    request: Request,
    db: Session = Depends(get_db),
    login: str = Form(...),
    password: str = Form(..., max_length=128),
    next: str = Form(default="/"),
):
    user = auth_service.authenticate(db, login, password)
    if not user:
        return templates.TemplateResponse(
            request, "auth/login.html",
            {"error": "Невалиден имейл/потребителско ime или парола.", "next": next},
            status_code=401,
        )
    request.session["user_id"] = user.id
    auth_service.update_last_login(db, user)
    db.commit()
    safe_next = next if next.startswith("/") else "/"
    return RedirectResponse(url=safe_next, status_code=302)


# ── Logout ─────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)


# ── Register ───────────────────────────────────────────────────────────────────

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if request.state.user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(
        request, "auth/register.html",
        {"PRIVACY_VERSION": PRIVACY_VERSION, "TERMS_VERSION": TERMS_VERSION},
    )


@router.post("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register_post(
    request: Request,
    db: Session = Depends(get_db),
    email: EmailStr = Form(..., max_length=255),
    username: str = Form(..., max_length=100),
    password: str = Form(..., max_length=128),
    password2: str = Form(..., max_length=128),
    full_name: str = Form(default="", max_length=255),
    accept_privacy: str = Form(default=""),
    accept_terms: str = Form(default=""),
):
    errors = []

    if not accept_privacy:
        errors.append("Трябва да приемете Политиката за поверителност.")
    if not accept_terms:
        errors.append("Трябва да приемете Общите условия.")
    if len(password) < 8:
        errors.append("Паролата трябва да е поне 8 символа.")
    if password != password2:
        errors.append("Двете пароли не съвпадат.")
    if not errors and auth_service.get_user_by_email(db, email):
        errors.append("Имейл адресът вече е регистриран.")
    if not errors and auth_service.get_user_by_username(db, username):
        errors.append("Потребителското ime вече е заето.")

    if errors:
        return templates.TemplateResponse(
            request, "auth/register.html",
            {
                "errors": errors,
                "form": {"email": email, "username": username, "full_name": full_name},
                "PRIVACY_VERSION": PRIVACY_VERSION,
                "TERMS_VERSION": TERMS_VERSION,
            },
            status_code=422,
        )

    # First user matching admin_email gets admin (only if no admin exists yet)
    role = "user"
    if settings.admin_email and email.lower().strip() == settings.admin_email.lower():
        existing_admin = db.query(User).filter(User.role == "admin").first()
        if not existing_admin:
            role = "admin"

    user = auth_service.create_user(
        db, email=email, username=username, password=password,
        full_name=full_name, role=role,
    )
    ip = _client_ip(request)
    auth_service.record_consent(db, user.id, "privacy_policy", PRIVACY_VERSION, ip)
    auth_service.record_consent(db, user.id, "terms", TERMS_VERSION, ip)
    db.commit()

    request.session["user_id"] = user.id
    return RedirectResponse(url="/", status_code=302)


# ── Profile ────────────────────────────────────────────────────────────────────

@router.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request, db: Session = Depends(get_db)):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login?next=/auth/profile", status_code=302)
    consents = (
        db.query(UserConsent)
        .filter_by(user_id=user.id)
        .order_by(UserConsent.accepted_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        request, "profile.html",
        {
            "consents": consents,
            "PRIVACY_VERSION": PRIVACY_VERSION,
            "TERMS_VERSION": TERMS_VERSION,
            "saved": request.query_params.get("saved"),
            "pw_saved": request.query_params.get("pw_saved"),
        },
    )


@router.post("/profile/update", response_class=HTMLResponse)
async def profile_update(
    request: Request,
    db: Session = Depends(get_db),
    full_name: str = Form(default="", max_length=255),
    email: EmailStr = Form(..., max_length=255),
    username: str = Form(..., max_length=100),
    appraiser_certificate_no: str = Form(default="", max_length=100),
):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    errors = []
    existing = auth_service.get_user_by_email(db, email)
    if existing and existing.id != user.id:
        errors.append("Имейлът вече се използва от друг акаунт.")
    existing_u = auth_service.get_user_by_username(db, username)
    if existing_u and existing_u.id != user.id:
        errors.append("Потребителското ime вече е заето.")

    if errors:
        consents = db.query(UserConsent).filter_by(user_id=user.id).order_by(UserConsent.accepted_at.desc()).all()
        return templates.TemplateResponse(
            request, "profile.html",
            {"errors": errors, "consents": consents, "PRIVACY_VERSION": PRIVACY_VERSION, "TERMS_VERSION": TERMS_VERSION},
            status_code=422,
        )

    db_user = db.get(User, user.id)
    db_user.email = email.lower().strip()
    db_user.username = username.strip()
    db_user.full_name = full_name.strip() or None
    db_user.appraiser_certificate_no = appraiser_certificate_no.strip() or None
    db.commit()
    return RedirectResponse(url="/auth/profile?saved=1", status_code=302)


@router.post("/profile/change-password", response_class=HTMLResponse)
async def change_password(
    request: Request,
    db: Session = Depends(get_db),
    current_password: str = Form(..., max_length=128),
    new_password: str = Form(..., max_length=128),
    new_password2: str = Form(..., max_length=128),
):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)

    errors = []
    if not auth_service.verify_password(current_password, user.hashed_password):
        errors.append("Текущата парола е грешна.")
    if len(new_password) < 8:
        errors.append("Новата парола трябва да е поне 8 символа.")
    if new_password != new_password2:
        errors.append("Двете нови пароли не съвпадат.")

    if errors:
        consents = db.query(UserConsent).filter_by(user_id=user.id).order_by(UserConsent.accepted_at.desc()).all()
        return templates.TemplateResponse(
            request, "profile.html",
            {"pw_errors": errors, "consents": consents, "PRIVACY_VERSION": PRIVACY_VERSION, "TERMS_VERSION": TERMS_VERSION},
            status_code=422,
        )

    db_user = db.get(User, user.id)
    db_user.hashed_password = auth_service.hash_password(new_password)
    db.commit()
    return RedirectResponse(url="/auth/profile?pw_saved=1", status_code=302)


@router.post("/profile/revoke-consent/{consent_id}")
async def revoke_consent(request: Request, consent_id: int, db: Session = Depends(get_db)):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    auth_service.revoke_consent(db, consent_id, user.id)
    db.commit()
    return RedirectResponse(url="/auth/profile#consents", status_code=302)


@router.post("/profile/accept-consent")
async def accept_consent(
    request: Request,
    db: Session = Depends(get_db),
    consent_type: str = Form(...),
):
    user = request.state.user
    if not user:
        return RedirectResponse(url="/auth/login", status_code=302)
    version = PRIVACY_VERSION if consent_type == "privacy_policy" else TERMS_VERSION
    auth_service.record_consent(db, user.id, consent_type, version, _client_ip(request))
    db.commit()
    return RedirectResponse(url="/auth/profile#consents", status_code=302)
