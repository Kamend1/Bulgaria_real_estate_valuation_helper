import logging
import logging.handlers
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.db.models import User
from app.db.session import get_db, db_session
from app.rate_limit import limiter
from app.routers import analytics, comparables, listings, reports, scrape
from app.routers import auth as auth_router
from app.routers import admin as admin_router
from app.services.csrf import verify_csrf_token
from app.templating import templates

_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}

_log_dir = Path(settings.log_dir)
_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        # 10 MB per file, keep 5 rotated backups -- bounds disk use for
        # long-running/unattended processes instead of growing forever.
        logging.handlers.RotatingFileHandler(
            _log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_title)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    message = "Твърде много опити. Изчакайте малко и опитайте отново."
    if request.headers.get("HX-Request"):
        return HTMLResponse(message, status_code=429)
    return HTMLResponse(
        f"<h2>429 — Твърде много заявки</h2><p>{message}</p><p><a href='/'>Начало</a></p>",
        status_code=429,
    )


@app.middleware("http")
async def attach_current_user(request: Request, call_next):
    """Load the authenticated user into request.state.user for all templates."""
    request.state.user = None
    if not request.url.path.startswith("/static"):
        try:
            uid = request.session.get("user_id")
            if uid:
                with db_session() as db:
                    u = db.get(User, int(uid))
                    if u and u.is_active:
                        db.expunge(u)   # detach before commit so attrs stay loaded
                        request.state.user = u
        except Exception:
            pass
    return await call_next(request)


@app.middleware("http")
async def csrf_protect(request: Request, call_next):
    """Synchronizer-token CSRF check for every state-changing request.
    Centralized here (rather than per-route) so a new POST route can't
    accidentally ship unprotected. See app/services/csrf.py for how the
    token reaches the client (meta tag + hx-headers + auto-injected
    hidden inputs, all in base.html) — nothing else needs to change per
    form/route as new ones are added.

    Registered before SessionMiddleware's add_middleware() call below so
    it ends up *inner* relative to it and request.session is already
    populated by the time this runs (session presence is required to have
    anything to compare the submitted token against)."""
    if request.method not in _CSRF_SAFE_METHODS and not request.url.path.startswith("/static"):
        submitted = request.headers.get("X-CSRF-Token")
        if submitted is None:
            try:
                body = await request.body()
                # BaseHTTPMiddleware hands the downstream app a *different*
                # Request instance than this one — consuming the body here
                # (to read the form) drains the one ASGI receive channel
                # both share, leaving nothing for FastAPI's own Form(...)
                # parsing to read. Re-arm it with the bytes already read so
                # the route handler still sees a full body.
                async def _replay_body() -> dict:
                    return {"type": "http.request", "body": body, "more_body": False}
                request._receive = _replay_body
                form = await request.form()
                submitted = form.get("csrf_token")
            except Exception:
                submitted = None
        if not verify_csrf_token(request, submitted):
            message = "Невалидна или изтекла сесийна заявка. Презаредете страницата и опитайте отново."
            if request.headers.get("HX-Request"):
                return HTMLResponse(message, status_code=403)
            return HTMLResponse(
                f"<h2>403 — Невалидна заявка (CSRF)</h2><p>{message}</p><p><a href='/'>Начало</a></p>",
                status_code=403,
            )
    return await call_next(request)


# SessionMiddleware must be outermost — registered AFTER attach_current_user
# so Starlette places it first in the request chain and populates request.session
# before attach_current_user reads it.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=86400 * 7,
    https_only=settings.session_https_only,
    same_site="lax",
)


@app.on_event("startup")
async def startup_cleanup() -> None:
    """Mark any runs that were left as 'running' from a previous process as 'interrupted'."""
    try:
        with db_session() as session:
            session.execute(text(
                "UPDATE scrape_runs SET status = 'interrupted', finished_at = now() "
                "WHERE status = 'running'"
            ))
        logger.info("Startup cleanup: stale 'running' scrape_runs marked as 'interrupted'")
    except Exception:
        logger.exception("Startup cleanup failed (non-fatal)")

app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(scrape.router)
app.include_router(listings.router)
app.include_router(comparables.router)
app.include_router(reports.router)
app.include_router(analytics.router)


@app.exception_handler(401)
async def auth_required_redirect(request: Request, exc):
    """Redirect unauthenticated users to login; return 401 for HTMX/XHR."""
    if request.headers.get("HX-Request") or request.headers.get("X-Requested-With"):
        return HTMLResponse("Необходима е идентификация", status_code=401)
    next_url = str(request.url.path)
    return RedirectResponse(url=f"/auth/login?next={next_url}", status_code=302)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled error on %s %s\n%s", request.method, request.url, tb)
    return HTMLResponse(
        content=(
            "<style>body{font-family:monospace;padding:2rem;background:#0f172a;color:#f8fafc}"
            "h2{color:#f87171}pre{background:#1e293b;padding:1rem;border-radius:6px;"
            "overflow:auto;font-size:0.8rem;color:#94a3b8}</style>"
            f"<h2>500 — Вътрешна грешка на сървъра</h2>"
            f"<p><a href='/' style='color:#60a5fa'>← Начало</a></p>"
            f"<pre>{tb}</pre>"
        ),
        status_code=500,
    )


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    return templates.TemplateResponse(request, "help.html", {})


@app.get("/legal/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    return templates.TemplateResponse(request, "legal/privacy_policy.html", {})


@app.get("/legal/terms", response_class=HTMLResponse)
async def terms(request: Request):
    return templates.TemplateResponse(request, "legal/terms.html", {})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    stats = {}
    try:
        db = next(get_db())
        rows = db.execute(text("""
            SELECT
                count(*) FILTER (WHERE deal_type_normalized = 'sale')   AS sale_count,
                count(*) FILTER (WHERE deal_type_normalized = 'rent')   AS rent_count,
                round(avg(price_per_sqm_model) FILTER (
                    WHERE deal_type_normalized = 'sale'
                )::numeric, 0)                                           AS avg_sale_ppsqm,
                round(avg(price_per_sqm_model) FILTER (
                    WHERE deal_type_normalized = 'rent'
                )::numeric, 2)                                           AS avg_rent_ppsqm,
                max(last_seen_at)::date                                  AS last_seen
            FROM listings
        """)).fetchone()
        if rows:
            stats = {
                "sale_count":    int(rows[0] or 0),
                "rent_count":    int(rows[1] or 0),
                "avg_sale_ppsqm": rows[2],
                "avg_rent_ppsqm": rows[3],
                "last_seen":     rows[4],
            }
        geo = db.execute(text("""
            SELECT geo_category, count(*) AS n
            FROM listings
            WHERE deal_type_normalized = 'sale' AND geo_category IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 6
        """)).fetchall()
        stats["geo"] = [(r[0], int(r[1])) for r in geo]
        db.close()
    except Exception:
        logger.exception("Stats query failed on home page")

    return templates.TemplateResponse(request, "home.html", {"stats": stats})
