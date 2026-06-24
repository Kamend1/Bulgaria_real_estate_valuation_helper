import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.db.models import User
from app.db.session import get_db, db_session
from app.routers import analytics, comparables, listings, reports, scrape
from app.routers import auth as auth_router
from app.routers import admin as admin_router
from app.templating import templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_title)

app.mount("/static", StaticFiles(directory="static"), name="static")


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


# SessionMiddleware must be outermost — registered AFTER attach_current_user
# so Starlette places it first in the request chain and populates request.session
# before attach_current_user reads it.
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, max_age=86400 * 7)


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
