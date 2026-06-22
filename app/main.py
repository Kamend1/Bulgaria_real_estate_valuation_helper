import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.config import settings
from app.db.session import get_db, db_session
from app.routers import comparables, listings, scrape
from app.templating import templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_title)

app.mount("/static", StaticFiles(directory="static"), name="static")


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

app.include_router(scrape.router)
app.include_router(listings.router)
app.include_router(comparables.router)


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
