import asyncio
import json
import logging
import threading
import uuid
from datetime import date

from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AgentLlmCall, AiValuationRun, AppraisalReport, ComparablePool, ReportCompileRun, User
from app.db.session import db_session, get_db
from app.dependencies import require_auth as get_current_user
from app.rate_limit import limiter
from app.templating import templates
from app.services import avm_service, gis_service
from app.services.llm import generation_store
from app.services.llm.providers import get_default_model, list_available_models, list_configured_providers
from app.services.llm.report_compiler import DOMAIN_LABELS, run_compile
from app.services.llm.retriever import retrieve_comparables
from app.services.llm.valuation_chain import GenerationProgress, generate_valuation_backbone

logger = logging.getLogger(__name__)
from app.services.comparable_service import (
    ADJUSTMENT_FACTOR_LABELS,
    MAX_PINNED,
    add_to_pool,
    clear_pool,
    delete_user_report,
    export_excel,
    finalize_user_report,
    generate_docx,
    get_draft_reports_for_user,
    get_or_create_draft,
    get_purpose_options,
    get_pool_with_stats,
    get_report_for_user,
    get_user_reports,
    new_draft,
    remove_from_pool,
    toggle_pin,
    update_conclusion,
    update_income_valuation,
    update_income_market_rationale,
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
            "draft_reports": get_draft_reports_for_user(db, user.id),
            "switch_next": "/comparables/",
            "compile_domain_labels": DOMAIN_LABELS,
            "configured_providers": list_configured_providers(),
            "default_provider": settings.llm_default_provider,
            "default_model": get_default_model(settings.llm_default_provider),
            "models_by_provider": {key: list_available_models(key) for key, _ in list_configured_providers()},
        },
    )


# ── AI-assisted valuation (Phase 7, Tier 2 — retrieval only) ────────────────────

@router.get("/ai-suggestions", response_class=HTMLResponse)
@limiter.limit("20/hour")
async def ai_suggestions_panel(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Lazy-loaded (only fires when the panel is opened, not on every page
    view) since retrieve_comparables makes a real, billed embeddings API
    call. Rate-limited for the same reason. Read-only for now (Tier 2) --
    no add-to-pool action yet, this is purely for sanity-checking retrieval
    quality; Tier 3 adds the generation step on top of this."""
    report = _active_report(request, db, user)
    error = None
    suggestions: list[dict] = []
    try:
        # Real network call (embeddings API) -- offloaded so it can't stall
        # the event loop for other concurrent requests, same as the AVM/
        # cadastre calls in comparables_page above.
        suggestions = await run_in_threadpool(retrieve_comparables, db, report, 6)
    except RuntimeError as e:
        error = str(e)  # e.g. "OPENAI_API_KEY is not set -- add it to .env"
    # {provider_key: [(model_id, tier_label), ...]} -- only for providers
    # that are actually configured, so the combined dropdown never offers a
    # provider:model combo that would just fail with "API_KEY is not set".
    models_by_provider = {key: list_available_models(key) for key, _ in list_configured_providers()}
    return templates.TemplateResponse(
        request,
        "comparables/_ai_suggestions_panel.html",
        {
            "suggestions": suggestions, "error": error, "report": report,
            "configured_providers": list_configured_providers(),
            "default_provider": settings.llm_default_provider,
            "default_model": get_default_model(settings.llm_default_provider),
            "models_by_provider": models_by_provider,
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def _run_generation(
    run_id: str, report_id: str, provider: str | None, model: str | None, include_income: bool,
) -> None:
    """Background-thread target (daemon thread, NOT a FastAPI BackgroundTask
    -- see CLAUDE.md's note on why: those are bound to the request and can
    time out). Uses its own db_session() since a SQLAlchemy Session isn't
    thread-safe to share with the request that spawned this thread."""
    def on_progress(p: GenerationProgress) -> None:
        generation_store.update(run_id, p)

    try:
        with db_session() as db:
            report = db.get(AppraisalReport, uuid.UUID(report_id))
            if report is None:
                generation_store.update(run_id, GenerationProgress(status="error", error="Докладът не е намерен."))
                return
            generate_valuation_backbone(
                db, report, on_progress=on_progress, provider=provider, model=model,
                include_income=include_income,
            )
    except Exception:
        logger.exception("AI valuation generation failed (run_id=%s)", run_id)
        generation_store.update(run_id, GenerationProgress(status="error", error="Неочаквана грешка при генерирането. Опитайте отново."))


@router.post("/ai-generate", response_class=HTMLResponse)
@limiter.limit("10/hour")
async def ai_generate_start(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    provider_model: str = Form(""),
    include_income: bool = Form(False),
):
    # provider_model encodes "provider:model" from a single combined
    # dropdown (see _ai_suggestions_panel.html) -- simpler than two selects
    # with JS show/hide per provider, and degrades gracefully (falls back
    # to settings defaults) if the field is empty or malformed.
    provider, _, model = provider_model.partition(":")
    provider, model = (provider or None), (model or None)
    report = _active_report(request, db, user)
    run_id = generation_store.create_run()
    generation_store.cleanup_old()
    thread = threading.Thread(
        target=_run_generation, args=(run_id, str(report.id), provider, model, include_income), daemon=True,
    )
    thread.start()
    return templates.TemplateResponse(
        request, "comparables/_ai_generation_progress.html", {"run_id": run_id},
    )


@router.get("/ai-generate/progress/{run_id}")
async def ai_generate_progress_sse(run_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            progress = generation_store.get(run_id)
            if progress is None:
                yield _sse("done", {"status": "error", "error": "Генерацията не е намерена (може да е изтекла)."})
                break

            yield _sse("progress", {
                "status": progress.status, "step": progress.step,
                "tokens_so_far": progress.tokens_so_far,
            })

            if progress.status in ("done", "error"):
                yield _sse("done", {"status": progress.status, "error": progress.error})
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _combined_texts(r: dict) -> tuple[str, str | None]:
    """Text for the "insert into report" actions -- target the EXISTING
    submarket_rationale/income_market_rationale textareas/save-forms
    already on the page rather than a new backend route: nothing is
    persisted until the appraiser reviews it in that textarea and clicks
    its pre-existing "Запази" button themselves. Shared between a
    just-finished generation (ai_generate_result) and past runs
    (ai_history) so both render identically.

    A failed run's `output` is a minimal {"failed": True, "error": ...} dict
    (see valuation_chain.py's cost-visibility audit, 2026-08-25) -- none of
    the narrative keys below exist for it, so this returns empty text rather
    than KeyError-ing; the history template shows a distinct failed-run
    summary instead of trying to render a narrative that was never written."""
    if r.get("failed"):
        return "", None
    combined_text = (
        f"{r.get('comparable_selection_rationale', '')}\n\n"
        f"Коментар по сравними:\n{r.get('comparable_commentary', '')}\n\n"
        f"{r.get('value_reasoning', '')}\n\n"
        f"Ограничения: {r.get('caveats', '')}"
    )
    combined_income_text = None
    income = r.get("income")
    if income and income.get("available"):
        combined_income_text = (
            f"{income.get('rationale', '')}\n\n"
            f"Коментар по наемни сравними:\n{income.get('commentary', '')}\n\n"
            f"{income.get('reasoning', '')}\n\n"
            f"Ограничения: {income.get('caveats', '')}"
        )
    return combined_text, combined_income_text


@router.get("/ai-generate/result/{run_id}", response_class=HTMLResponse)
async def ai_generate_result(request: Request, run_id: str):
    progress = generation_store.get(run_id)
    if progress is None or progress.status != "done":
        return HTMLResponse('<p class="hint">Резултатът вече не е наличен.</p>')

    r = progress.result
    combined_text, combined_income_text = _combined_texts(r)
    return templates.TemplateResponse(
        request, "comparables/_ai_generation_result.html",
        {"result": r, "combined_text": combined_text, "combined_income_text": combined_income_text},
    )


# ── Report Compiler (Phase 13, 2026-09-01) ─────────────────────────────────────
# Runs several specialists sequentially against the whole report as an
# explicit, standalone action -- not a chat question. See
# app/services/llm/report_compiler.py's module docstring for why this is
# sequential, not parallel, and how it reuses the exact same specialist
# tools/prompts as the AI Assistant chat.

_COMPILE_DOMAINS = ("income", "market", "market_analysis", "legal")


def _run_compile_thread(run_id: str, report_id: str, domains: list[str], provider: str | None, model: str | None) -> None:
    """Background-thread target (daemon thread, NOT a FastAPI BackgroundTask
    -- see CLAUDE.md's note on why). Own db_session(), never the request's
    -- same reasoning as _run_generation above, doubly important here since
    the Idea F audit found specialist tools are NOT safe to share a Session
    across threads."""
    def on_progress(step: str) -> None:
        generation_store.update(run_id, GenerationProgress(status="running", step=step))

    try:
        with db_session() as db:
            report = db.get(AppraisalReport, uuid.UUID(report_id))
            run = db.get(ReportCompileRun, uuid.UUID(run_id))
            if report is None or run is None:
                generation_store.update(run_id, GenerationProgress(status="error", error="Докладът или заявката не са намерени."))
                return
            run_compile(db, report, run, domains, provider, model, on_progress=on_progress)
            generation_store.update(run_id, GenerationProgress(status="done"))
    except Exception:
        logger.exception("Report compile failed (run_id=%s)", run_id)
        generation_store.update(run_id, GenerationProgress(status="error", error="Неочаквана грешка при компилирането. Опитайте отново."))


@router.post("/compile", response_class=HTMLResponse)
@limiter.limit("10/hour")
async def compile_report_start(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    domains: list[str] = Form([]),
    provider_model: str = Form(""),
):
    report = _active_report(request, db, user)
    selected = [d for d in domains if d in _COMPILE_DOMAINS]
    if not selected:
        return HTMLResponse('<p class="hint">Избери поне един специалист.</p>', status_code=400)

    provider, _, model = provider_model.partition(":")
    provider, model = (provider or None), (model or None)

    run = ReportCompileRun(report_id=report.id, requested_domains=selected, status="running")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = str(run.id)

    generation_store.update(run_id, GenerationProgress(status="running", step="Стартиране…"))
    thread = threading.Thread(
        target=_run_compile_thread, args=(run_id, str(report.id), selected, provider, model), daemon=True,
    )
    thread.start()
    return templates.TemplateResponse(
        request, "comparables/_compile_progress.html", {"run_id": run_id},
    )


@router.get("/compile/progress/{run_id}")
async def compile_progress_sse(run_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            progress = generation_store.get(run_id)
            if progress is None:
                yield _sse("done", {"status": "error", "error": "Компилирането не е намерено (може да е изтекло)."})
                break

            yield _sse("progress", {"status": progress.status, "step": progress.step})

            if progress.status in ("done", "error"):
                yield _sse("done", {"status": progress.status, "error": progress.error})
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/compile/result/{run_id}", response_class=HTMLResponse)
async def compile_result(
    request: Request,
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Results are read from the DB (report_compile_runs.results), not the
    # in-memory generation_store -- a compile run can take a while
    # (several specialists, sequential), and the durable row is what
    # survives generation_store's cleanup_old() eviction.
    run = db.get(ReportCompileRun, run_id)
    if run is None:
        return HTMLResponse('<p class="hint">Компилирането не е намерено.</p>', status_code=404)
    report = get_report_for_user(db, run.report_id, user.id)
    if report is None:
        raise HTTPException(status_code=404)
    if run.status != "done":
        return HTMLResponse(f'<p class="hint">Статус: {run.status}. {run.error_message or ""}</p>')
    return templates.TemplateResponse(
        request, "comparables/_compile_result.html",
        {"run": run, "results": run.results or {}, "domain_labels": DOMAIN_LABELS},
    )


@router.get("/ai-history", response_class=HTMLResponse)
@limiter.limit("30/hour")
async def ai_history(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Persisted past AI generations for the active report (Tier 6 audit
    fix, 2026-08-25): ai_valuation_runs.output already held the full
    narrative, but nothing ever read it back -- once the ephemeral
    generation_store entry aged out or the appraiser navigated away without
    clicking "Insert", the generated text was effectively gone even though
    it was sitting in the DB. Lazy-loaded like the suggestions panel (no
    reason to query on every page view)."""
    report = _active_report(request, db, user)
    runs = (
        db.query(AiValuationRun)
        .filter(AiValuationRun.report_id == report.id)
        .order_by(AiValuationRun.created_at.desc())
        .all()
    )
    # Per-call breakdown (Tier 1, 2026-08-26) -- one query for all runs'
    # calls rather than N+1, grouped back onto each item below.
    run_ids = [run.id for run in runs]
    calls_by_run: dict = {}
    if run_ids:
        calls = (
            db.query(AgentLlmCall)
            .filter(AgentLlmCall.ai_valuation_run_id.in_(run_ids))
            .order_by(AgentLlmCall.created_at.asc())
            .all()
        )
        for call in calls:
            calls_by_run.setdefault(call.ai_valuation_run_id, []).append(call)

    items = []
    for run in runs:
        r = run.output or {}
        combined_text, combined_income_text = _combined_texts(r)
        items.append({
            "id": str(run.id), "created_at": run.created_at,
            "provider": run.provider, "model": run.model,
            "result": r, "combined_text": combined_text, "combined_income_text": combined_income_text,
            "calls": calls_by_run.get(run.id, []),
        })
    return templates.TemplateResponse(request, "comparables/_ai_history.html", {"items": items})


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
    sale_price_per_sqm: str = Form(""),
    expenses_pct: str = Form(""),
    vacancy_pct: str = Form(""),
    growth_pct: str = Form(""),
    period_years: str = Form(""),
    terminal_cap_rate_pct: str = Form(""),
    method: str = Form("direct"),
    subject_area_sqm: str = Form(""),
    source: str = Form("manual"),
    discount_rate_pct: str = Form(""),
):
    # source is accepted from the form (unlike save-sales' hardcoded
    # "manual") because it's purely a provenance label here, not a trust
    # boundary -- the actual number is always freshly computed server-side
    # by update_income_valuation() from the raw assumption inputs below,
    # never taken as a client-submitted final figure. The AI-confirm button
    # in _ai_generation_result.html posts source="ai" with the same
    # assumptions_used the appraiser already reviewed on screen.
    def _f(v): return float(v) if v.strip() else None
    def _i(v):
        try:
            return int(v) if v.strip() else None
        except ValueError:
            return None
    report = _active_report(request, db, user)
    update_income_valuation(
        db, report.id,
        rent_per_sqm_month=_f(rent_per_sqm_month),
        cap_rate_pct=_f(cap_rate_pct),
        method=method,
        source=source,
        sale_price_per_sqm=_f(sale_price_per_sqm),
        expenses_pct=_f(expenses_pct),
        vacancy_pct=_f(vacancy_pct),
        growth_pct=_f(growth_pct),
        period_years=_i(period_years),
        terminal_cap_rate_pct=_f(terminal_cap_rate_pct),
        subject_area_sqm=_f(subject_area_sqm),
        discount_rate_pct=_f(discount_rate_pct),
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


@router.post("/save-income-rationale")
async def save_income_rationale(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    income_market_rationale: str = Form(""),
):
    report = _active_report(request, db, user)
    update_income_market_rationale(db, report.id, income_market_rationale.strip())
    return RedirectResponse(url="/comparables/#income-panel", status_code=303)


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
