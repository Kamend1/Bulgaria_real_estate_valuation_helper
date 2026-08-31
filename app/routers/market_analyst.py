"""Market analyst chat console (Phase 10, 2026-08-28) -- second,
report-agnostic AI agent for free-form market research. Structurally
mirrors app/routers/assistant.py closely (same SSE-progress/background-
thread/document-upload patterns), but every route here is scoped only to
the current user, never to an active report -- there is no _active_report
call anywhere in this file.
"""
import asyncio
import json
import logging
import threading
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AgentConversation, AgentLlmCall, AgentMessage, MarketDocument, User
from app.db.session import db_session, get_db
from app.dependencies import require_auth as get_current_user
from app.rate_limit import limiter
from app.services import market_documents as market_documents_service
from app.services.llm import chat_store
from app.services.llm.analyst_chain import (
    ChatProgress, get_or_create_analyst_conversation, list_analyst_conversations,
    new_analyst_conversation, run_analyst_turn,
)
from app.services.llm.assistant_chain import get_conversation_for_user, rename_conversation
from app.services.llm.providers import get_default_model, get_sampling_capabilities, list_available_models, list_configured_providers
from app.templating import templates

_MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyst", tags=["market_analyst"])


def _conversation_context(db: Session, conversation: AgentConversation) -> dict:
    """Same shape as assistant.py's own _conversation_context -- kept as a
    separate copy (not imported) since this agent has no propose_text_update-
    style proposal card to special-case, and diverging that one difference
    from a shared helper isn't worth the indirection."""
    rows = (
        db.query(AgentMessage)
        .filter(AgentMessage.conversation_id == conversation.id)
        .order_by(AgentMessage.created_at)
        .all()
    )
    messages = []
    for row in rows:
        item = {"role": row.role, "content": row.content, "tool_calls": row.tool_calls, "created_at": row.created_at, "truncated": row.truncated}
        if row.role == "tool" and row.content:
            try:
                item["parsed"] = json.loads(row.content)
            except Exception:
                item["parsed"] = None
        messages.append(item)

    calls = (
        db.query(AgentLlmCall)
        .filter(AgentLlmCall.conversation_id == conversation.id)
        .order_by(AgentLlmCall.created_at)
        .all()
    )
    total_in = sum(c.input_tokens for c in calls)
    total_out = sum(c.output_tokens for c in calls)
    total_cost = sum((c.estimated_cost_usd or 0) for c in calls)
    return {
        "messages": messages,
        "calls": calls,
        "total_tokens": total_in + total_out,
        "total_cost": total_cost,
    }


def _active_analyst_conversation(request: Request, db: Session, user: User) -> AgentConversation:
    """Session-selected market-analyst conversation (Phase 12, 2026-08-31)
    -- mirrors assistant.py's _active_conversation, minus the report
    re-validation (this agent has no report to re-check against)."""
    cid_str = request.session.get("active_analyst_conversation_id")
    if cid_str:
        try:
            conv = get_conversation_for_user(db, uuid.UUID(cid_str), user.id)
            if conv and conv.agent_type == "market_analyst":
                return conv
        except Exception:
            pass
    conv = get_or_create_analyst_conversation(db, user.id)
    request.session["active_analyst_conversation_id"] = str(conv.id)
    return conv


@router.get("/", response_class=HTMLResponse)
async def analyst_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conversation = _active_analyst_conversation(request, db, user)
    ctx = _conversation_context(db, conversation)
    docs = db.query(MarketDocument).order_by(MarketDocument.created_at.desc()).all()
    return templates.TemplateResponse(
        request, "market_analyst.html",
        {
            "conversation": conversation,
            "conversations": list_analyst_conversations(db, user.id),
            "conversation_open_url_prefix": "/analyst/conversations",
            "conversation_new_url": "/analyst/conversations/new",
            "configured_providers": list_configured_providers(),
            "default_provider": settings.llm_default_provider,
            "default_model": get_default_model(settings.llm_default_provider),
            "models_by_provider": {key: list_available_models(key) for key, _ in list_configured_providers()},
            "sampling_capabilities": get_sampling_capabilities(),
            "docs": docs,
            "document_type_labels": market_documents_service.DOCUMENT_TYPE_LABELS,
            **ctx,
        },
    )


@router.post("/conversations/new")
async def new_conversation_route(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = new_analyst_conversation(db, user.id)
    request.session["active_analyst_conversation_id"] = str(conv.id)
    return RedirectResponse(url="/analyst/", status_code=303)


@router.post("/conversations/{conversation_id}/open")
async def open_conversation_route(
    conversation_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = get_conversation_for_user(db, conversation_id, user.id)
    if not conv or conv.agent_type != "market_analyst":
        raise HTTPException(status_code=404)
    request.session["active_analyst_conversation_id"] = str(conv.id)
    return RedirectResponse(url="/analyst/", status_code=303)


@router.post("/conversations/{conversation_id}/rename")
async def rename_conversation_route(
    conversation_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    title: str = Form(...),
):
    conv = rename_conversation(db, conversation_id, user.id, title)
    if not conv:
        raise HTTPException(status_code=404)
    return RedirectResponse(url="/analyst/", status_code=303)


def _run_turn(
    turn_id: str, conversation_id: str, user_text: str,
    provider: str | None, model: str | None, sampling: dict, max_output_tokens: int | None, max_tool_iterations: int | None,
) -> None:
    """Background-thread target (daemon thread, not a FastAPI BackgroundTask
    -- see CLAUDE.md's note on why). Own db_session(), never the request's."""
    def on_progress(p: ChatProgress) -> None:
        chat_store.update(turn_id, p)

    try:
        with db_session() as db:
            conversation = db.get(AgentConversation, uuid.UUID(conversation_id))
            if conversation is None:
                chat_store.update(turn_id, ChatProgress(status="error", error="Разговорът не е намерен."))
                return
            run_analyst_turn(
                db, conversation, user_text, on_progress=on_progress, provider=provider, model=model,
                max_output_tokens=max_output_tokens, max_tool_iterations=max_tool_iterations, **sampling,
            )
    except Exception:
        logger.exception("Analyst chat turn failed (turn_id=%s)", turn_id)
        chat_store.update(turn_id, ChatProgress(status="error", error="Неочаквана грешка. Опитайте отново."))


def _parse_optional_float(raw: str) -> float | None:
    raw = (raw or "").strip()
    return float(raw) if raw else None


def _parse_optional_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    return int(float(raw)) if raw else None


@router.post("/send", response_class=HTMLResponse)
@limiter.limit("30/hour")
async def send_message(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    message: str = Form(...),
    provider_model: str = Form(""),
    temperature: str = Form(""),
    top_p: str = Form(""),
    top_k: str = Form(""),
    frequency_penalty: str = Form(""),
    presence_penalty: str = Form(""),
    seed: str = Form(""),
    max_output_tokens: str = Form(""),
    max_tool_iterations: str = Form(""),
):
    message = message.strip()
    if not message:
        return HTMLResponse("", status_code=204)

    conversation = _active_analyst_conversation(request, db, user)

    provider, _, model = provider_model.partition(":")
    provider, model = (provider or None), (model or None)

    try:
        sampling = {
            "temperature": _parse_optional_float(temperature),
            "top_p": _parse_optional_float(top_p),
            "top_k": _parse_optional_int(top_k),
            "frequency_penalty": _parse_optional_float(frequency_penalty),
            "presence_penalty": _parse_optional_float(presence_penalty),
            "seed": _parse_optional_int(seed),
        }
        max_output_tokens_val = _parse_optional_int(max_output_tokens)
        max_tool_iterations_val = _parse_optional_int(max_tool_iterations)
    except ValueError:
        sampling = {k: None for k in ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "seed")}
        max_output_tokens_val = None
        max_tool_iterations_val = None

    turn_id = chat_store.create_turn()
    chat_store.cleanup_old()
    thread = threading.Thread(
        target=_run_turn,
        args=(turn_id, str(conversation.id), message, provider, model, sampling, max_output_tokens_val, max_tool_iterations_val),
        daemon=True,
    )
    thread.start()
    return templates.TemplateResponse(
        request, "market_analyst/_progress.html", {"turn_id": turn_id, "user_message": message},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


@router.get("/progress/{turn_id}")
async def turn_progress_sse(turn_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            progress = chat_store.get(turn_id)
            if progress is None:
                yield _sse("done", {"status": "error", "error": "Разговорът не е намерен (може да е изтекъл)."})
                break

            yield _sse("progress", {
                "status": progress.status, "step": progress.step,
                "tokens_so_far": progress.tokens_so_far,
            })

            if progress.status in ("done", "error"):
                yield _sse("done", {"status": progress.status, "error": progress.error})
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/messages", response_class=HTMLResponse)
async def messages_partial(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conversation = _active_analyst_conversation(request, db, user)
    ctx = _conversation_context(db, conversation)
    return templates.TemplateResponse(request, "market_analyst/_messages.html", ctx)


# ── Document library (shared, not report-scoped) ───────────────────────────────

@router.get("/documents", response_class=HTMLResponse)
async def list_documents(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    docs = db.query(MarketDocument).order_by(MarketDocument.created_at.desc()).all()
    return templates.TemplateResponse(
        request, "market_analyst/_documents.html",
        {"docs": docs, "document_type_labels": market_documents_service.DOCUMENT_TYPE_LABELS},
    )


@router.post("/documents/upload", response_class=HTMLResponse)
@limiter.limit("20/hour")
async def upload_document(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    document_type: str = Form(...),
    file: UploadFile = File(...),
):
    if document_type not in market_documents_service.DOCUMENT_TYPE_LABELS:
        return HTMLResponse('<p class="hint">Непознат тип документ.</p>', status_code=400)
    file_bytes = await file.read()
    if not file_bytes:
        return HTMLResponse('<p class="hint">Празен файл.</p>', status_code=400)
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        return HTMLResponse('<p class="hint">Файлът е твърде голям (макс. 15 MB).</p>', status_code=400)

    storage_path = market_documents_service.save_upload(user.id, file.filename or "document", file_bytes)
    doc = MarketDocument(
        uploaded_by=user.id, filename=file.filename or "document",
        document_type=document_type, storage_path=storage_path, mime_type=file.content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Synchronous for v1, same reasoning as documents.py's report-scoped
    # upload -- typical extraction is a few seconds, well inside a normal
    # request timeout.
    market_documents_service.extract_document(db, doc)
    db.refresh(doc)

    doc_count = db.query(MarketDocument).count()
    item_html = templates.env.get_template("market_analyst/_document_item.html").render(
        {"doc": doc, "document_type_labels": market_documents_service.DOCUMENT_TYPE_LABELS, "request": request},
    )
    count_oob_html = f'<span id="analyst-doc-count" hx-swap-oob="true">{doc_count}</span>'
    return HTMLResponse(item_html + count_oob_html)
