"""Multi-agent chat console (Tier 2, 2026-08-26) -- conversational
alternative to the comparables page's form-driven AI panel. Scoped to the
active report (same _active_report resolution the rest of the app uses).
"""
import asyncio
import json
import logging
import re
import threading
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AgentConversation, AgentLlmCall, AgentMessage, ReportDocument, User
from app.db.session import db_session, get_db
from app.dependencies import require_auth as get_current_user
from app.rate_limit import limiter
from app.routers.comparables import _active_report
from app.services import documents as documents_service
from app.services.comparable_service import get_draft_reports_for_user, update_income_market_rationale, update_submarket_rationale, update_subject
from app.services.llm import chat_store
from app.services.llm.assistant_chain import (
    ChatProgress, get_conversation_for_user, get_or_create_conversation,
    list_conversations, new_conversation, rename_conversation, run_assistant_turn,
)
from app.services.llm.providers import get_default_model, get_sampling_capabilities, list_available_models, list_configured_providers
from app.templating import templates

_MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant", tags=["assistant"])

# field -> how to actually persist an applied proposal, reusing the exact
# same functions the rest of the app's own save forms call -- an applied
# proposal is indistinguishable in the DB from a manually-typed save.
_APPLY_HANDLERS = {
    "subject_description": lambda db, report_id, text: update_subject(db, report_id, {"subject_description": text}),
    "submarket_rationale": update_submarket_rationale,
    "income_market_rationale": update_income_market_rationale,
}


def _natural_sort_key(filename: str) -> list:
    """Appraisers commonly number their own uploaded files ("01. Нотариален
    акт...", "02. Скица...") to fix a reading order for the report's
    document annex -- but the list was sorted by created_at DESC (newest
    upload first), which only matches that numbering if the files happen to
    be uploaded in the same order they're numbered (found live 2026-08-26:
    a batch upload out of numbering order made the list look "randomly"
    ordered). Sorting by filename instead respects the appraiser's own
    numbering. Splits into digit/non-digit runs so "001" sorts as 1, before
    "02" as 2 -- a plain string sort would put "001" before "02"
    lexicographically wrong ("0" < "2") but also "1" after "02" wrong."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", filename)]


def _sorted_documents(db: Session, report_id) -> list[ReportDocument]:
    docs = db.query(ReportDocument).filter(ReportDocument.report_id == report_id).all()
    docs.sort(key=lambda d: _natural_sort_key(d.filename or ""))
    return docs


def _conversation_context(db: Session, conversation: AgentConversation) -> dict:
    """Message history + running token/cost total, shared by the main page
    and the post-turn partial refresh. Tool-message content is stored as a
    JSON string (see assistant_chain._persist_message) -- parsed here into
    plain dicts so the template never needs a from_json Jinja filter, and
    can special-case a propose_text_update result into a confirm card."""
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


def _active_conversation(request: Request, db: Session, user: User, report) -> AgentConversation:
    """Session-selected conversation for the report-scoped assistant
    (Phase 12, 2026-08-31) -- mirrors comparables._active_report's shape
    exactly: read session id -> ownership check -> fallback + write back.
    Also re-validated against the CURRENTLY active report: if the
    appraiser switches reports via _report_switcher.html, a conversation
    id left over from the previous report is no longer a valid choice and
    falls back to that new report's own default conversation instead."""
    cid_str = request.session.get("active_assistant_conversation_id")
    if cid_str:
        try:
            conv = get_conversation_for_user(db, uuid.UUID(cid_str), user.id)
            if conv and conv.agent_type == "report_assistant" and conv.report_id == report.id:
                return conv
        except Exception:
            pass
    conv = get_or_create_conversation(db, report.id, user.id)
    request.session["active_assistant_conversation_id"] = str(conv.id)
    return conv


@router.get("/", response_class=HTMLResponse)
async def assistant_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    conversation = _active_conversation(request, db, user, report)
    ctx = _conversation_context(db, conversation)
    docs = _sorted_documents(db, report.id)
    return templates.TemplateResponse(
        request, "assistant.html",
        {
            "report": report,
            "conversation": conversation,
            "conversations": list_conversations(db, user.id, agent_type="report_assistant", report_id=report.id),
            "conversation_open_url_prefix": "/assistant/conversations",
            "conversation_new_url": "/assistant/conversations/new",
            "configured_providers": list_configured_providers(),
            "default_provider": settings.llm_default_provider,
            "default_model": get_default_model(settings.llm_default_provider),
            "models_by_provider": {key: list_available_models(key) for key, _ in list_configured_providers()},
            "sampling_capabilities": get_sampling_capabilities(),
            "docs": docs,
            "document_type_labels": documents_service.DOCUMENT_TYPE_LABELS,
            "draft_reports": get_draft_reports_for_user(db, user.id),
            "switch_next": "/assistant/",
            **ctx,
        },
    )


@router.post("/conversations/new")
async def new_conversation_route(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    conv = new_conversation(db, user.id, agent_type="report_assistant", report_id=report.id)
    request.session["active_assistant_conversation_id"] = str(conv.id)
    return RedirectResponse(url="/assistant/", status_code=303)


@router.post("/conversations/{conversation_id}/open")
async def open_conversation_route(
    conversation_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conv = get_conversation_for_user(db, conversation_id, user.id)
    if not conv or conv.agent_type != "report_assistant":
        raise HTTPException(status_code=404)
    request.session["active_assistant_conversation_id"] = str(conv.id)
    # Keep the two selectors consistent -- opening a conversation that
    # belongs to a different report than the currently active one also
    # switches the active report, so /comparables/ and this page agree.
    if conv.report_id:
        request.session["active_report_id"] = str(conv.report_id)
    return RedirectResponse(url="/assistant/", status_code=303)


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
    return RedirectResponse(url="/assistant/", status_code=303)


def _run_turn(
    turn_id: str, conversation_id: str, report_id: str, user_text: str,
    provider: str | None, model: str | None, sampling: dict, max_output_tokens: int | None, max_tool_iterations: int | None,
) -> None:
    """Background-thread target (daemon thread, not a FastAPI BackgroundTask
    -- see CLAUDE.md's note on why). Own db_session(), never the request's."""
    def on_progress(p: ChatProgress) -> None:
        chat_store.update(turn_id, p)

    try:
        with db_session() as db:
            conversation = db.get(AgentConversation, uuid.UUID(conversation_id))
            report = conversation.report if conversation else None
            if conversation is None or report is None:
                chat_store.update(turn_id, ChatProgress(status="error", error="Разговорът не е намерен."))
                return
            run_assistant_turn(
                db, conversation, report, user_text, on_progress=on_progress, provider=provider, model=model,
                max_output_tokens=max_output_tokens, max_tool_iterations=max_tool_iterations, **sampling,
            )
    except Exception:
        logger.exception("Assistant chat turn failed (turn_id=%s)", turn_id)
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

    report = _active_report(request, db, user)
    conversation = _active_conversation(request, db, user, report)

    provider, _, model = provider_model.partition(":")
    provider, model = (provider or None), (model or None)

    # Form fields arrive as strings (blank = "use provider/app default");
    # parsing/clamping to actual types happens once here, not scattered
    # across the background thread + assistant_chain -- invalid input
    # (e.g. a non-numeric paste) falls back to None (provider default)
    # rather than 500ing the request.
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
        args=(turn_id, str(conversation.id), str(report.id), message, provider, model, sampling, max_output_tokens_val, max_tool_iterations_val),
        daemon=True,
    )
    thread.start()
    return templates.TemplateResponse(
        request, "assistant/_progress.html", {"turn_id": turn_id, "user_message": message},
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
    """Re-rendered after a turn completes -- simplest way to show the new
    assistant/tool messages (incl. any proposal cards) without hand-rolling
    incremental DOM patching for a first version."""
    report = _active_report(request, db, user)
    conversation = _active_conversation(request, db, user, report)
    ctx = _conversation_context(db, conversation)
    return templates.TemplateResponse(request, "assistant/_messages.html", ctx)


@router.post("/apply-proposal", response_class=HTMLResponse)
async def apply_proposal(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    field: str = Form(...),
    text: str = Form(...),
):
    """Applies a propose_text_update tool result -- the appraiser has
    already reviewed the text (rendered in the chat), this just persists it
    via the SAME update_* function the field's own save form elsewhere in
    the app uses. Never called automatically by the assistant itself."""
    handler = _APPLY_HANDLERS.get(field)
    if handler is None:
        return HTMLResponse('<span class="hint">Непознато поле.</span>', status_code=400)
    report = _active_report(request, db, user)
    handler(db, report.id, text)
    return HTMLResponse('<span class="badge badge-ok" style="display:inline-block">Приложено ✓</span>')


# ── Documents (Tier 3, 2026-08-26) ────────────────────────────────────────────

@router.get("/documents", response_class=HTMLResponse)
async def list_documents(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = _active_report(request, db, user)
    docs = _sorted_documents(db, report.id)
    return templates.TemplateResponse(
        request, "assistant/_documents.html",
        {"docs": docs, "document_type_labels": documents_service.DOCUMENT_TYPE_LABELS},
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
    if document_type not in documents_service.DOCUMENT_TYPE_LABELS:
        return HTMLResponse('<p class="hint">Непознат тип документ.</p>', status_code=400)
    file_bytes = await file.read()
    if not file_bytes:
        return HTMLResponse('<p class="hint">Празен файл.</p>', status_code=400)
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        return HTMLResponse('<p class="hint">Файлът е твърде голям (макс. 15 MB).</p>', status_code=400)

    report = _active_report(request, db, user)
    storage_path, _ = documents_service.save_upload(
        report.id, user.id, file.filename or "document", document_type, file_bytes, file.content_type,
    )
    doc = ReportDocument(
        report_id=report.id, uploaded_by=user.id, filename=file.filename or "document",
        document_type=document_type, storage_path=storage_path, mime_type=file.content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Synchronous for v1 (see documents.py's module docstring) -- typical
    # extraction is a few seconds, well inside a normal request timeout;
    # avoids standing up a second background-thread+SSE mechanism alongside
    # the chat turns' one just for this.
    documents_service.extract_document(db, doc)
    db.refresh(doc)

    doc_count = db.query(ReportDocument).filter(ReportDocument.report_id == report.id).count()
    item_html = templates.env.get_template("assistant/_document_item.html").render(
        {"doc": doc, "document_type_labels": documents_service.DOCUMENT_TYPE_LABELS, "request": request},
    )
    # hx-swap-oob: the visible upload response only ever targets #doc-list
    # (see _documents.html's form), so the "Документи (N)" count in the
    # <summary> above it -- outside that swap target -- silently never
    # updated after an upload (real bug, caught live 2026-08-26: uploaded
    # docs appeared in the list but the header still read "Документи (0)").
    # An out-of-band span lets htmx patch that count elsewhere in the DOM
    # in the same response, no second request needed.
    count_oob_html = f'<span id="doc-count" hx-swap-oob="true">{doc_count}</span>'
    return HTMLResponse(item_html + count_oob_html)
