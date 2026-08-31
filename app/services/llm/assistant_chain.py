"""Conversational multi-agent chat console (Tier 2, 2026-08-26; re-platformed
onto a LangGraph supervisor in Phase 11, 2026-08-28).

Persisted, multi-turn conversation scoped to one AppraisalReport (real or
is_scratch=True hypothetical -- identical treatment, see is_scratch's own
docstring). The actual tool-calling work now happens inside
orchestrator_graph.py's Supervisor + specialist nodes (income/market/
market_analysis/legal/auditor) -- this module owns everything AROUND that:
conversation/message persistence, the per-call token/cost ledger
(agent_llm_calls), and the live SSE progress callback. Mirrors
valuation_chain.py's streaming-for-live-progress pattern and "numbers come
from tools, never model arithmetic" guardrail, same as before Phase 11 --
only the internal loop moved, not the external contract.

Human-in-the-loop guardrail for this feature specifically: every specialist
can retrieve/compute freely (all read-only), but the one tool that could
change the report (propose_text_update) never writes anything itself -- it
returns a proposal that the UI renders as a card with an explicit "Apply"
button, wired to the same update_* functions the rest of the app already
uses. See tools.propose_text_update's own docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.orm import Session

from app.db.models import AgentConversation, AgentLlmCall, AgentMessage, AppraisalReport, ReportDocument
from app.services.llm.orchestrator_graph import build_orchestrator_graph
from app.services.llm.providers import build_sampling_kwargs, estimate_cost_usd, resolve_chat_model

MAX_OUTPUT_TOKENS = 3000
MAX_TOOL_ITERATIONS = 6

# Hard bounds for the chat console's user-facing max-reply-length / max-
# tool-steps controls (2026-08-26) -- clamp rather than trust the form
# value, since both directly drive spend (a huge max_output_tokens or an
# unbounded tool loop is a real cost/latency risk, not just a UX nicety).
MIN_OUTPUT_TOKENS = 200
MAX_OUTPUT_TOKENS_HARD_CAP = 8000
MAX_TOOL_ITERATIONS_HARD_CAP = 12

# Trailing-window cap on how much conversation history is actually sent to
# the model (Phase 12, 2026-08-31) -- found live that a conversation left
# to grow indefinitely (the pre-Phase-12 "one eternal thread" design) can
# reach 25k+ input tokens by turn 5-6, at which point the model routinely
# wants to write a longer answer than MAX_OUTPUT_TOKENS allows and gets cut
# off mid-sentence (see is_length_truncated). Starting new named
# conversations (new_conversation) is the primary fix -- a fresh topic
# gets a small context again -- this is just the safety net for a single
# conversation that still runs long. Deliberately a simple trailing count,
# not token-counting or LLM-based summarization: cheap, deterministic, and
# good enough as a first cut. Only affects what the MODEL sees -- the UI's
# own message list (_conversation_context in the routers) always shows the
# full history regardless.
MAX_HISTORY_MESSAGES = 40


@dataclass
class ChatProgress:
    status: str = "running"   # running | done | error
    step: str = "Мисля…"
    tokens_so_far: int = 0
    error: str | None = None


def get_or_create_conversation(db: Session, report_id, user_id: int, agent_type: str = "report_assistant") -> AgentConversation:
    """One conversation per (report, user) for the report-scoped assistant
    (v1's default agent_type -- reopening the assistant on the same report
    resumes the same thread rather than starting fresh each time). For a
    report-agnostic agent like the market analyst (Phase 10, 2026-08-28),
    pass report_id=None and agent_type="market_analyst" -- one persistent
    conversation per (user, agent_type) instead of per (user, report)."""
    query = db.query(AgentConversation).filter(
        AgentConversation.user_id == user_id, AgentConversation.agent_type == agent_type,
    )
    if report_id is not None:
        query = query.filter(AgentConversation.report_id == report_id)
    conv = query.order_by(AgentConversation.created_at.desc()).first()
    if conv:
        return conv
    conv = AgentConversation(report_id=report_id, user_id=user_id, agent_type=agent_type)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def new_conversation(db: Session, user_id: int, agent_type: str = "report_assistant", report_id=None) -> AgentConversation:
    """Always creates a fresh, separate conversation -- unlike get_or_create_
    conversation, which deliberately reuses the latest existing one. This is
    the one operation that was actually missing (Phase 12, 2026-08-31): the
    schema already supported multiple conversations per (user, agent_type,
    report_id), nothing here reuses or archives the old thread -- it just
    stays there, reachable via list_conversations()."""
    conv = AgentConversation(report_id=report_id, user_id=user_id, agent_type=agent_type)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def list_conversations(db: Session, user_id: int, agent_type: str = "report_assistant", report_id=None) -> list[AgentConversation]:
    """All of a user's conversations in this scope, most-recently-updated
    first -- backs the conversation switcher dropdown. Same (user_id,
    agent_type, report_id) filter shape as get_or_create_conversation,
    backed by the composite index added alongside this function."""
    query = db.query(AgentConversation).filter(
        AgentConversation.user_id == user_id, AgentConversation.agent_type == agent_type,
    )
    if report_id is not None:
        query = query.filter(AgentConversation.report_id == report_id)
    return query.order_by(AgentConversation.updated_at.desc()).all()


def get_conversation_for_user(db: Session, conversation_id, user_id: int) -> AgentConversation | None:
    """Ownership-checked lookup by id -- the ownership-check building block
    for _active_conversation()-style session resolution in the routers,
    mirroring comparable_service.get_report_for_user's role for reports."""
    return (
        db.query(AgentConversation)
        .filter(AgentConversation.id == conversation_id, AgentConversation.user_id == user_id)
        .first()
    )


def _trim_history(messages: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Keeps only the most recent `max_messages` -- see MAX_HISTORY_MESSAGES's
    own docstring for why. A no-op for any conversation shorter than the cap,
    which is the overwhelming majority of real conversations.

    A plain tail slice can cut a conversation between a tool-calling
    AIMessage and its ToolMessage result(s) -- the AIMessage falls before
    the cut, its result(s) after it, leaving one or more dangling
    ToolMessages with no matching tool_call in view. Every provider's API
    rejects that shape (an orphaned tool result), so any leading
    ToolMessages are additionally dropped from the trimmed slice."""
    if len(messages) <= max_messages:
        return messages
    trimmed = messages[-max_messages:]
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]
    return trimmed


def _load_langchain_messages(db: Session, conversation_id) -> list:
    rows = (
        db.query(AgentMessage)
        .filter(AgentMessage.conversation_id == conversation_id)
        .order_by(AgentMessage.created_at)
        .all()
    )
    messages: list = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content or ""))
        elif row.role == "assistant":
            messages.append(AIMessage(content=row.content or "", tool_calls=row.tool_calls or []))
        elif row.role == "tool":
            messages.append(ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id))
    return _trim_history(messages)


def _documents_note(db: Session, report_id) -> str:
    """Appended to the system prompt, listing this report's already-
    processed documents by name -- fixes a real gap found live (2026-08-26):
    the prompt only told the model to check documents "if the appraiser
    mentions" one, so a plain "write the property description" request
    (with no document named in that same message) produced a generic
    please-fill-this-in proposal despite 4 real, already-extracted
    documents sitting right there. Putting the actual filenames in front of
    the model every turn is more reliable than a buried instruction to
    remember they might exist -- it doesn't have to recall a rule, it can
    see the documents."""
    docs = (
        db.query(ReportDocument)
        .filter(ReportDocument.report_id == report_id, ReportDocument.status == "ready")
        .all()
    )
    if not docs:
        return ""
    lines = "\n".join(f"- {d.filename} ({d.document_type})" for d in docs)
    return (
        "\n\nНАЛИЧНИ КАЧЕНИ ДОКУМЕНТИ ЗА ТОЗИ ДОКЛАД (вече обработени, готови за четене "
        f"чрез list_documents/read_document):\n{lines}\n"
        "Прочети релевантните от тях, преди да пишеш описание на имота или друг текст за "
        "доклада -- дори ако оценителят не ги спомене изрично в текущото съобщение."
    )


def _persist_message(db: Session, conversation_id, role: str, content: str | None = None,
                      tool_calls: list | None = None, tool_call_id: str | None = None,
                      truncated: bool = False) -> AgentMessage:
    msg = AgentMessage(
        conversation_id=conversation_id, role=role, content=content,
        tool_calls=tool_calls, tool_call_id=tool_call_id, truncated=truncated,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def _persist_call_log(db: Session, conversation_id, provider: str, model_id: str, call_log: list[dict]) -> None:
    """Same shape as valuation_chain._persist_call_log, but keyed by
    conversation_id instead of ai_valuation_run_id -- agent_llm_calls
    supports both (see its own docstring). Best-effort, never fails the
    turn."""
    if not call_log:
        return
    try:
        for entry in call_log:
            cost = estimate_cost_usd(model_id, entry["input_tokens"], entry["output_tokens"], provider=provider)
            db.add(AgentLlmCall(
                conversation_id=conversation_id,
                call_label=entry["call_label"],
                provider=provider,
                model=model_id,
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                estimated_cost_usd=cost,
            ))
        db.commit()
    except Exception:
        db.rollback()


def run_assistant_turn(
    db: Session,
    conversation: AgentConversation,
    report: AppraisalReport,
    user_text: str,
    on_progress: Callable[[ChatProgress], None] | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    max_output_tokens: int | None = None,
    max_tool_iterations: int | None = None,
) -> ChatProgress:
    """Synchronous, blocking (real network calls) -- run on a background
    thread from the router, exactly like valuation_chain.generate_valuation_backbone.
    Persists every message (user/assistant/tool) as it happens, so a page
    reload or a later turn always sees the full history, not just this
    turn's result."""
    progress = ChatProgress()

    def emit():
        if on_progress:
            on_progress(progress)

    emit()

    if not conversation.title:
        conversation.title = user_text[:80]
    # Prefixes every call_log label with which user message this turn is
    # answering -- call_log itself is a fresh list per run_assistant_turn()
    # call, so without this every turn's LLM calls would restart labeling
    # at "turn_1", making multiple messages' calls indistinguishable in the
    # detailed breakdown (caught during live verification, 2026-08-26).
    prior_user_messages = (
        db.query(AgentMessage)
        .filter(AgentMessage.conversation_id == conversation.id, AgentMessage.role == "user")
        .count()
    )
    msg_seq = prior_user_messages + 1
    _persist_message(db, conversation.id, "user", content=user_text)

    provider, model_id = resolve_chat_model(provider, model)
    effective_max_tokens = MAX_OUTPUT_TOKENS if max_output_tokens is None else max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS_HARD_CAP, max_output_tokens))
    effective_max_iterations = MAX_TOOL_ITERATIONS if max_tool_iterations is None else max(1, min(MAX_TOOL_ITERATIONS_HARD_CAP, max_tool_iterations))
    sampling_kwargs = build_sampling_kwargs(
        provider, temperature=temperature, top_p=top_p, top_k=top_k,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, seed=seed,
    )

    call_log: list[dict] = []

    def on_progress_cb(step: str) -> None:
        progress.step = step
        emit()

    def on_message_cb(role: str, content: str | None, tool_calls: list | None, tool_call_id: str | None, truncated: bool = False) -> None:
        _persist_message(db, conversation.id, role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id, truncated=truncated)

    graph = build_orchestrator_graph(
        db=db, report=report, documents_note=_documents_note(db, report.id),
        provider=provider, model_id=model_id, sampling_kwargs=sampling_kwargs,
        max_tokens=effective_max_tokens, max_tool_iterations=effective_max_iterations,
        call_log=call_log, msg_seq=msg_seq, on_progress=on_progress_cb, on_message=on_message_cb,
    )
    initial_state = {
        "messages": _load_langchain_messages(db, conversation.id),
        "next_specialist": None, "direct_answer": None, "findings": {},
    }

    try:
        graph.invoke(initial_state)
    except Exception as exc:
        _persist_call_log(db, conversation.id, provider, model_id, call_log)
        progress.status = "error"
        progress.error = f"Грешка при отговора: {exc}"
        emit()
        return progress

    _persist_call_log(db, conversation.id, provider, model_id, call_log)
    db.commit()   # conversation.title / updated_at

    progress.status = "done"
    progress.step = "Готово"
    progress.tokens_so_far = sum(e["input_tokens"] + e["output_tokens"] for e in call_log)
    emit()
    return progress
