"""Conversational multi-agent chat console (Tier 2, 2026-08-26).

Generalizes valuation_chain.py's one-shot streaming tool-calling loop into
a persisted, multi-turn conversation: same streaming-for-live-progress
pattern, same "numbers come from tools, never model arithmetic" guardrail,
same per-call token/cost ledger (agent_llm_calls, via conversation_id this
time instead of ai_valuation_run_id) -- but now scoped to an ongoing chat
instead of a single generate-and-done call.

Human-in-the-loop guardrail for this feature specifically: the assistant
can retrieve/compute freely (all read-only), but the one tool that could
change the report (propose_text_update) never writes anything itself -- it
returns a proposal that the UI renders as a card with an explicit "Apply"
button, wired to the same update_* functions the rest of the app already
uses. See tools.propose_text_update's own docstring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.db.models import AgentConversation, AgentLlmCall, AgentMessage, AppraisalReport, ReportDocument
from app.services.llm.providers import build_sampling_kwargs, estimate_cost_usd, get_chat_model, resolve_chat_model
from app.services.llm.tools import build_assistant_tools
from app.services.llm.valuation_chain import _extract_text

MAX_OUTPUT_TOKENS = 2000
MAX_TOOL_ITERATIONS = 6

# Hard bounds for the chat console's user-facing max-reply-length / max-
# tool-steps controls (2026-08-26) -- clamp rather than trust the form
# value, since both directly drive spend (a huge max_output_tokens or an
# unbounded tool loop is a real cost/latency risk, not just a UX nicety).
MIN_OUTPUT_TOKENS = 200
MAX_OUTPUT_TOKENS_HARD_CAP = 8000
MAX_TOOL_ITERATIONS_HARD_CAP = 12

_SYSTEM_PROMPT = """Ти си асистент на лицензиран оценител на недвижими имоти в България,
работещ в контекста на КОНКРЕТЕН оценителски доклад. Помагаш чрез разговор -- отговаряш
на въпроси, търсиш сравними обяви, смяташ стойности с инструментите, и по желание
предлагаш текст за полетата на доклада.

СТРОГИ ПРАВИЛА:
- Никога не смятай пазарни агрегати, доходни стойности или претеглени заключения наум --
  винаги викай съответния инструмент (get_pool_stats, get_market_trend_stats,
  compute_income_valuation, compute_weighted_value).
- Когато оценителят поиска да напишеш/обновиш описание на имота, обосновка на
  съпоставимата зона, или обосновка на доходния подход -- ВИНАГИ извиквай
  propose_text_update, никога не пиши текста директно в отговора си вместо това.
  Инструментът не записва нищо -- само предлага текст, който оценителят вижда като
  карта с бутон "Приложи" и решава сам дали да приложи.
- Ползвай retrieve_sale_comparables/retrieve_rent_comparables за конкретни сравними
  обяви, вместо да измисляш адреси или цени.
- Ако за доклада има качени документи (виж списъка по-долу, ако е непразен) -- ПРОВЕРЯВАЙ
  и ПОЛЗВАЙ ги ПРОАКТИВНО, особено преди да пишеш описание на имота или друг текст за
  доклада, дори ако оценителят не ги е споменал изрично в текущото съобщение. Не карай
  оценителя да ти напомня за вече качени документи -- той очаква да ги знаеш. Викай
  list_documents за да видиш какво е налично, после read_document за конкретния файл,
  преди да отговориш. За скица: прочетеното е широк критичен прочит на чертежа
  (разпределение, пропорции, съответствие с декларираните данни), не само площи -- ако
  има area_summary с тераси, коефициентът за коригирана площ вече е приложен
  детерминирано (не преизчислявай сам с друг коефициент).
- Ако оценителят поиска критичен преглед/сверка/втори поглед върху доклада -- викай
  request_critical_review вместо да разсъждаваш сам. Това е отделен, по-задълбочен анализ
  (собствен LLM разговор върху цялото състояние на доклада), не просто твоето мнение в
  този отговор.
- Пиши кратко и конкретно, като разговор с колега, не като генериран доклад."""


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
    return messages


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
                      tool_calls: list | None = None, tool_call_id: str | None = None) -> AgentMessage:
    msg = AgentMessage(
        conversation_id=conversation_id, role=role, content=content,
        tool_calls=tool_calls, tool_call_id=tool_call_id,
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
    # conversation_id/provider/model_id let request_critical_review (Tier 4)
    # log its own separate LLM call to agent_llm_calls under this
    # conversation -- resolved before building tools so the critic reuses
    # the SAME model the appraiser already picked, not a second choice.
    tools = build_assistant_tools(db, report, conversation_id=conversation.id, provider=provider, model=model_id)
    effective_max_tokens = MAX_OUTPUT_TOKENS if max_output_tokens is None else max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS_HARD_CAP, max_output_tokens))
    effective_max_iterations = MAX_TOOL_ITERATIONS if max_tool_iterations is None else max(1, min(MAX_TOOL_ITERATIONS_HARD_CAP, max_tool_iterations))
    sampling_kwargs = build_sampling_kwargs(
        provider, temperature=temperature, top_p=top_p, top_k=top_k,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, seed=seed,
    )
    chat_model = get_chat_model(provider, model_id, max_tokens=effective_max_tokens, **sampling_kwargs)
    model_with_tools = chat_model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    system_content = _SYSTEM_PROMPT + _documents_note(db, report.id)
    messages: list = [SystemMessage(content=system_content)] + _load_langchain_messages(db, conversation.id)

    total_input_tokens = 0
    total_output_tokens = 0
    call_log: list[dict] = []
    final_text = ""

    try:
        for iteration_idx in range(effective_max_iterations):
            progress.step = "Мисля…"
            chunk_accum = None
            last_chunk_usage = {}
            for chunk in model_with_tools.stream(messages):
                chunk_accum = chunk if chunk_accum is None else chunk_accum + chunk
                if chunk.usage_metadata:
                    last_chunk_usage = chunk.usage_metadata
                live_text = _extract_text(chunk_accum.content)
                progress.tokens_so_far = total_input_tokens + total_output_tokens + max(len(live_text) // 4, 0)
                emit()
            response: AIMessage = chunk_accum
            call_in = last_chunk_usage.get("input_tokens", 0)
            call_out = last_chunk_usage.get("output_tokens", 0)
            total_input_tokens += call_in
            total_output_tokens += call_out
            call_log.append({"call_label": f"msg{msg_seq}_step{iteration_idx + 1}", "input_tokens": call_in, "output_tokens": call_out})
            progress.tokens_so_far = total_input_tokens + total_output_tokens
            emit()

            messages.append(response)

            if not response.tool_calls:
                final_text = _extract_text(response.content)
                _persist_message(db, conversation.id, "assistant", content=final_text)
                break

            _persist_message(
                db, conversation.id, "assistant",
                content=_extract_text(response.content) or None,
                tool_calls=response.tool_calls,
            )

            for call in response.tool_calls:
                progress.step = f"Изпълнявам: {call['name']}…"
                emit()
                tool = tools_by_name.get(call["name"])
                result = tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
                result_json = json.dumps(result, default=str, ensure_ascii=False)
                messages.append(ToolMessage(content=result_json, tool_call_id=call["id"]))
                _persist_message(db, conversation.id, "tool", content=result_json, tool_call_id=call["id"])
        else:
            # Loop exhausted without a final answer -- still a completed
            # turn (all messages up to here are persisted), just without a
            # closing assistant reply. Rare at the default MAX_TOOL_ITERATIONS=6
            # (or whatever max_tool_iterations the appraiser dialed in).
            final_text = ""
            progress.step = "Достигнат лимит на стъпките за този отговор."
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
    progress.tokens_so_far = total_input_tokens + total_output_tokens
    emit()
    return progress
