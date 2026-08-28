"""Conversational market analyst agent (Phase 10, 2026-08-28) -- a second,
report-agnostic agent for free-form market research across the whole
listings corpus (time series, segment/geo/construction-type
cross-sections, rental market analysis) plus a shared library of
market-research documents it can read and cross-reference against live
data.

Deliberately NOT scoped to a report -- see AgentConversation's own
docstring for why this shares the agent_conversations/agent_messages/
agent_llm_calls tables with the report-scoped assistant
(app/services/llm/assistant_chain.py) rather than duplicating them. Reuses
that module's message (de)serialization and call-log persistence directly
-- both are already report-agnostic (they only ever touched
conversation_id), so no code with report-specific behavior is shared here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.db.models import AgentConversation, AgentMessage
from app.services.llm.analyst_tools import build_analyst_tools
from app.services.llm.assistant_chain import ChatProgress, _load_langchain_messages, _persist_call_log, _persist_message
from app.services.llm.providers import build_sampling_kwargs, get_chat_model, resolve_chat_model
from app.services.llm.valuation_chain import _extract_text

MAX_OUTPUT_TOKENS = 2000
MAX_TOOL_ITERATIONS = 6
MIN_OUTPUT_TOKENS = 200
MAX_OUTPUT_TOKENS_HARD_CAP = 8000
MAX_TOOL_ITERATIONS_HARD_CAP = 12

_SYSTEM_PROMPT = """Ти си пазарен анализатор на недвижими имоти в България, с достъп до
целия корпус събрани обяви от imot.bg (стотици хиляди обяви, множество засичания във
времето) -- НЕ работиш в контекста на конкретен оценителски доклад или конкретен имот.
Помагаш чрез свободен разговор: времеви анализ, сегментен анализ, сравнения между
квартали/градове/типове строителство/типове имот, пазар на продажби и на наеми.

СТРОГИ ПРАВИЛА:
- Никога не смятай статистика (медиани, проценти, тенденции) наум -- ВИНАГИ викай
  query_market_stats. Ако въпросът включва повече от една стойност на измерение (напр.
  "двустайни И тристайни", "няколко квартала"), подавай ги като списък в едно извикване,
  или използвай group_by, за да получиш всички наведнъж.
- Преди да ползваш конкретно име на град/квартал/тип строителство, ако не си сигурен в
  точния му запис в базата -- викай list_market_filter_values. Точен low-case филтър върху
  грешен правопис тихо връща нула резултати, не грешка -- не гадай.
- Когато сравняваш нещо (квартали, типове строителство, сегменти), извиквай
  query_market_stats с подходящия group_by и коментирай РЕАЛНИТЕ числа от резултата -- не
  описвай сравнение, което не си действително извлякъл.
- Ако оценителят спомене или качи документ (пазарен анализ, статия, статистика) -- викай
  list_market_documents/read_market_document. След като прочетеш конкретни твърдения или
  цифри от документ, ПРОВЕРИ ги с query_market_stats за съответния сегмент/период и кажи
  изрично дали реалните данни съвпадат, или се разминават -- не приемай документа за верен
  само защото е публикуван.
- Имай предвид: пълни засичания на пазара се правят на всеки 2-3 седмици, не ежедневно --
  "времеви ред" реалистично означава шепа точки (виж n_runs в резултата), не гъста серия.
  Не подвеждай оценителя да очаква повече детайлност, отколкото данните реално имат.
- Пиши кратко и конкретно, с реални числа, като разговор с колега-анализатор."""


def get_or_create_analyst_conversation(db: Session, user_id: int) -> AgentConversation:
    """One persistent market-analyst conversation per user (v1 -- same
    "one thread" simplicity as the report assistant's v1, see
    assistant_chain.get_or_create_conversation, just keyed by
    (user, agent_type) instead of (user, report))."""
    from app.services.llm.assistant_chain import get_or_create_conversation
    return get_or_create_conversation(db, report_id=None, user_id=user_id, agent_type="market_analyst")


def run_analyst_turn(
    db: Session,
    conversation: AgentConversation,
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
    """Mirrors assistant_chain.run_assistant_turn's tool-calling loop
    exactly (streaming, per-call token ledger, persisted history) but
    against build_analyst_tools(db) and no report/document-note/critic
    wiring -- this agent has no report to write to or critique."""
    progress = ChatProgress()

    def emit():
        if on_progress:
            on_progress(progress)

    emit()

    if not conversation.title:
        conversation.title = user_text[:80]
    prior_user_messages = (
        db.query(AgentMessage)
        .filter(AgentMessage.conversation_id == conversation.id, AgentMessage.role == "user")
        .count()
    )
    msg_seq = prior_user_messages + 1
    _persist_message(db, conversation.id, "user", content=user_text)

    provider, model_id = resolve_chat_model(provider, model)
    tools = build_analyst_tools(db)
    effective_max_tokens = MAX_OUTPUT_TOKENS if max_output_tokens is None else max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS_HARD_CAP, max_output_tokens))
    effective_max_iterations = MAX_TOOL_ITERATIONS if max_tool_iterations is None else max(1, min(MAX_TOOL_ITERATIONS_HARD_CAP, max_tool_iterations))
    sampling_kwargs = build_sampling_kwargs(
        provider, temperature=temperature, top_p=top_p, top_k=top_k,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, seed=seed,
    )
    chat_model = get_chat_model(provider, model_id, max_tokens=effective_max_tokens, **sampling_kwargs)
    model_with_tools = chat_model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)] + _load_langchain_messages(db, conversation.id)

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
            final_text = ""
            progress.step = "Достигнат лимит на стъпките за този отговор."
    except Exception as exc:
        _persist_call_log(db, conversation.id, provider, model_id, call_log)
        progress.status = "error"
        progress.error = f"Грешка при отговора: {exc}"
        emit()
        return progress

    _persist_call_log(db, conversation.id, provider, model_id, call_log)
    db.commit()

    progress.status = "done"
    progress.step = "Готово"
    progress.tokens_so_far = total_input_tokens + total_output_tokens
    emit()
    return progress
