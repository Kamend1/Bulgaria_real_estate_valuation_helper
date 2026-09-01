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

Phase 13 (2026-09-01) retired this module's own hand-rolled tool-calling
loop -- it was a byte-for-byte copy of orchestrator_graph.py's
_build_tool_loop (bind_tools -> stream -> accumulate -> execute -> repeat),
the exact kind of duplication that extraction was meant to kill. Now calls
that shared loop directly with report_id=None (report_memory.persist_finding
treats that as a legitimate no-op -- this agent has no report to attach
memory to) instead of maintaining a third copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.db.models import AgentConversation, AgentMessage
from app.services.llm.analyst_tools import build_analyst_tools
from app.services.llm.assistant_chain import ChatProgress, _load_langchain_messages, _persist_call_log, _persist_message
from app.services.llm.orchestrator_graph import _build_tool_loop
from app.services.llm.providers import build_sampling_kwargs, resolve_chat_model

MAX_OUTPUT_TOKENS = 3000
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
    """Fallback conversation resolution when no conversation is explicitly
    selected in session -- reuses the latest existing one (or creates the
    first), same as assistant_chain.get_or_create_conversation, just keyed
    by (user, agent_type) instead of (user, report). Since Phase 12
    (2026-08-31) a user can have MULTIPLE market-analyst conversations --
    see new_analyst_conversation/list_analyst_conversations below -- this
    is only the "nothing selected yet" default, not the only thread."""
    from app.services.llm.assistant_chain import get_or_create_conversation
    return get_or_create_conversation(db, report_id=None, user_id=user_id, agent_type="market_analyst")


def new_analyst_conversation(db: Session, user_id: int) -> AgentConversation:
    """Always starts a fresh, separate market-analyst conversation --
    thin wrapper over assistant_chain.new_conversation, kept here so
    callers (market_analyst.py) never need to know agent_type="market_analyst"
    is the right string to pass."""
    from app.services.llm.assistant_chain import new_conversation
    return new_conversation(db, user_id, agent_type="market_analyst")


def list_analyst_conversations(db: Session, user_id: int) -> list[AgentConversation]:
    """All of a user's market-analyst conversations, for the switcher."""
    from app.services.llm.assistant_chain import list_conversations
    return list_conversations(db, user_id, agent_type="market_analyst")


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
    """Builds and runs the shared specialist tool loop (orchestrator_graph.
    _build_tool_loop) with report_id=None -- same streaming/per-call token
    ledger/persisted-history behavior as before Phase 13, just no longer a
    separate hand-written copy of that loop."""
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

    call_log: list[dict] = []

    def on_progress_cb(step: str) -> None:
        progress.step = step
        emit()

    def on_message_cb(role: str, content: str | None, tool_calls: list | None, tool_call_id: str | None, truncated: bool = False) -> None:
        _persist_message(db, conversation.id, role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id, truncated=truncated)

    node = _build_tool_loop(
        provider=provider, model_id=model_id, sampling_kwargs=sampling_kwargs, max_tokens=effective_max_tokens,
        tools=tools, system_prompt=_SYSTEM_PROMPT, max_iterations=effective_max_iterations,
        call_log=call_log, call_label_prefix=f"msg{msg_seq}", on_progress=on_progress_cb, on_message=on_message_cb,
        step_label="Мисля…", db=db, report_id=None, memory_source="chat", memory_source_id=conversation.id,
    )

    try:
        node({"messages": _load_langchain_messages(db, conversation.id)})
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
    progress.tokens_so_far = sum(e["input_tokens"] + e["output_tokens"] for e in call_log)
    emit()
    return progress
