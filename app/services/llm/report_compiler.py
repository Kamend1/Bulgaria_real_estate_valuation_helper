"""Report Compiler (Phase 13, 2026-09-01) -- runs several specialists
sequentially against one report as an explicit, standalone action
(/comparables/compile), not a chat conversation. The appraiser picks which
domains to run via checkboxes; each runs once with a fixed "do a full
review" instruction instead of a free-form question, and there is no
Supervisor routing step -- the appraiser already chose the domains.

Reuses build_specialist_tools_and_prompt/_build_tool_loop from
orchestrator_graph.py directly, so a prompt or tool change made for the
chat path is automatically picked up here too -- nothing about the
specialist logic is duplicated.

Deliberately SEQUENTIAL, not parallel (see the approved plan's "Идея F"
section): every specialist's tools close over ONE shared SQLAlchemy
Session, which is not safe for concurrent use across threads/branches.
Parallelizing this would require each branch to open its own
db_session() -- a real architecture change, explicitly deferred, not
attempted here.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, ReportCompileRun
from app.services.llm.assistant_chain import _documents_note, _persist_call_log
from app.services.llm.orchestrator_graph import _build_tool_loop, build_specialist_tools_and_prompt
from app.services.llm.providers import build_sampling_kwargs, resolve_chat_model

_COMPILE_INSTRUCTIONS = {
    "income": (
        "Направи пълен преглед по доходния подход за този доклад -- изчисли стойността с "
        "наличните пазарни данни (наемни сравними/пул) и коментирай резултата. Ако е "
        "подходящо, предложи текст за доходната обосновка."
    ),
    "market": (
        "Направи пълен преглед по пазарния подход за този доклад -- провери сравнимите "
        "продажби и пазарния тренд, коментирай позицията на имота. Ако е подходящо, предложи "
        "описание на имота или обосновка на съпоставимата зона."
    ),
    "market_analysis": (
        "Направи кратък по-широк пазарен анализ, релевантен за сегмента/квартала/типа "
        "строителство на този имот, използвайки целия корпус обяви."
    ),
    "legal": (
        "Направи кратък преглед на правните/етичните съображения, релевантни за тази оценка, "
        "проверявайки качената библиотека с нормативни документи."
    ),
}

DOMAIN_LABELS = {
    "income": "Доходен подход", "market": "Пазарен подход",
    "market_analysis": "Пазарен анализ", "legal": "Правно/етично",
}


def run_compile(
    db: Session, report: AppraisalReport, run: ReportCompileRun, domains: list[str],
    provider: str | None, model: str | None,
    on_progress=None,
) -> ReportCompileRun:
    """Synchronous, blocking (real network calls) -- run on a background
    thread from the router, same convention as
    assistant_chain.run_assistant_turn. `run` is an already-created,
    already-committed ReportCompileRun row (the router creates it
    synchronously so it can return run.id to the client immediately,
    before backgrounding) -- and `db` must be a session opened BY the
    background thread itself, never the request's own session (see
    app/routers/comparables.py's _run_compile_thread: reusing a
    request-scoped Session across threads is exactly the concurrency
    hazard flagged in the Idea F audit). on_progress(str), if given,
    reports step text for the SSE progress stream."""
    resolved_provider, resolved_model = resolve_chat_model(provider, model)
    run.provider, run.model = resolved_provider, resolved_model
    db.commit()

    def _progress(step: str) -> None:
        if on_progress:
            on_progress(step)

    documents_note = _documents_note(db, report.id)
    sampling_kwargs = build_sampling_kwargs(resolved_provider)
    call_log: list[dict] = []
    results: dict[str, dict] = {}

    try:
        for domain in domains:
            _progress(f"{DOMAIN_LABELS.get(domain, domain)}…")
            tools, prompt, step_label = build_specialist_tools_and_prompt(domain, db, report, documents_note)
            proposals: list[dict] = []

            def _on_message(role, content, tool_calls, tool_call_id, truncated=False, _proposals=proposals):
                if role != "tool" or not content:
                    return
                try:
                    parsed = json.loads(content)
                except Exception:
                    return
                if isinstance(parsed, dict) and parsed.get("proposed"):
                    _proposals.append(parsed)

            node = _build_tool_loop(
                provider=resolved_provider, model_id=resolved_model, sampling_kwargs=sampling_kwargs,
                max_tokens=2500, tools=tools, system_prompt=prompt, max_iterations=6,
                call_log=call_log, call_label_prefix=f"compile_{domain}",
                on_progress=_progress, on_message=_on_message, step_label=step_label,
                db=db, report_id=report.id, memory_source="compile", memory_source_id=run.id,
            )
            state = {
                "messages": [HumanMessage(content=_COMPILE_INSTRUCTIONS[domain])],
                "next_specialist": None, "direct_answer": None, "findings": {}, "hops": 0,
            }
            node_result = node(state)
            final_text = node_result.get("findings", {}).get(domain, "")
            results[domain] = {"text": final_text, "proposals": proposals}

        run.results = results
        run.status = "done"
    except Exception as exc:
        run.status = "error"
        run.error_message = str(exc)[:2000]
        db.commit()
        _persist_call_log(db, run.id, resolved_provider, resolved_model, call_log)
        raise

    db.commit()
    _persist_call_log(db, run.id, resolved_provider, resolved_model, call_log)
    return run
