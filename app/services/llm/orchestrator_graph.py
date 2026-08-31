"""Multi-agent supervisor graph for the report-scoped AI Assistant (Phase 11,
2026-08-28) -- replaces assistant_chain.py's single flat-tool-list loop
with real LangGraph routing to focused specialist nodes, each with its own
system prompt and only its own relevant tools.

Why real specialist nodes, not a bigger flat tool list: the assistant
already juggles 9 tools in one prompt (tools.py's build_assistant_tools).
Adding legal reasoning + market-wide analysis + a dedicated audit pass on
top would push well past ~15 tools bound to one call, where an LLM's
tool-selection accuracy measurably degrades (established practice, not
speculation). Routing to a focused specialist -- its own prompt, only ITS
tools -- keeps each individual LLM call's tool surface small.

v1 deliberately does NOT loop back to the supervisor after a specialist
answers (no add_conditional_edges cycle) -- one routing decision per turn,
specialist's answer IS the turn's answer. Lower risk, no infinite-loop
guard needed, and a genuinely two-domain question just gets a natural
follow-up next message (conversation history carries forward). A cyclic
"specialist hands back to supervisor for more" version is a valid future
step once this is proven, not attempted here.

Follows critic_graph.py's established pattern (the one real LangGraph
precedent in this codebase): StateGraph, factory-closure nodes,
with_structured_output(..., include_raw=True) for the supervisor's routing
choice so its own token usage isn't silently lost from the ledger.
"""
from __future__ import annotations

import json
from typing import Callable, Literal, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, MarketDocument
from app.services.llm import critic_graph
from app.services.llm.analyst_tools import build_analyst_tools
from app.services.llm.providers import get_chat_model, is_length_truncated
from app.services.llm.tools import (
    _income_valuation_description,
    _income_valuation_fn,
    _list_documents_fn,
    _market_trend_fn,
    _pool_stats_fn,
    _propose_text_update_fn,
    _read_document_fn,
    _retrieve_comparables_fn,
    _weighted_value_fn,
)
from app.services.llm.valuation_chain import _extract_text
from app.services.market_documents import LEGAL_DOCUMENT_TYPE

# OnMessage persists one message (role, content, tool_calls, tool_call_id,
# truncated) -- assistant_chain._persist_message's exact signature shape,
# passed in as a callback so this module never touches the DB/
# conversation_id directly. `truncated` (Phase 12, 2026-08-31) flags an
# assistant answer that was cut short by the provider's max-tokens ceiling
# -- see is_length_truncated.
OnMessage = Callable[[str, str | None, list | None, str | None, bool], None]
OnProgress = Callable[[str], None]

_SPECIALIST_DISPLAY_NAMES = {
    "income": "доходен подход", "market": "пазарен подход",
    "market_analysis": "пазарен анализ", "legal": "правно/етично",
}


class OrchestratorState(TypedDict):
    messages: list
    next_specialist: str | None
    direct_answer: str | None
    findings: dict


class RouteDecision(BaseModel):
    next: Literal["income", "market", "market_analysis", "legal", "auditor", "answer"] = Field(
        description=(
            "Кой специалист да поеме тази реплика: 'income' (доходен подход/DCF/"
            "капитализация), 'market' (сравними обяви/пазарен подход/описание на имота/"
            "документи), 'market_analysis' (по-широк пазарен анализ извън тази конкретна "
            "сделка -- времеви редове, сравнение на квартали/типове строителство), "
            "'legal' (правни/етични въпроси), 'auditor' (критичен преглед на целия "
            "доклад), или 'answer' ако въпросът е достатъчно прост/общ да отговориш "
            "директно, без специалист (напр. поздрав, въпрос какво умееш)."
        )
    )
    reasoning: str = Field(description="Едно кратко изречение защо.")
    direct_answer: str | None = Field(
        default=None,
        description="Ако next='answer', попълни директния отговор тук. Иначе остави празно.",
    )


_SUPERVISOR_PROMPT = """Ти си диспечер (supervisor) в екип от специализирани AI агенти,
подпомагащи лицензиран оценител на недвижими имоти в България по КОНКРЕТЕН доклад
(може да е реален случай или хипотетичен сценарий -- третирай ги еднакво). Задачата ти
е само да решиш кой специалист да отговори на текущата реплика на оценителя -- ти самият
НЕ смяташ стойности и не пишеш дълги отговори (освен ако next='answer' за наистина
прост/общ въпрос)."""


def _build_tool_loop(
    provider: str, model_id: str, sampling_kwargs: dict, max_tokens: int,
    tools: list[StructuredTool], system_prompt: str, max_iterations: int,
    call_log: list[dict], call_label_prefix: str, on_progress: OnProgress, on_message: OnMessage,
    step_label: str,
) -> Callable[[OrchestratorState], dict]:
    """Shared tool-calling loop body used by every specialist node --
    extracted so income/market/market_analysis/legal don't each reimplement
    assistant_chain.run_assistant_turn's original loop (bind_tools -> stream
    -> accumulate -> execute tool calls -> repeat). Mirrors that loop
    exactly, just parameterized per specialist."""
    tools_by_name = {t.name: t for t in tools}

    def node(state: OrchestratorState) -> dict:
        chat_model = get_chat_model(provider, model_id, max_tokens=max_tokens, **sampling_kwargs)
        model_with_tools = chat_model.bind_tools(tools) if tools else chat_model
        messages: list = [SystemMessage(content=system_prompt)] + list(state["messages"])
        final_text = ""

        for i in range(max_iterations):
            on_progress(f"{step_label}…")
            chunk_accum = None
            last_usage = {}
            for chunk in model_with_tools.stream(messages):
                chunk_accum = chunk if chunk_accum is None else chunk_accum + chunk
                if chunk.usage_metadata:
                    last_usage = chunk.usage_metadata
            response = chunk_accum
            call_log.append({
                "call_label": f"{call_label_prefix}_step{i + 1}",
                "input_tokens": last_usage.get("input_tokens", 0),
                "output_tokens": last_usage.get("output_tokens", 0),
            })
            messages.append(response)

            if not response.tool_calls:
                final_text = _extract_text(response.content)
                on_message("assistant", final_text, None, None, is_length_truncated(response))
                break

            on_message("assistant", _extract_text(response.content) or None, response.tool_calls, None, False)
            for call in response.tool_calls:
                on_progress(f"Изпълнявам: {call['name']}…")
                tool = tools_by_name.get(call["name"])
                result = tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
                result_json = json.dumps(result, default=str, ensure_ascii=False)
                messages.append(ToolMessage(content=result_json, tool_call_id=call["id"]))
                on_message("tool", result_json, None, call["id"])
        else:
            final_text = ""
            on_progress("Достигнат лимит на стъпките за този специалист.")

        return {"findings": {**state.get("findings", {}), call_label_prefix.split("_", 1)[-1]: final_text}}

    return node


def _supervisor_node_fn(provider: str, model_id: str, call_log: list[dict], msg_seq: int, on_progress: OnProgress):
    def node(state: OrchestratorState) -> dict:
        on_progress("Насочвам въпроса…")
        chat = get_chat_model(provider, model_id, max_tokens=400)
        structured = chat.with_structured_output(RouteDecision, include_raw=True)
        result = structured.invoke([SystemMessage(content=_SUPERVISOR_PROMPT), *state["messages"]])
        raw_msg = result.get("raw")
        usage = (getattr(raw_msg, "usage_metadata", None) or {}) if raw_msg is not None else {}
        call_log.append({
            "call_label": f"msg{msg_seq}_supervisor",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
        parsed: RouteDecision | None = result.get("parsed")
        if parsed is None:
            return {"next_specialist": "answer", "direct_answer": "Извинявай, не успях да обработя въпроса. Опитай да го преформулираш."}
        return {"next_specialist": parsed.next, "direct_answer": parsed.direct_answer}
    return node


def _direct_answer_node_fn(on_message: OnMessage):
    def node(state: OrchestratorState) -> dict:
        text = state.get("direct_answer") or "Няма какво да добавя."
        on_message("assistant", text, None, None)
        return {"findings": {**state.get("findings", {}), "answer": text}}
    return node


def _auditor_node_fn(
    db: Session, report: AppraisalReport, provider: str | None, model: str | None,
    call_log: list[dict], msg_seq: int, on_progress: OnProgress, on_message: OnMessage,
):
    def node(state: OrchestratorState) -> dict:
        on_progress("Правя критичен преглед на доклада…")
        critique, sub_call_log = critic_graph.run_critical_review(db, report, provider, model)
        for entry in sub_call_log:
            call_log.append({
                "call_label": f"msg{msg_seq}_auditor",
                "input_tokens": entry.get("input_tokens", 0),
                "output_tokens": entry.get("output_tokens", 0),
            })
        result_json = json.dumps(critique, default=str, ensure_ascii=False)
        # Persisted as a tool-call/tool-result pair shaped exactly like the
        # pre-Phase-11 request_critical_review TOOL result, so the existing
        # _messages.html branch (msg.parsed.overall_assessment is defined)
        # renders the same critique card unchanged -- zero template changes
        # needed for this node.
        on_message("assistant", None, [{"id": "auditor-0", "name": "request_critical_review", "args": {}}], None)
        on_message("tool", result_json, None, "auditor-0")
        summary = critique.get("overall_assessment", "")
        on_message("assistant", summary, None, None)
        return {"findings": {**state.get("findings", {}), "auditor": summary}}
    return node


def _list_legal_documents_fn(db: Session):
    def list_legal_documents() -> dict:
        """Lists uploaded legal/regulatory reference documents (statutes, наредби,
        the КНОБ ethics code) available in the shared library -- id, filename, title,
        issuing body. Call this first to see what's actually been uploaded before
        answering from general knowledge."""
        docs = (
            db.query(MarketDocument)
            .filter(MarketDocument.document_type == LEGAL_DOCUMENT_TYPE, MarketDocument.status == "ready")
            .order_by(MarketDocument.created_at.desc())
            .all()
        )
        return {
            "documents": [
                {
                    "id": str(d.id), "filename": d.filename,
                    "title": (d.extracted_data or {}).get("title"),
                    "issuing_body": (d.extracted_data or {}).get("issuing_body"),
                }
                for d in docs
            ]
        }
    return list_legal_documents


def _read_legal_document_fn(db: Session):
    def read_legal_document(document_id: str) -> dict:
        """Returns the FULL verbatim text of one uploaded legal/regulatory document.
        Use this to quote an exact чл./ал./точка instead of paraphrasing from general
        knowledge -- once a real source is available, prefer citing it explicitly."""
        doc = (
            db.query(MarketDocument)
            .filter(MarketDocument.id == document_id, MarketDocument.document_type == LEGAL_DOCUMENT_TYPE)
            .first()
        )
        if doc is None:
            return {"error": "Документът не е намерен."}
        if doc.status != "ready":
            return {"status": doc.status, "error": doc.error_message}
        data = doc.extracted_data or {}
        return {
            "filename": doc.filename, "title": data.get("title"), "issuing_body": data.get("issuing_body"),
            "text": data.get("full_text", ""),
        }
    return read_legal_document


def _route_from_supervisor(state: OrchestratorState) -> str:
    return state.get("next_specialist") or "answer"


def build_orchestrator_graph(
    db: Session,
    report: AppraisalReport,
    documents_note: str,
    provider: str,
    model_id: str,
    sampling_kwargs: dict,
    max_tokens: int,
    max_tool_iterations: int,
    call_log: list[dict],
    msg_seq: int,
    on_progress: OnProgress,
    on_message: OnMessage,
):
    """Builds a fresh graph for this turn (mirrors build_assistant_tools()
    being reconstructed fresh per turn too -- no cross-turn caching, DB is
    the source of truth for history). report may be a real or a
    is_scratch=True (hypothetical) report -- every specialist works
    identically either way, since it's a perfectly normal AppraisalReport
    row (see is_scratch's own docstring in app/db/models.py)."""

    def base_kwargs(prefix: str):
        return dict(
            provider=provider, model_id=model_id, sampling_kwargs=sampling_kwargs, max_tokens=max_tokens,
            max_iterations=max_tool_iterations, call_log=call_log, call_label_prefix=f"msg{msg_seq}_{prefix}",
            on_progress=on_progress, on_message=on_message,
        )

    income_tools = [
        StructuredTool.from_function(_income_valuation_fn(report), name="compute_income_valuation", description=_income_valuation_description()),
        StructuredTool.from_function(_pool_stats_fn(db, report), name="get_pool_stats"),
        StructuredTool.from_function(_retrieve_comparables_fn(db, report, "rent"), name="retrieve_rent_comparables", description="Semantic search for rent comparables closest to the subject property (up to k, default 6)."),
        StructuredTool.from_function(_weighted_value_fn(), name="compute_weighted_value"),
        StructuredTool.from_function(_propose_text_update_fn(report), name="propose_text_update", description="Propose new text for income_market_rationale (or subject_description/submarket_rationale). Never writes -- returns a proposal the appraiser must apply."),
    ]
    income_prompt = (
        "Ти си специалист по ДОХОДЕН ПОДХОД (Income Approach) за оценка на недвижими имоти -- "
        "директна капитализация, DCF (дисконтиран паричен поток), терминална стойност, "
        "чувствителност спрямо cap rate/наем. Разбираш анюитети, перпетуитети, дисконтови "
        "фактори -- обяснявай ги, когато е полезно, но НИКОГА не смятай наум: винаги викай "
        "compute_income_valuation. Вземай наемни данни от retrieve_rent_comparables/"
        "get_pool_stats, не измисляй наем. Годината на строеж и състоянието на сградата "
        "влияят на риска (по-висок cap rate за старо/амортизирано строителство) -- коментирай "
        "го, ако е известно от доклада." + documents_note
    )

    market_tools = [
        StructuredTool.from_function(_pool_stats_fn(db, report), name="get_pool_stats"),
        StructuredTool.from_function(_market_trend_fn(db, report), name="get_market_trend_stats"),
        StructuredTool.from_function(_retrieve_comparables_fn(db, report, "sale"), name="retrieve_sale_comparables", description="Semantic search for sale comparables closest to the subject property (up to k, default 6)."),
        StructuredTool.from_function(_list_documents_fn(db, report), name="list_documents"),
        StructuredTool.from_function(_read_document_fn(db, report), name="read_document"),
        StructuredTool.from_function(_propose_text_update_fn(report), name="propose_text_update", description="Propose new text for subject_description or submarket_rationale. Never writes -- returns a proposal the appraiser must apply."),
    ]
    market_prompt = (
        "Ти си специалист по ПАЗАРЕН ПОДХОД (Sales Comparison Approach) за КОНКРЕТНИЯ имот в "
        "този доклад -- сравними продажби, описание на имота, качени документи (нотариален "
        "акт, скица и др.). Никога не смятай пазарни агрегати наум -- винаги викай "
        "get_pool_stats/get_market_trend_stats. Когато оценителят поиска описание на имота "
        "или обосновка на съпоставимата зона -- ВИНАГИ викай propose_text_update, никога не "
        "пиши текста директно в отговора си." + documents_note
    )

    legal_tools = [
        StructuredTool.from_function(_list_legal_documents_fn(db), name="list_legal_documents"),
        StructuredTool.from_function(_read_legal_document_fn(db), name="read_legal_document"),
    ]
    legal_prompt = (
        "Ти си правен/етичен консултант за оценителска дейност в България -- познания за "
        "принципите на ЗННД, Наредба № 1/14.02.2007, международните стандарти за оценяване "
        "(МСО/IVS) и общата рамка на професионалната етика (КНОБ).\n\n"
        "ЗАДЪЛЖИТЕЛНО за всеки въпрос: първо викай list_legal_documents, за да видиш какви "
        "реални нормативни текстове са качени в библиотеката. Ако има релевантен документ -- "
        "викай read_legal_document и цитирай ТОЧНИЯ член/алинея/точка от върнатия дословен "
        "текст, вместо да разчиташ на общи познания. Ако няма качен релевантен документ (или "
        "библиотеката е празна) -- отговори от общи познания, но ЗАДЪЛЖИТЕЛНО завърши отговора "
        "си с изрично предупреждение, че преценката не е закотвена в конкретен качен източник и "
        "трябва да се провери от юрист/КНОБ, преди оценителят да разчита на нея за реално "
        "решение. Никога не представяй правно заключение като сигурно, освен ако не цитираш "
        "буквално качен източник."
    )

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", _supervisor_node_fn(provider, model_id, call_log, msg_seq, on_progress))
    graph.add_node("income", _build_tool_loop(tools=income_tools, system_prompt=income_prompt, step_label="Мисля (доходен подход)", **base_kwargs("income")))
    graph.add_node("market", _build_tool_loop(tools=market_tools, system_prompt=market_prompt, step_label="Мисля (пазарен подход)", **base_kwargs("market")))
    graph.add_node("market_analysis", _build_tool_loop(tools=build_analyst_tools(db), system_prompt=_market_analysis_prompt(), step_label="Мисля (пазарен анализ)", **base_kwargs("market_analysis")))
    graph.add_node("legal", _build_tool_loop(tools=legal_tools, system_prompt=legal_prompt, step_label="Мисля (правно/етично)", **base_kwargs("legal")))
    graph.add_node("auditor", _auditor_node_fn(db, report, provider, model_id, call_log, msg_seq, on_progress, on_message))
    graph.add_node("direct_answer", _direct_answer_node_fn(on_message))

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", _route_from_supervisor, {
        "income": "income", "market": "market", "market_analysis": "market_analysis",
        "legal": "legal", "auditor": "auditor", "answer": "direct_answer",
    })
    for node_name in ("income", "market", "market_analysis", "legal", "auditor", "direct_answer"):
        graph.add_edge(node_name, END)

    return graph.compile()


def _market_analysis_prompt() -> str:
    return (
        "Ти си специалист по ШИРОК ПАЗАРЕН АНАЛИЗ -- времеви редове, сравнение между "
        "квартали/градове/типове строителство/типове имот, наемен пазар -- отвъд "
        "конкретната сделка в този доклад. Разполагаш с целия корпус обяви. Никога не "
        "смятай статистика наум -- винаги викай query_market_stats. Ако не си сигурен в "
        "точния запис на град/квартал/тип строителство -- викай list_market_filter_values "
        "първо."
    )
