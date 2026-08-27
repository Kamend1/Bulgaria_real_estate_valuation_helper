"""Critic/orchestrator graph (Tier 4, 2026-08-26) -- the "one agent that
uses all outputs and thinks critically" from the owner's own framing of the
multi-agent idea.

A genuine LangGraph StateGraph, not a single function dressed up as one:
gather_context (pure DB/service reads, no LLM call, independently testable)
-> critique (the one LLM call, structured output). Kept as two nodes so
this is the natural place to extend later (e.g. a conditional re-critique
loop, or a second specialist node) without a redesign -- exactly the
supervisor/delegation shape LangGraph is actually for, as opposed to the
single conversational agent's tool-calling loop in assistant_chain.py,
which stays a hand-rolled loop because it has no branching/delegation to
justify the graph machinery.

Guardrail: the critique is READ-ONLY commentary -- it never writes to the
report, mirroring propose_text_update's own "never silently write" rule
elsewhere in this app.
"""
from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport
from app.services import avm_service
from app.services.comparable_service import get_pool_with_stats
from app.services.llm.providers import get_chat_model, resolve_chat_model


class CritiqueResult(BaseModel):
    consistency_issues: list[str] = Field(default_factory=list, description="Несъответствия между написаните текстове и числата в доклада")
    avm_commentary: str | None = Field(default=None, description="Коментар дали AVM прогнозата изглежда разумна спрямо сравнимите, ако има такава")
    missing_caveats: list[str] = Field(default_factory=list, description="Ограничения/допускания, които липсват предвид целта на оценката")
    overall_assessment: str = Field(description="Кратко общо заключение -- готов ли изглежда докладът, или има какво да се провери")


class CriticState(TypedDict):
    report_id: str
    context: dict
    critique: dict


_CRITIC_SYSTEM_PROMPT = """Ти си критичен рецензент на чернова на оценителски доклад за
недвижим имот в България. Получаваш JSON обобщение на текущото състояние на доклада --
статистика на сравнимите, изчислени стойности, AVM прогноза (ако има), вече написани
текстове -- и трябва да го прегледаш критично, НЕ да го пренапишеш и НЕ да предлагаш нов
текст.

Провери конкретно:
- Съответства ли написаният наратив (submarket_rationale/income_market_rationale) на
  числата от сравнимите и заключените стойности?
- Ако има avm_prediction -- изглежда ли разумна спрямо статистиката на сравнимите, или
  има съществено разминаване, което си струва изрично да се спомене?
- Липсват ли очевидни ограничения/допускания предвид report_purpose (напр. непарична
  вноска по чл. 72 ТЗ има различни изисквания от обща пазарна консултация)?

Бъди конкретен и кратък -- не преповтаряй суровите данни. Ако нещо изглежда наред, кажи
го кратко вместо да измисляш проблем, за да запълниш поле."""


def _gather_context_node(db: Session, report: AppraisalReport):
    def gather_context(state: CriticState) -> dict:
        pool_sale = get_pool_with_stats(db, "sale", report.id)
        pool_rent = get_pool_with_stats(db, "rent", report.id)

        avm_prediction = None
        try:
            pred = avm_service.predict_sales_value(db, report)
            if pred.get("ok"):
                avm_prediction = {k: v for k, v in pred.items() if k != "metrics"}
        except Exception:
            avm_prediction = None   # non-fatal -- critique proceeds without AVM commentary

        context = {
            "subject": {
                "property_type": report.subject_property_type,
                "geo_category": report.subject_geo_category,
                "area_sqm": float(report.subject_area_sqm) if report.subject_area_sqm else None,
                "description": report.subject_description,
            },
            "report_purpose": report.report_purpose,
            "sale_pool_stats": pool_sale["stats"],
            "rent_pool_stats": pool_rent["stats"],
            "avm_prediction": avm_prediction,
            "concluded_value_sales": float(report.concluded_value_sales) if report.concluded_value_sales else None,
            "concluded_value_income": float(report.concluded_value_income) if report.concluded_value_income else None,
            "concluded_value_residual": float(report.concluded_value_residual) if report.concluded_value_residual else None,
            "concluded_value_weighted": float(report.concluded_value) if report.concluded_value else None,
            "submarket_rationale": report.submarket_rationale,
            "income_market_rationale": report.income_market_rationale,
        }
        return {"context": context}
    return gather_context


def _critique_node(provider: str | None, model: str | None, call_log: list[dict]):
    def critique(state: CriticState) -> dict:
        resolved_provider, resolved_model = resolve_chat_model(provider, model)
        chat = get_chat_model(resolved_provider, resolved_model, max_tokens=1200)
        # include_raw=True: keeps the raw AIMessage (and its usage_metadata)
        # alongside the parsed Pydantic result -- with_structured_output()
        # discards the raw message by default, which would silently lose
        # this call from the token/cost ledger (Tier 1).
        structured = chat.with_structured_output(CritiqueResult, include_raw=True)
        result = structured.invoke([
            SystemMessage(content=_CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(state["context"], default=str, ensure_ascii=False)),
        ])
        raw_msg = result.get("raw")
        usage = (getattr(raw_msg, "usage_metadata", None) or {}) if raw_msg is not None else {}
        call_log.append({
            "call_label": "critic",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "provider": resolved_provider,
            "model": resolved_model,
        })
        parsed = result.get("parsed")
        critique_dict = parsed.model_dump() if parsed is not None else {
            "overall_assessment": "Неуспешен структуриран отговор от модела.",
            "consistency_issues": [], "avm_commentary": None, "missing_caveats": [],
        }
        return {"critique": critique_dict}
    return critique


def build_critic_graph(db: Session, report: AppraisalReport, provider: str | None, model: str | None, call_log: list[dict]):
    graph = StateGraph(CriticState)
    graph.add_node("gather_context", _gather_context_node(db, report))
    graph.add_node("critique", _critique_node(provider, model, call_log))
    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "critique")
    graph.add_edge("critique", END)
    return graph.compile()


def run_critical_review(
    db: Session, report: AppraisalReport,
    provider: str | None = None, model: str | None = None,
) -> tuple[dict, list[dict]]:
    """Runs the graph end to end. Returns (critique_dict, call_log) -- the
    caller (tools.py's request_critical_review) is responsible for
    persisting call_log to agent_llm_calls, same division of
    responsibility as assistant_chain.run_assistant_turn's own call_log."""
    call_log: list[dict] = []
    graph = build_critic_graph(db, report, provider, model, call_log)
    final_state = graph.invoke({"report_id": str(report.id), "context": {}, "critique": {}})
    return final_state["critique"], call_log
