"""LangChain tool wrappers around existing service functions (Phase 7,
Tier 3) -- the structural guardrail from the plan doc: the model calls
these for numbers instead of computing arithmetic itself. build_tools()
closes over `db`/`report` so the tool signatures exposed to the model stay
simple (no DB session as a tool argument).
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.db.models import AgentLlmCall, AppraisalReport, ReportDocument
from app.services import analytics_service
from app.services.comparable_service import (
    INCOME_ASSUMPTION_BOUNDS,
    INCOME_ASSUMPTION_DEFAULTS,
    compute_income_valuation,
    compute_weighted_conclusion,
    get_pool_with_stats,
)
from app.services.llm import critic_graph
from app.services.llm.providers import estimate_cost_usd
from app.services.llm.retriever import retrieve_comparables


def _pool_stats_fn(db: Session, report: AppraisalReport):
    def get_pool_stats(comparable_type: str = "sale") -> dict:
        """Statistics (min/max/mean/median/p25/p75 price-per-sqm, area range)
        over the comparable pool already assembled for this report. Use this
        instead of estimating averages yourself. comparable_type is "sale"
        or "rent"."""
        pool = get_pool_with_stats(db, comparable_type, report.id)
        return {
            "stats": pool["stats"],
            "pinned_count": pool["pinned_count"],
            "total_count": pool["total_count"],
        }
    return get_pool_stats


def _market_trend_fn(db: Session, report: AppraisalReport):
    def get_market_trend_stats() -> dict:
        """Recent market trend (median/p25/p75 price-per-sqm, median days on
        market) for the subject's own city/geo_category/property_type, most
        recent scrape run first. Use this for broader market context beyond
        the immediate comparable pool -- do not guess a market trend."""
        trend = analytics_service.get_market_trend(
            db,
            deal_type="sale",
            geo_category=report.subject_geo_category,
            city=report.subject_city,
            property_type_slug=report.subject_property_type,
            n_runs=6,
        )
        return {"runs": trend}
    return get_market_trend_stats


def _weighted_value_fn():
    def compute_weighted_value(
        sales_value: float | None = None,
        income_value: float | None = None,
        residual_value: float | None = None,
        weight_sales_pct: float | None = None,
        weight_income_pct: float | None = None,
        weight_residual_pct: float | None = None,
    ) -> dict:
        """Normalized weighted average across whichever approach values you
        supply with a matching positive weight. Use this instead of doing
        the weighted-average arithmetic yourself when combining more than
        one valuation approach."""
        result = compute_weighted_conclusion(
            sales_value, income_value, residual_value,
            weight_sales_pct, weight_income_pct, weight_residual_pct,
        )
        return {"weighted_value": result}
    return compute_weighted_value


def _clamp(value: float, key: str) -> float:
    lo, hi = INCOME_ASSUMPTION_BOUNDS[key]
    return max(lo, min(hi, value))


def _income_valuation_description() -> str:
    """Built as a plain string (NOT an f-string docstring on the tool
    function itself -- an f-string as a function's first statement does
    NOT populate __doc__, since Python's docstring detection only fires for
    a literal string constant; it would silently leave the tool with no
    description, hiding every bound below from the model). Passed
    explicitly via StructuredTool.from_function(description=...)."""
    b, d = INCOME_ASSUMPTION_BOUNDS, INCOME_ASSUMPTION_DEFAULTS
    return (
        "Computes the income approach: direct capitalization AND a "
        "multi-year DCF with terminal value (NOI discounted at cap_rate_pct "
        "over period_years, plus a terminal value at terminal_cap_rate_pct). "
        "rent_per_sqm_month/sale_price_per_sqm should come from the rent/sale "
        "comparable stats already shown to you -- never invent them.\n\n"
        "You MAY deviate from the defaults below for the assumption "
        "parameters (e.g. a lower vacancy_pct if market data suggests a tight "
        "rental market), but ANY deviation from a default must be explained "
        "in your narrative -- state which assumption you changed and why. "
        "Values outside the allowed range are silently clamped to the "
        "nearest bound, so stay within them:\n"
        f"  expenses_pct: {b['expenses_pct'][0]}-{b['expenses_pct'][1]} (default {d['expenses_pct']})\n"
        f"  vacancy_pct: {b['vacancy_pct'][0]}-{b['vacancy_pct'][1]} (default {d['vacancy_pct']})\n"
        f"  cap_rate_pct: {b['cap_rate_pct'][0]}-{b['cap_rate_pct'][1]} (default {d['cap_rate_pct']})\n"
        f"  growth_pct: {b['growth_pct'][0]}-{b['growth_pct'][1]} (default {d['growth_pct']})\n"
        f"  period_years: {b['period_years'][0]}-{b['period_years'][1]} (default {d['period_years']})\n"
        f"  terminal_cap_rate_pct: {b['terminal_cap_rate_pct'][0]}-{b['terminal_cap_rate_pct'][1]} (default {d['terminal_cap_rate_pct']})\n"
        f"  discount_rate_pct: {b['discount_rate_pct'][0]}-{b['discount_rate_pct'][1]} (optional -- leave unset "
        "to discount the DCF at cap_rate_pct, same as direct capitalization; set explicitly only if you have "
        "a reasoned, risk-adjusted required return that differs from the cap rate, and explain why)\n\n"
        "Returns gross_yield_pct, net_yield_pct, noi_per_sqm_year, "
        "direct_value_per_sqm, dcf_value_per_sqm, dcf_rows (year-by-year), discount_rate_pct_used, "
        "terminal_value_pv_per_sqm, and sensitivity (a 5x5 grid of "
        "direct-capitalization value at cap_rate x rent variants, ±10%/±20% "
        "around the values you passed) -- all computed here, not by you. "
        "Comment on what the sensitivity grid implies (e.g. how much the "
        "value swings with cap rate) in your reasoning section."
    )


def _income_valuation_fn(report: AppraisalReport):
    d = INCOME_ASSUMPTION_DEFAULTS

    def compute_income_valuation_tool(
        rent_per_sqm_month: float,
        sale_price_per_sqm: float | None = None,
        expenses_pct: float = d["expenses_pct"],
        vacancy_pct: float = d["vacancy_pct"],
        cap_rate_pct: float = d["cap_rate_pct"],
        growth_pct: float = d["growth_pct"],
        period_years: int = d["period_years"],
        terminal_cap_rate_pct: float = d["terminal_cap_rate_pct"],
        discount_rate_pct: float | None = None,
    ) -> dict:
        clamped = {
            "expenses_pct": _clamp(expenses_pct, "expenses_pct"),
            "vacancy_pct": _clamp(vacancy_pct, "vacancy_pct"),
            "cap_rate_pct": _clamp(cap_rate_pct, "cap_rate_pct"),
            "growth_pct": _clamp(growth_pct, "growth_pct"),
            "period_years": int(_clamp(period_years, "period_years")),
            "terminal_cap_rate_pct": _clamp(terminal_cap_rate_pct, "terminal_cap_rate_pct"),
        }
        discount_rate_clamped = _clamp(discount_rate_pct, "discount_rate_pct") if discount_rate_pct is not None else None
        result = compute_income_valuation(
            rent_per_sqm_month=rent_per_sqm_month,
            sale_price_per_sqm=sale_price_per_sqm,
            discount_rate_pct=discount_rate_clamped,
            **clamped,
        )
        result["assumptions_used"] = {
            "rent_per_sqm_month": rent_per_sqm_month, "sale_price_per_sqm": sale_price_per_sqm,
            **clamped, "discount_rate_pct": discount_rate_clamped,
        }
        return result

    return compute_income_valuation_tool


def build_tools(db: Session, report: AppraisalReport) -> list[StructuredTool]:
    """Tools bound to this specific db session + report -- construct fresh
    per generation call, never reuse across requests/sessions."""
    return [
        StructuredTool.from_function(_pool_stats_fn(db, report), name="get_pool_stats"),
        StructuredTool.from_function(_market_trend_fn(db, report), name="get_market_trend_stats"),
        StructuredTool.from_function(_weighted_value_fn(), name="compute_weighted_value"),
        StructuredTool.from_function(
            _income_valuation_fn(report),
            name="compute_income_valuation",
            description=_income_valuation_description(),
        ),
    ]


# ── Assistant-only tools (Tier 2, 2026-08-26) ─────────────────────────────────
# Two additions beyond build_tools() above, for the free-form chat console:
# on-demand retrieval (the owner's own "let me prompt the retriever more
# freely" ask) and a WRITE-INTENT tool that never actually writes -- see
# propose_text_update's own docstring for why.

_PROPOSABLE_FIELDS = {
    "subject_description": "Свободното описание на оценявания имот",
    "submarket_rationale": "Обосновката на съпоставимата зона (пазарен подход)",
    "income_market_rationale": "Обосновката на доходния подход",
    "appraiser_notes": "Бележки на оценителя (свободен текст, вкл. правни/пазарни наблюдения)",
}


def _retrieve_comparables_fn(db: Session, report: AppraisalReport, comparable_type: str):
    # No docstring here -- StructuredTool.from_function's explicit
    # description= at the call site is what the model actually sees (an
    # f-string docstring would NOT populate __doc__ anyway, same trap as
    # _income_valuation_description's own note).
    def retrieve(k: int = 6) -> dict:
        comps = retrieve_comparables(db, report, k=k, comparable_type=comparable_type)
        trimmed = [
            {
                "id": c["id"], "ad_url": c.get("ad_url"),
                "location": f"{c.get('title_city_model') or ''} {c.get('title_geo_2_model') or ''}".strip(),
                "area_sqm": c.get("area_sqm_model"), "total_price": c.get("total_price"),
                "price_per_sqm": c.get("price_per_sqm_model"),
                "construction": c.get("construction_type_model"), "year": c.get("construction_year_model"),
                "closeness_pct": round((1 - float(c["distance"])) * 100) if c.get("distance") is not None else None,
            }
            for c in comps
        ]
        return {"comparables": trimmed, "count": len(trimmed)}

    return retrieve


def _propose_text_update_fn(report: AppraisalReport):
    def propose_text_update(field: str, new_text: str) -> dict:
        """Propose new text for one of the report's free-text fields:
        subject_description, submarket_rationale, income_market_rationale, or
        appraiser_notes (general notes -- use this one for legal/market-analysis
        observations that don't fit the other three).
        This does NOT save anything -- it only returns a proposal that the
        appraiser sees as a card in the chat with an explicit "Приложи"
        (Apply) button. Nothing is ever written to the report without the
        appraiser clicking it, same as every other AI-generated text
        elsewhere in this app. Call this when the appraiser asks you to
        write/update/draft one of these fields; do not just print the text
        in your own reply instead of calling this."""
        if field not in _PROPOSABLE_FIELDS:
            return {"error": f"Unknown field {field!r}. Valid: {sorted(_PROPOSABLE_FIELDS)}"}
        return {
            "proposed": True,
            "field": field,
            "field_label": _PROPOSABLE_FIELDS[field],
            "text": new_text,
        }
    return propose_text_update


def _list_documents_fn(db: Session, report: AppraisalReport):
    def list_documents() -> dict:
        """Lists uploaded documents for this report (id, filename, type,
        status). Call this first if you're not sure which document_id to
        pass to read_document, or the appraiser asks what's been uploaded."""
        docs = (
            db.query(ReportDocument)
            .filter(ReportDocument.report_id == report.id)
            .order_by(ReportDocument.created_at.desc())
            .all()
        )
        return {
            "documents": [
                {"id": str(d.id), "filename": d.filename, "document_type": d.document_type, "status": d.status}
                for d in docs
            ]
        }
    return list_documents


def _read_document_fn(db: Session, report: AppraisalReport):
    def read_document(document_id: str) -> dict:
        """Returns the extracted structured facts for one uploaded document
        (notarial act, company document, or скица -- see list_documents for
        available ids). Use this when the appraiser asks you to use
        information from an uploaded document, e.g. to enrich the property
        description or explain what a скица's terraces mean for area."""
        doc = (
            db.query(ReportDocument)
            .filter(ReportDocument.id == document_id, ReportDocument.report_id == report.id)
            .first()
        )
        if doc is None:
            return {"error": "Документът не е намерен за този доклад."}
        if doc.status != "ready":
            return {"status": doc.status, "error": doc.error_message}
        return {
            "filename": doc.filename,
            "document_type": doc.document_type,
            "extraction_method": doc.extraction_method,
            "data": doc.extracted_data,
        }
    return read_document


def _critical_review_fn(db: Session, report: AppraisalReport, conversation_id, provider: str | None, model: str | None):
    def request_critical_review() -> dict:
        """Runs a SEPARATE, structured critical review of the report's
        current state (comparables statistics, computed values, AVM
        prediction if attached, already-written texts) -- checks whether
        the narrative matches the numbers, comments on the AVM prediction,
        and flags missing caveats given the report's purpose. Read-only,
        never writes to the report. Call this when the appraiser asks for a
        critical review, sanity check, or second opinion on the draft --
        this is a genuinely different, more thorough pass than reasoning
        about it yourself in this reply."""
        critique, call_log = critic_graph.run_critical_review(db, report, provider, model)
        if call_log and conversation_id is not None:
            try:
                for entry in call_log:
                    cost = estimate_cost_usd(entry["model"], entry["input_tokens"], entry["output_tokens"], provider=entry["provider"])
                    db.add(AgentLlmCall(
                        conversation_id=conversation_id, call_label=entry["call_label"],
                        provider=entry["provider"], model=entry["model"],
                        input_tokens=entry["input_tokens"], output_tokens=entry["output_tokens"],
                        estimated_cost_usd=cost,
                    ))
                db.commit()
            except Exception:
                db.rollback()
        return critique
    return request_critical_review


def build_assistant_tools(
    db: Session, report: AppraisalReport,
    conversation_id=None, provider: str | None = None, model: str | None = None,
) -> list[StructuredTool]:
    """build_tools() (read-only stats/compute) plus retrieval-on-demand, the
    write-intent proposal tool, document reading, and the critic graph --
    the full toolbox for the chat console's conversational agent.
    conversation_id/provider/model are only used to log request_critical_review's
    own LLM call to agent_llm_calls under the right conversation; the
    critique itself always uses `provider`/`model` (the conversation's
    current model choice, not necessarily a separate "critic model")."""
    tools = build_tools(db, report)
    tools.append(StructuredTool.from_function(_list_documents_fn(db, report), name="list_documents"))
    tools.append(StructuredTool.from_function(_read_document_fn(db, report), name="read_document"))
    tools.append(StructuredTool.from_function(
        _critical_review_fn(db, report, conversation_id, provider, model), name="request_critical_review",
    ))
    tools.append(StructuredTool.from_function(
        _retrieve_comparables_fn(db, report, "sale"), name="retrieve_sale_comparables",
        description="Semantic search for sale comparables closest to the subject property (up to k, default 6).",
    ))
    tools.append(StructuredTool.from_function(
        _retrieve_comparables_fn(db, report, "rent"), name="retrieve_rent_comparables",
        description="Semantic search for rent comparables closest to the subject property (up to k, default 6).",
    ))
    tools.append(StructuredTool.from_function(
        _propose_text_update_fn(report), name="propose_text_update",
        description=(
            "Propose new text for subject_description, submarket_rationale, or "
            "income_market_rationale. Never writes anything -- returns a proposal "
            "the appraiser must explicitly apply. Always use this tool (not a plain "
            "reply) when asked to draft/update one of these fields."
        ),
    ))
    return tools
