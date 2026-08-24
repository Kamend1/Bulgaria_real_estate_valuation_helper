"""LangChain tool wrappers around existing service functions (Phase 7,
Tier 3) -- the structural guardrail from the plan doc: the model calls
these for numbers instead of computing arithmetic itself. build_tools()
closes over `db`/`report` so the tool signatures exposed to the model stay
simple (no DB session as a tool argument).
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport
from app.services import analytics_service
from app.services.comparable_service import (
    INCOME_ASSUMPTION_BOUNDS,
    INCOME_ASSUMPTION_DEFAULTS,
    compute_income_valuation,
    compute_weighted_conclusion,
    get_pool_with_stats,
)


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
        f"  terminal_cap_rate_pct: {b['terminal_cap_rate_pct'][0]}-{b['terminal_cap_rate_pct'][1]} (default {d['terminal_cap_rate_pct']})\n\n"
        "Returns gross_yield_pct, net_yield_pct, noi_per_sqm_year, "
        "direct_value_per_sqm, dcf_value_per_sqm, dcf_rows (year-by-year), "
        "terminal_value_pv_per_sqm -- all computed here, not by you."
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
    ) -> dict:
        clamped = {
            "expenses_pct": _clamp(expenses_pct, "expenses_pct"),
            "vacancy_pct": _clamp(vacancy_pct, "vacancy_pct"),
            "cap_rate_pct": _clamp(cap_rate_pct, "cap_rate_pct"),
            "growth_pct": _clamp(growth_pct, "growth_pct"),
            "period_years": int(_clamp(period_years, "period_years")),
            "terminal_cap_rate_pct": _clamp(terminal_cap_rate_pct, "terminal_cap_rate_pct"),
        }
        result = compute_income_valuation(
            rent_per_sqm_month=rent_per_sqm_month,
            sale_price_per_sqm=sale_price_per_sqm,
            **clamped,
        )
        result["assumptions_used"] = {"rent_per_sqm_month": rent_per_sqm_month, "sale_price_per_sqm": sale_price_per_sqm, **clamped}
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
