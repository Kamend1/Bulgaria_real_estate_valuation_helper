"""Tools for the market analyst agent (Phase 10, 2026-08-28) -- a second,
report-agnostic agent for free-form market research across the whole
listings corpus, distinct from app/services/llm/tools.py's report-scoped
toolbox. Same structural guardrail as that module: the model gets numbers
only by calling these, never by computing them itself.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.db.models import MarketDocument
from app.services import analytics_service

_GROUP_BY_VALUES = ("run", "geo_category", "city", "neighborhood", "construction_type", "property_type_slug")


def _query_market_stats_description() -> str:
    return (
        "Computes real market statistics (n_listings, median/p25/p75/mean price-per-sqm, "
        "median days-on-market) over the whole listings corpus, filtered however you need. "
        "NEVER estimate or guess these numbers yourself -- always call this tool.\n\n"
        "deal_type: \"sale\" or \"rent\" (required).\n"
        "property_type_slugs/geo_categories/cities/construction_types: optional lists -- "
        "e.g. pass both \"dvustaen\" and \"tristaen\" together to combine them in one query. "
        "Use list_market_filter_values first if you're not sure of the exact spelling/slug.\n"
        "neighborhoods: optional list, matched with ILIKE (partial match tolerant).\n"
        "area_min/area_max: optional sqm bounds, e.g. for office-size bands.\n"
        "group_by: \"run\" (default) returns a chronological time series across the last "
        "n_runs scrape runs for the given filters -- use this for trend-over-time questions. "
        "Any other value (\"geo_category\"/\"city\"/\"neighborhood\"/\"construction_type\"/"
        "\"property_type_slug\") returns one row PER DISTINCT VALUE of that dimension, "
        "pooling the last n_runs together -- use this to compare neighborhoods against each "
        "other, or old vs. new construction against each other, etc. Call this tool once per "
        "dimension value if you need an even narrower side-by-side (e.g. one call per "
        "neighborhood with a single-item cities filter) -- but group_by is usually the "
        "faster way to get all values in one call.\n"
        "n_runs: how many of the most recent scrape runs to include (default 6 -- there are "
        "only a handful of full scrape runs total so far, so this usually covers the whole "
        "available history; do not assume it means a dense daily series)."
    )


def _query_market_stats_fn(db: Session):
    def query_market_stats(
        deal_type: str,
        property_type_slugs: list[str] | None = None,
        geo_categories: list[str] | None = None,
        cities: list[str] | None = None,
        neighborhoods: list[str] | None = None,
        construction_types: list[str] | None = None,
        area_min: float | None = None,
        area_max: float | None = None,
        group_by: str = "run",
        n_runs: int = 6,
    ) -> dict:
        if group_by not in _GROUP_BY_VALUES:
            return {"error": f"Invalid group_by {group_by!r}. Valid: {list(_GROUP_BY_VALUES)}"}
        return analytics_service.get_market_stats_flexible(
            db, deal_type=deal_type,
            property_type_slugs=property_type_slugs, geo_categories=geo_categories,
            cities=cities, neighborhoods=neighborhoods, construction_types=construction_types,
            area_min=area_min, area_max=area_max, group_by=group_by, n_runs=n_runs,
        )
    return query_market_stats


def _list_market_filter_values_fn(db: Session):
    def list_market_filter_values(city: str | None = None) -> dict:
        """Real, currently-queryable filter values -- geo_category enum, property-type
        segments (with their exact slugs), top cities by listing count, construction
        types, and (if you pass a city) top neighborhoods for that city. Call this
        before guessing a city/neighborhood/construction_type spelling for
        query_market_stats -- an exact-string filter on a wrong spelling silently
        returns zero rows, not an error."""
        return analytics_service.get_market_filter_values(db, city=city)
    return list_market_filter_values


def _list_market_documents_fn(db: Session):
    def list_market_documents() -> dict:
        """Lists documents in the shared market-research library (id, filename, type,
        status) -- market reports, research articles, official statistics uploaded by
        any user. Call this first if you're not sure which document_id to pass to
        read_market_document, or the appraiser asks what's been uploaded."""
        docs = db.query(MarketDocument).order_by(MarketDocument.created_at.desc()).all()
        return {
            "documents": [
                {"id": str(d.id), "filename": d.filename, "document_type": d.document_type, "status": d.status}
                for d in docs
            ]
        }
    return list_market_documents


def _read_market_document_fn(db: Session):
    def read_market_document(document_id: str) -> dict:
        """Returns the extracted facts (source, claims, cited figures) for one document
        in the market-research library. After reading a document's claims, cross-check
        them with query_market_stats for the relevant segment/period and state explicitly
        whether the real data agrees or disagrees -- do not just repeat the document's
        claims as if verified."""
        doc = db.query(MarketDocument).filter(MarketDocument.id == document_id).first()
        if doc is None:
            return {"error": "Документът не е намерен."}
        if doc.status != "ready":
            return {"status": doc.status, "error": doc.error_message}
        return {
            "filename": doc.filename,
            "document_type": doc.document_type,
            "extraction_method": doc.extraction_method,
            "data": doc.extracted_data,
        }
    return read_market_document


def build_analyst_tools(db: Session) -> list[StructuredTool]:
    """Tools bound to this db session -- construct fresh per turn, never
    reuse across requests. No report/conversation_id closed over here
    (unlike build_assistant_tools) -- this agent has no report to write to
    and no critic sub-call to log separately; its own turn-level token
    ledger entries are logged the same way run_assistant_turn already does."""
    return [
        StructuredTool.from_function(
            _query_market_stats_fn(db), name="query_market_stats",
            description=_query_market_stats_description(),
        ),
        StructuredTool.from_function(_list_market_filter_values_fn(db), name="list_market_filter_values"),
        StructuredTool.from_function(_list_market_documents_fn(db), name="list_market_documents"),
        StructuredTool.from_function(_read_market_document_fn(db), name="read_market_document"),
    ]
