import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, ForeignKey,
    Index, Integer, Numeric, SmallInteger, String, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from app.db.base import Base

# text-embedding-3-small's default dimensionality -- see alembic/versions/
# 0019_ai_valuation.py for the migration-side rationale.
EMBEDDING_DIM = 1536


# ── Taxonomy tables ────────────────────────────────────────────────────────────

class TaxonomyPropertyType(Base):
    """22 canonical property-type slugs from imot.bg sitemap."""
    __tablename__ = "taxonomy_property_types"

    slug = Column(Text, primary_key=True)          # e.g. "dvustaen"
    display_name_bg = Column(Text)                  # e.g. "Двустаен апартамент"
    route_count = Column(Integer)


class TaxonomyGeoPath(Base):
    """Scraping routes from imot.bg sitemap (geo_1 = city, geo_2 = quarter)."""
    __tablename__ = "taxonomy_geo_paths"

    deal_type = Column(Text, primary_key=True)      # "prodazhbi" | "naemi"
    geo_path = Column(Text, primary_key=True)       # "grad-sofia/lozenets"
    geo_level_count = Column(SmallInteger)
    geo_1 = Column(Text)                            # "grad-sofia"
    geo_2 = Column(Text)                            # "lozenets"
    geo_3 = Column(Text)
    route_count = Column(Integer)


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(TIMESTAMP(timezone=True))
    status = Column(String(20), nullable=False, default="running")
    # running | completed | failed | imported

    deal_types = Column(ARRAY(Text))
    geo_paths = Column(ARRAY(Text))
    property_types = Column(ARRAY(Text))

    routes_total = Column(Integer)
    routes_done = Column(Integer, default=0)
    listings_found = Column(Integer, default=0)
    listings_upserted = Column(Integer, default=0)
    listings_downloaded = Column(Integer, default=0)
    listings_ingested = Column(Integer, default=0)
    error_message = Column(Text)
    notes = Column(Text)

    # Process management
    pid = Column(Integer)
    phase = Column(String(20))
    last_heartbeat_at = Column(TIMESTAMP(timezone=True))
    stop_requested = Column(Boolean, default=False, nullable=False)
    last_message = Column(Text)
    log_text = Column(Text)  # full accumulated stdout log, written at completion

    snapshots = relationship("ListingSnapshot", back_populates="scrape_run")


class Listing(Base):
    __tablename__ = "listings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ad_url = Column(Text, nullable=False, unique=True)
    ad_id = Column(Text)

    first_seen_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_scrape_run_id = Column(UUID(as_uuid=True), ForeignKey("scrape_runs.id"))

    # --- Raw parsed fields ---
    listing_type = Column(Text)
    deal_raw = Column(Text)
    property_type_raw = Column(Text)
    title = Column(Text)
    title_city_raw = Column(Text)
    title_geo_2_raw = Column(Text)
    location_raw = Column(Text)

    total_price = Column(Numeric(14, 2))
    currency = Column(String(5))
    price_raw = Column(Text)
    vat_status = Column(Text)
    area_sqm = Column(Numeric(10, 2))
    price_per_sqm = Column(Numeric(10, 2))

    floor = Column(SmallInteger)
    total_floors = Column(SmallInteger)
    construction_type = Column(Text)
    construction_year = Column(SmallInteger)

    description_clean = Column(Text)
    features_pipe = Column(Text)
    features_count = Column(SmallInteger)

    views = Column(Integer)
    published_raw = Column(Text)
    training_eligible = Column(Boolean, default=False)
    html_path = Column(Text)
    parse_error = Column(Text)

    # --- Engineered fields ---
    deal_type_normalized = Column(String(10))          # sale | rent | unknown
    published_date = Column(Date)

    price_per_sqm_model = Column(Numeric(10, 2))       # recalculated from total_price / area_sqm_model
    area_sqm_model = Column(Numeric(10, 2))            # area after agricultural decare correction

    title_city_model = Column(Text)
    title_geo_2_model = Column(Text)

    location_level_1 = Column(Text)
    location_level_2 = Column(Text)
    location_level_3 = Column(Text)
    location_level_1_model = Column(Text)
    location_level_2_model = Column(Text)
    location_level_3_model = Column(Text)

    property_type_slug = Column(Text)               # taxonomy slug extracted from ad_url

    geo_category = Column(Text)
    exclude_foreign = Column(Boolean, default=False)

    construction_type_model = Column(Text)
    construction_year_model = Column(SmallInteger)

    floor_model = Column(SmallInteger)
    total_floors_model = Column(SmallInteger)
    floor_applicability = Column(Text)

    # Archiving (Phase 5)
    status = Column(String(20), default="active", nullable=False)   # active | archived
    archived_at = Column(TIMESTAMP(timezone=True))
    archived_by_run_id = Column(UUID(as_uuid=True), ForeignKey("scrape_runs.id"))

    snapshots = relationship("ListingSnapshot", back_populates="listing", cascade="all, delete-orphan")
    report_comparables = relationship("ReportComparable", back_populates="listing")


class ListingSnapshot(Base):
    __tablename__ = "listing_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    listing_id = Column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    scrape_run_id = Column(UUID(as_uuid=True), ForeignKey("scrape_runs.id"), nullable=False)
    scraped_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    total_price = Column(Numeric(14, 2))
    currency = Column(String(5))
    price_per_sqm_model = Column(Numeric(10, 2))
    area_sqm_model = Column(Numeric(10, 2))
    vat_status = Column(Text)
    views = Column(Integer)

    days_on_market = Column(Integer)

    parsed_data = Column(JSONB)

    listing = relationship("Listing", back_populates="snapshots")
    scrape_run = relationship("ScrapeRun", back_populates="snapshots")


class AppraisalReport(Base):
    __tablename__ = "appraisal_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    title = Column(Text, nullable=False)
    status = Column(String(20), default="draft")
    # draft | finalized | exported
    report_purpose = Column(String(30), nullable=False, default="market_opinion")
    # market_opinion | fair_value_ifrs | noncash_contribution -- selects which
    # front-matter boilerplate (purpose/standard-of-value text) generate_docx()
    # uses; see comparable_service._PURPOSE_TEXTS.
    is_scratch = Column(Boolean, nullable=False, default=False, server_default="false")
    # Phase 11, 2026-08-28: a "hypothetical property" scenario -- a normal
    # AppraisalReport in every respect (gets every subject/income/weighting
    # field and the full comparables/AI-assistant toolset for free), just
    # hidden from /reports/'s default list so exploratory scenarios don't
    # clutter real casework. "Направи истински доклад" flips this to False;
    # nothing else about the row changes on promotion.

    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    owner = relationship("User", back_populates="reports")

    subject_address = Column(Text)
    subject_city = Column(Text)
    subject_area_sqm = Column(Numeric(10, 2))
    subject_floor = Column(SmallInteger)
    subject_total_floors = Column(SmallInteger)
    subject_construction = Column(Text)
    subject_year = Column(SmallInteger)
    subject_description = Column(Text)
    subject_property_type = Column(Text)        # taxonomy slug, e.g. "dvustaen" | "ofis"
    subject_geo_category = Column(Text)          # one of map_geo_category's 8 buckets
    subject_neighborhood = Column(Text)          # matches listings.title_geo_2_model
    subject_cadastral_id = Column(Text)           # AGKK ПИ identifier, e.g. "68134.1234.567"

    concluded_value_sales = Column(Numeric(14, 2))
    concluded_value_sales_source = Column(String(20))   # avm | manual

    legal_description = Column(Text)              # generated (utils.gis) or manually edited
    legal_description_source = Column(String(20))  # agkk | manual

    submarket_rationale = Column(Text)   # why these comparables/this zone were chosen (F8)
    income_market_rationale = Column(Text)   # AI/manual narrative for the income approach (Phase 7, Tier 5)

    annual_rent_estimate = Column(Numeric(14, 2))
    gross_rent_multiplier = Column(Numeric(6, 3))
    capitalization_rate = Column(Numeric(6, 4))
    concluded_value_income = Column(Numeric(14, 2))
    concluded_value_residual = Column(Numeric(14, 2))

    # Full compute_income_valuation() result (NOI, direct-cap, DCF, yields,
    # sensitivity grid, assumptions_used) behind concluded_value_income above
    # -- without this, the report only ever kept the single final number,
    # with no way to show how it was derived (audit finding, 2026-08-25).
    # source mirrors concluded_value_sales_source's manual|avm pattern.
    income_valuation_detail = Column(JSONB)
    income_valuation_source = Column(String(20))   # manual | ai

    # Weight (%) each approach carries in the final concluded_value below --
    # only approaches with both a weight and a saved concluded_value_* are
    # included in the weighted average (see comparable_service.update_conclusion).
    weight_sales_pct = Column(Numeric(5, 2))
    weight_income_pct = Column(Numeric(5, 2))
    weight_residual_pct = Column(Numeric(5, 2))
    weighting_rationale = Column(Text)   # appraiser's written reasoning for the weights chosen

    concluded_value = Column(Numeric(14, 2))
    concluded_currency = Column(String(5), default="EUR")
    valuation_date = Column(Date)
    appraiser_notes = Column(Text)

    docx_path = Column(Text)
    pdf_path = Column(Text)

    comparables = relationship(
        "ReportComparable",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="ReportComparable.comparable_type, ReportComparable.position",
    )


class ComparablePool(Base):
    """Per-report analysis pool. pinned_for_report=True → included in Word/Excel report."""
    __tablename__ = "comparable_pool"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    listing_id = Column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    comparable_type = Column(String(10), nullable=False)   # 'sale' | 'rent'
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    pinned_for_report = Column(Boolean, nullable=False, default=False)
    adjustment_pct = Column(Numeric(6, 2))
    # Named-factor breakdown, e.g. {"market": -5, "location": 7, "size": -11,
    # "floor": 0, "condition": 0} -- when set, adjustment_pct above is
    # DERIVED as the sum of these and kept in sync (see
    # comparable_service.update_pool_adjustment). Null/empty means the
    # appraiser is using the older single-blended-% entry mode instead.
    adjustment_factors = Column(JSONB)
    analyst_note = Column(Text)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=True)

    __table_args__ = (
        UniqueConstraint("listing_id", "comparable_type", "report_id", name="uq_pool_listing_ctype_report"),
    )

    listing = relationship("Listing")


class ReportComparable(Base):
    __tablename__ = "report_comparables"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False)
    listing_id = Column(BigInteger, ForeignKey("listings.id"), nullable=False)
    comparable_type = Column(String(10), nullable=False)  # sale | rent
    position = Column(SmallInteger, nullable=False)        # 1–6 per type per report
    adjustment_pct = Column(Numeric(6, 2))
    analyst_note = Column(Text)

    __table_args__ = (
        UniqueConstraint("report_id", "comparable_type", "listing_id"),
        UniqueConstraint("report_id", "comparable_type", "position"),
    )

    report = relationship("AppraisalReport", back_populates="comparables")
    listing = relationship("Listing", back_populates="report_comparables")


class AvmModel(Base):
    """One row per trained AVM pipeline. Only one is_active=True row per segment."""
    __tablename__ = "avm_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment = Column(Text, nullable=False)          # residential | office | retail | industrial | hospitality
    trained_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    algorithm = Column(Text, nullable=False)

    feature_columns = Column(JSONB, nullable=False)
    hyperparams = Column(JSONB, nullable=False)
    metrics = Column(JSONB)
    target_transform = Column(Text, nullable=False, default="raw")   # raw | log1p

    training_row_count = Column(Integer, nullable=False)
    min_row_threshold = Column(Integer, nullable=False)

    model_path = Column(Text, nullable=False)
    quantile_low_path = Column(Text)
    quantile_high_path = Column(Text)

    is_active = Column(Boolean, nullable=False, default=False)
    notes = Column(Text)

    # Optional second model blended with this row's primary algorithm.
    # blend_weight = weight on THIS row's algorithm; companion gets
    # (1 - blend_weight). NULL means "single model, no blend" — the
    # default for every segment unless explicitly configured otherwise.
    companion_algorithm = Column(Text)
    companion_model_path = Column(Text)
    companion_quantile_low_path = Column(Text)
    companion_quantile_high_path = Column(Text)
    blend_weight = Column(Numeric(4, 3))

    # TF-IDF+SVD transformer fit on training descriptions (Round 3).
    # NULL for segments where it showed no benefit (office).
    text_transformer_path = Column(Text)


class ListingEmbedding(Base):
    """Semantic-text embedding of one listing (app/services/llm/listing_doc.py
    builds the text; app/services/llm/embeddings.py builds the vector). One
    row per (listing, provider, model) -- re-embedding upserts, see the
    unique constraint."""
    __tablename__ = "listing_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    listing_id = Column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    provider = Column(Text, nullable=False)     # e.g. "openai"
    model = Column(Text, nullable=False)        # e.g. "text-embedding-3-small"
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    embedded_text = Column(Text, nullable=False)   # the exact text that was embedded, for debugging/audit
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("listing_id", "provider", "model", name="uq_listing_embeddings_listing_provider_model"),
    )

    listing = relationship("Listing")


class AiValuationRun(Base):
    """One AI-assisted valuation generation call -- cost/audit trail, mirrors
    the AvmModel/ScrapeRun provenance pattern. `output` holds the full
    structured ValuationBackbone response (JSONB)."""
    __tablename__ = "ai_valuation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False)
    provider = Column(Text, nullable=False)
    model = Column(Text, nullable=False)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    estimated_cost_usd = Column(Numeric(10, 4))
    output = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    report = relationship("AppraisalReport")


class AgentLlmCall(Base):
    """One individual LLM API call -- finer-grained than AiValuationRun,
    which logs one row per whole generation with tokens SUMMED across every
    internal call in its tool-calling loop. Added for the multi-agent chat
    console (Tier 1, 2026-08-26): "which step ate the budget" needs
    per-call detail an aggregate can't show (see the gpt-5.4-pro truncation
    audit, 2026-08-25, where this distinction mattered directly).
    ai_valuation_run_id links calls belonging to the existing single-shot
    generation flow; conversation_id is reserved, unused until Tier 2's
    chat console exists -- a call may eventually have one or the other, not
    necessarily both."""
    __tablename__ = "agent_llm_calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ai_valuation_run_id = Column(UUID(as_uuid=True), ForeignKey("ai_valuation_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    call_label = Column(Text, nullable=False)   # e.g. "tool_loop_1", "fallback_stream", (later) "critic"
    provider = Column(Text, nullable=False)
    model = Column(Text, nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Numeric(10, 6))
    notes = Column(Text)   # Phase 13, 2026-09-01: Supervisor's RouteDecision.reasoning, unused by every other call type
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


class AgentConversation(Base):
    """One chat conversation with an AI agent (Tier 2, 2026-08-26; extended
    for the report-agnostic market analyst, Phase 10, 2026-08-28).

    Two kinds share this table rather than duplicating it (agent_type
    discriminates):
    - "report_assistant" (v1 default) -- scoped to a single report via
      report_id; the owner's own framing was that the end goal is still
      writing an appraisal report.
    - "market_analyst" -- report_id is NULL; one persistent conversation
      per user for free-form market research (time series, segment/geo/
      construction-type cross-sections) across the whole listings corpus,
      not tied to any one case.

    Phase 12 (2026-08-31): a user can now have MULTIPLE conversations per
    (user_id, agent_type, report_id) -- get_or_create_conversation() still
    picks "the latest" as a fallback, but new_conversation() lets the
    appraiser start a fresh, separate thread (see assistant_chain.py). The
    composite index below backs list_conversations()'s switcher query,
    which never mattered before this -- the old "always reuse the single
    row" pattern never listed more than one."""
    __tablename__ = "agent_conversations"
    __table_args__ = (
        Index("ix_agent_conversations_user_agent_report", "user_id", "agent_type", "report_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=True, index=True)
    agent_type = Column(Text, nullable=False, default="report_assistant")   # report_assistant | market_analyst
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    report = relationship("AppraisalReport")
    messages = relationship("AgentMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="AgentMessage.created_at")


class AgentMessage(Base):
    """One turn in an AgentConversation -- enough to reconstruct the
    LangChain message list on the next turn (role/content, plus tool_calls
    for an assistant message that called tools, or tool_call_id for the
    matching tool-result message).

    truncated (Phase 12, 2026-08-31): set on an assistant message whose
    text was cut short by the provider's max-tokens ceiling (detected via
    finish_reason/stop_reason, see providers.is_length_truncated) --
    lets the UI show an explicit "response was cut off" warning instead of
    silently treating a truncated answer as a complete one. Deliberately a
    real column, not a string sentinel appended to content -- queryable,
    and never risks colliding with real model output."""
    __tablename__ = "agent_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Text, nullable=False)   # user | assistant | tool
    content = Column(Text)
    tool_calls = Column(JSONB)            # [{id, name, args}]
    tool_call_id = Column(Text)
    truncated = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("AgentConversation", back_populates="messages")


class ReportCompileRun(Base):
    """Provenance record for the "Компилирай доклада" action (Phase 13,
    2026-09-01) -- app/routers/comparables.py's compile route runs several
    specialists sequentially against one report (not a chat conversation)
    and logs which ones were requested here, mirroring ScrapeRun's own
    role as a background-action progress/provenance record."""
    __tablename__ = "report_compile_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    requested_domains = Column(ARRAY(Text), nullable=False)
    provider = Column(Text)
    model = Column(Text)
    status = Column(Text, nullable=False, default="running", server_default="running")   # running | done | error
    error_message = Column(Text)
    results = Column(JSONB)   # {domain: {"text": str, "proposals": [...]}} -- filled once status='done'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    report = relationship("AppraisalReport")


class ReportAgentFinding(Base):
    """Append-only log of every specialist's final answer for a report
    (Phase 13, 2026-09-01) -- mirrors ListingSnapshot's append-only pattern
    (history of rows, never one blob overwritten in place). Closes the gap
    found while auditing orchestrator_graph.py: OrchestratorState.findings
    only ever lived for one chat turn, and separate conversations on the
    same report had zero awareness of each other. get_report_memory()
    (app/services/llm/report_memory.py) reads the latest row per domain --
    seen by the Supervisor's routing prompt and by the Report Compiler
    action, regardless of which conversation (or compile run) produced it.

    source_id points at agent_conversations.id (source='chat') or
    report_compile_runs.id (source='compile') -- deliberately no hard FK,
    same polymorphic-provenance choice already made for
    ai_valuation_runs.output."""
    __tablename__ = "report_agent_findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    domain = Column(Text, nullable=False)   # income | market | market_analysis | legal | auditor
    source = Column(Text, nullable=False)   # chat | compile
    source_id = Column(UUID(as_uuid=True))
    summary = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    report = relationship("AppraisalReport")


class ReportDocument(Base):
    """One uploaded document (Tier 3, 2026-08-26) -- notarial act, company
    founding/partnership document, or a floor-plan скица. Scoped to a
    report, same v1 framing as AgentConversation. extracted_data's shape
    depends on document_type (see app/services/documents.py's per-type
    Pydantic schemas) -- deliberately generic JSONB, not per-type columns."""
    __tablename__ = "report_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("appraisal_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(Text, nullable=False)
    document_type = Column(Text, nullable=False)   # notarial_act | founding_document | partnership_agreement | sketch | other
    storage_path = Column(Text, nullable=False)     # relative to settings.documents_dir
    mime_type = Column(Text)
    status = Column(Text, nullable=False, default="processing")   # processing | ready | failed
    extraction_method = Column(Text)   # text | ocr_vision
    extracted_data = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    report = relationship("AppraisalReport")


class MarketDocument(Base):
    """One uploaded document in the market analyst's reference library
    (Phase 10, 2026-08-28) -- market reports, research articles, official
    statistics, etc. Deliberately NOT scoped to a report or a user: a
    shared reference library across the whole app, mirroring how the
    underlying listings corpus itself is shared, not per-user siloed (see
    app/services/market_documents.py). extracted_data stays purely
    descriptive (what the document says) -- cross-referencing it against
    live imot.bg data happens dynamically in conversation via
    query_market_stats, not baked in at upload time."""
    __tablename__ = "market_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(Text, nullable=False)
    document_type = Column(Text, nullable=False)   # market_report | research_article | government_statistic | news | other
    storage_path = Column(Text, nullable=False)     # relative to settings.market_documents_dir
    mime_type = Column(Text)
    status = Column(Text, nullable=False, default="processing")   # processing | ready | failed
    extraction_method = Column(Text)   # text | ocr_vision
    extracted_data = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


# ── Auth ───────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    appraiser_certificate_no = Column(String(100))   # КНОБ REV certificate no., shown in report front-matter
    role = Column(String(20), nullable=False, default="user")   # user | admin
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_login_at = Column(TIMESTAMP(timezone=True))

    consents = relationship("UserConsent", back_populates="user", cascade="all, delete-orphan")
    reports = relationship(
        "AppraisalReport",
        back_populates="owner",
        order_by="AppraisalReport.updated_at.desc()",
        cascade="all, delete-orphan",
    )


class UserConsent(Base):
    __tablename__ = "user_consents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    consent_type = Column(String(50), nullable=False)   # privacy_policy | terms
    accepted = Column(Boolean, nullable=False, default=True)
    version = Column(String(20), nullable=False)
    accepted_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    revoked_at = Column(TIMESTAMP(timezone=True))
    ip_address = Column(String(45))

    user = relationship("User", back_populates="consents")
