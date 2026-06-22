import uuid

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, ForeignKey,
    Integer, Numeric, SmallInteger, String, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from app.db.base import Base


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
    # draft | generated | exported

    subject_address = Column(Text)
    subject_city = Column(Text)
    subject_area_sqm = Column(Numeric(10, 2))
    subject_floor = Column(SmallInteger)
    subject_total_floors = Column(SmallInteger)
    subject_construction = Column(Text)
    subject_year = Column(SmallInteger)
    subject_description = Column(Text)

    concluded_value_sales = Column(Numeric(14, 2))

    annual_rent_estimate = Column(Numeric(14, 2))
    gross_rent_multiplier = Column(Numeric(6, 3))
    capitalization_rate = Column(Numeric(6, 4))
    concluded_value_income = Column(Numeric(14, 2))

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
    """Unbounded analysis pool. pinned_for_report=True → included in Word/Excel report."""
    __tablename__ = "comparable_pool"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    listing_id = Column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    comparable_type = Column(String(10), nullable=False)   # 'sale' | 'rent'
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    pinned_for_report = Column(Boolean, nullable=False, default=False)
    adjustment_pct = Column(Numeric(6, 2))
    analyst_note = Column(Text)

    __table_args__ = (
        UniqueConstraint("listing_id", "comparable_type", name="uq_pool_listing_type"),
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
