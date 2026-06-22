"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-19
"""

from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scrape_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("deal_types", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("geo_paths", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("property_types", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("routes_total", sa.Integer(), nullable=True),
        sa.Column("routes_done", sa.Integer(), server_default="0", nullable=True),
        sa.Column("listings_found", sa.Integer(), server_default="0", nullable=True),
        sa.Column("listings_upserted", sa.Integer(), server_default="0", nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "listings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ad_url", sa.Text(), nullable=False),
        sa.Column("ad_id", sa.Text(), nullable=True),
        sa.Column("first_seen_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_seen_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_scrape_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Raw parsed
        sa.Column("listing_type", sa.Text(), nullable=True),
        sa.Column("deal_raw", sa.Text(), nullable=True),
        sa.Column("property_type_raw", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("title_city_raw", sa.Text(), nullable=True),
        sa.Column("title_geo_2_raw", sa.Text(), nullable=True),
        sa.Column("location_raw", sa.Text(), nullable=True),
        sa.Column("total_price", sa.Numeric(14, 2), nullable=True),
        sa.Column("currency", sa.String(5), nullable=True),
        sa.Column("price_raw", sa.Text(), nullable=True),
        sa.Column("vat_status", sa.Text(), nullable=True),
        sa.Column("area_sqm", sa.Numeric(10, 2), nullable=True),
        sa.Column("price_per_sqm", sa.Numeric(10, 2), nullable=True),
        sa.Column("floor", sa.SmallInteger(), nullable=True),
        sa.Column("total_floors", sa.SmallInteger(), nullable=True),
        sa.Column("construction_type", sa.Text(), nullable=True),
        sa.Column("construction_year", sa.SmallInteger(), nullable=True),
        sa.Column("description_clean", sa.Text(), nullable=True),
        sa.Column("features_pipe", sa.Text(), nullable=True),
        sa.Column("features_count", sa.SmallInteger(), nullable=True),
        sa.Column("views", sa.Integer(), nullable=True),
        sa.Column("published_raw", sa.Text(), nullable=True),
        sa.Column("training_eligible", sa.Boolean(), server_default="false", nullable=True),
        sa.Column("html_path", sa.Text(), nullable=True),
        sa.Column("parse_error", sa.Text(), nullable=True),
        # Engineered
        sa.Column("deal_type_normalized", sa.String(10), nullable=True),
        sa.Column("published_date", sa.Date(), nullable=True),
        sa.Column("price_per_sqm_model", sa.Numeric(10, 2), nullable=True),
        sa.Column("area_sqm_model", sa.Numeric(10, 2), nullable=True),
        sa.Column("title_city_model", sa.Text(), nullable=True),
        sa.Column("title_geo_2_model", sa.Text(), nullable=True),
        sa.Column("location_level_1", sa.Text(), nullable=True),
        sa.Column("location_level_2", sa.Text(), nullable=True),
        sa.Column("location_level_3", sa.Text(), nullable=True),
        sa.Column("location_level_1_model", sa.Text(), nullable=True),
        sa.Column("location_level_2_model", sa.Text(), nullable=True),
        sa.Column("location_level_3_model", sa.Text(), nullable=True),
        sa.Column("geo_category", sa.Text(), nullable=True),
        sa.Column("exclude_foreign", sa.Boolean(), server_default="false", nullable=True),
        sa.Column("construction_type_model", sa.Text(), nullable=True),
        sa.Column("construction_year_model", sa.SmallInteger(), nullable=True),
        sa.Column("floor_model", sa.SmallInteger(), nullable=True),
        sa.Column("total_floors_model", sa.SmallInteger(), nullable=True),
        sa.Column("floor_applicability", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["last_scrape_run_id"], ["scrape_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ad_url"),
    )

    # Indexes for common search/filter patterns
    op.create_index("ix_listings_ad_url", "listings", ["ad_url"], unique=True)
    op.create_index("ix_listings_title_city_model", "listings", ["title_city_model"])
    op.create_index("ix_listings_deal_type_normalized", "listings", ["deal_type_normalized"])
    op.create_index("ix_listings_property_type_raw", "listings", ["property_type_raw"])
    op.create_index("ix_listings_geo_category", "listings", ["geo_category"])
    op.create_index("ix_listings_total_price", "listings", ["total_price"])
    op.create_index("ix_listings_price_per_sqm_model", "listings", ["price_per_sqm_model"])
    op.create_index("ix_listings_area_sqm_model", "listings", ["area_sqm_model"])
    op.create_index("ix_listings_last_seen_at", "listings", ["last_seen_at"])
    op.create_index("ix_listings_published_date", "listings", ["published_date"])
    op.create_index("ix_listings_training_eligible", "listings", ["training_eligible"])

    op.create_table(
        "listing_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("listing_id", sa.BigInteger(), nullable=False),
        sa.Column("scrape_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scraped_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("total_price", sa.Numeric(14, 2), nullable=True),
        sa.Column("currency", sa.String(5), nullable=True),
        sa.Column("price_per_sqm_model", sa.Numeric(10, 2), nullable=True),
        sa.Column("area_sqm_model", sa.Numeric(10, 2), nullable=True),
        sa.Column("vat_status", sa.Text(), nullable=True),
        sa.Column("views", sa.Integer(), nullable=True),
        sa.Column("days_on_market", sa.Integer(), nullable=True),
        sa.Column("parsed_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["listing_id"], ["listings.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scrape_run_id"], ["scrape_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_snapshots_listing_id", "listing_snapshots", ["listing_id"])
    op.create_index("ix_snapshots_scraped_at", "listing_snapshots", ["scraped_at"])

    op.create_table(
        "appraisal_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), server_default="draft", nullable=True),
        sa.Column("subject_address", sa.Text(), nullable=True),
        sa.Column("subject_city", sa.Text(), nullable=True),
        sa.Column("subject_area_sqm", sa.Numeric(10, 2), nullable=True),
        sa.Column("subject_floor", sa.SmallInteger(), nullable=True),
        sa.Column("subject_total_floors", sa.SmallInteger(), nullable=True),
        sa.Column("subject_construction", sa.Text(), nullable=True),
        sa.Column("subject_year", sa.SmallInteger(), nullable=True),
        sa.Column("subject_description", sa.Text(), nullable=True),
        sa.Column("concluded_value_sales", sa.Numeric(14, 2), nullable=True),
        sa.Column("annual_rent_estimate", sa.Numeric(14, 2), nullable=True),
        sa.Column("gross_rent_multiplier", sa.Numeric(6, 3), nullable=True),
        sa.Column("capitalization_rate", sa.Numeric(6, 4), nullable=True),
        sa.Column("concluded_value_income", sa.Numeric(14, 2), nullable=True),
        sa.Column("concluded_value", sa.Numeric(14, 2), nullable=True),
        sa.Column("concluded_currency", sa.String(5), server_default="EUR", nullable=True),
        sa.Column("valuation_date", sa.Date(), nullable=True),
        sa.Column("appraiser_notes", sa.Text(), nullable=True),
        sa.Column("docx_path", sa.Text(), nullable=True),
        sa.Column("pdf_path", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "report_comparables",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("report_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("listing_id", sa.BigInteger(), nullable=False),
        sa.Column("comparable_type", sa.String(10), nullable=False),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("adjustment_pct", sa.Numeric(6, 2), nullable=True),
        sa.Column("analyst_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["listing_id"], ["listings.id"]),
        sa.ForeignKeyConstraint(["report_id"], ["appraisal_reports.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("report_id", "comparable_type", "listing_id"),
        sa.UniqueConstraint("report_id", "comparable_type", "position"),
    )


def downgrade() -> None:
    op.drop_table("report_comparables")
    op.drop_table("appraisal_reports")
    op.drop_table("listing_snapshots")
    op.drop_table("listings")
    op.drop_table("scrape_runs")
