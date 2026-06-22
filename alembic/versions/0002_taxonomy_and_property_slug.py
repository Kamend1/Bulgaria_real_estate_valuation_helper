"""Add taxonomy tables and property_type_slug

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-22
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "taxonomy_property_types",
        sa.Column("slug", sa.Text(), primary_key=True),
        sa.Column("display_name_bg", sa.Text(), nullable=True),
        sa.Column("route_count", sa.Integer(), nullable=True),
    )

    op.create_table(
        "taxonomy_geo_paths",
        sa.Column("deal_type", sa.Text(), nullable=False),
        sa.Column("geo_path", sa.Text(), nullable=False),
        sa.Column("geo_level_count", sa.SmallInteger(), nullable=True),
        sa.Column("geo_1", sa.Text(), nullable=True),
        sa.Column("geo_2", sa.Text(), nullable=True),
        sa.Column("geo_3", sa.Text(), nullable=True),
        sa.Column("route_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("deal_type", "geo_path"),
    )
    op.create_index("ix_taxonomy_geo_paths_geo_1", "taxonomy_geo_paths", ["geo_1"])
    op.create_index("ix_taxonomy_geo_paths_deal_type", "taxonomy_geo_paths", ["deal_type"])

    op.add_column("listings", sa.Column("property_type_slug", sa.Text(), nullable=True))
    op.create_index("ix_listings_property_type_slug", "listings", ["property_type_slug"])


def downgrade() -> None:
    op.drop_index("ix_listings_property_type_slug", table_name="listings")
    op.drop_column("listings", "property_type_slug")
    op.drop_index("ix_taxonomy_geo_paths_deal_type", table_name="taxonomy_geo_paths")
    op.drop_index("ix_taxonomy_geo_paths_geo_1", table_name="taxonomy_geo_paths")
    op.drop_table("taxonomy_geo_paths")
    op.drop_table("taxonomy_property_types")
