"""Tie appraisal_reports and comparable_pool to users; add report_id to pool

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. appraisal_reports.user_id ──────────────────────────────────────────
    op.add_column("appraisal_reports", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_appraisal_reports_user", "appraisal_reports",
        "users", ["user_id"], ["id"], ondelete="SET NULL",
    )
    # Backfill existing rows to first admin user (if any)
    op.execute("""
        UPDATE appraisal_reports
        SET user_id = (SELECT id FROM users WHERE role = 'admin' ORDER BY id LIMIT 1)
        WHERE user_id IS NULL
    """)
    op.create_index("ix_appraisal_reports_user_id", "appraisal_reports", ["user_id"])

    # ── 2. comparable_pool.user_id ────────────────────────────────────────────
    op.add_column("comparable_pool", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_comparable_pool_user", "comparable_pool",
        "users", ["user_id"], ["id"], ondelete="SET NULL",
    )

    # ── 3. comparable_pool.report_id ──────────────────────────────────────────
    op.add_column("comparable_pool", sa.Column("report_id", PGUUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_comparable_pool_report", "comparable_pool",
        "appraisal_reports", ["report_id"], ["id"], ondelete="CASCADE",
    )

    # ── 4. Backfill pool: link existing entries to admin's draft report ────────
    op.execute("""
        DO $$
        DECLARE
            v_admin_id  INTEGER;
            v_draft_id  UUID;
        BEGIN
            SELECT id INTO v_admin_id
            FROM users WHERE role = 'admin' ORDER BY id LIMIT 1;

            IF v_admin_id IS NOT NULL
               AND EXISTS (SELECT 1 FROM comparable_pool WHERE report_id IS NULL)
            THEN
                -- Reuse an existing draft for admin if available
                SELECT id INTO v_draft_id
                FROM appraisal_reports
                WHERE user_id = v_admin_id AND status = 'draft'
                ORDER BY created_at DESC LIMIT 1;

                -- Otherwise create a migration-placeholder draft
                IF v_draft_id IS NULL THEN
                    v_draft_id := gen_random_uuid();
                    INSERT INTO appraisal_reports
                        (id, title, status, user_id, created_at, updated_at)
                    VALUES
                        (v_draft_id, 'Мигриран пул', 'draft', v_admin_id, now(), now());
                END IF;

                UPDATE comparable_pool
                SET report_id = v_draft_id, user_id = v_admin_id
                WHERE report_id IS NULL;
            END IF;
        END $$;
    """)

    # ── 5. Replace unique constraint to include report_id ─────────────────────
    op.drop_constraint("uq_pool_listing_type", "comparable_pool", type_="unique")
    op.create_unique_constraint(
        "uq_pool_listing_ctype_report", "comparable_pool",
        ["listing_id", "comparable_type", "report_id"],
    )
    op.create_index("ix_pool_report_id", "comparable_pool", ["report_id"])


def downgrade() -> None:
    op.drop_index("ix_pool_report_id", "comparable_pool")
    op.drop_constraint("uq_pool_listing_ctype_report", "comparable_pool", type_="unique")
    op.create_unique_constraint(
        "uq_pool_listing_type", "comparable_pool", ["listing_id", "comparable_type"]
    )
    op.drop_constraint("fk_comparable_pool_report", "comparable_pool", type_="foreignkey")
    op.drop_column("comparable_pool", "report_id")
    op.drop_constraint("fk_comparable_pool_user", "comparable_pool", type_="foreignkey")
    op.drop_column("comparable_pool", "user_id")
    op.drop_index("ix_appraisal_reports_user_id", "appraisal_reports")
    op.drop_constraint("fk_appraisal_reports_user", "appraisal_reports", type_="foreignkey")
    op.drop_column("appraisal_reports", "user_id")
