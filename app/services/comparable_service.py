from __future__ import annotations

import io
import uuid

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, ComparablePool

MAX_PINNED = 6   # max pinned-for-report per type (Word table limit)


# ── Draft / report management ─────────────────────────────────────────────────

def get_or_create_draft(db: Session, user_id: int) -> AppraisalReport:
    draft = (
        db.query(AppraisalReport)
        .filter(AppraisalReport.status == "draft", AppraisalReport.user_id == user_id)
        .order_by(AppraisalReport.updated_at.desc())
        .first()
    )
    if draft is None:
        draft = _new_draft_obj(db, user_id)
    return draft


def new_draft(db: Session, user_id: int) -> AppraisalReport:
    return _new_draft_obj(db, user_id)


def _new_draft_obj(db: Session, user_id: int) -> AppraisalReport:
    draft = AppraisalReport(title="Нов доклад", status="draft", user_id=user_id)
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


def get_user_reports(db: Session, user_id: int) -> list[dict]:
    rows = db.execute(text("""
        SELECT
            ar.id,
            ar.title,
            ar.status,
            ar.created_at,
            ar.updated_at,
            ar.subject_address,
            ar.subject_city,
            ar.valuation_date,
            count(cp.id) FILTER (WHERE cp.comparable_type = 'sale') AS sale_count,
            count(cp.id) FILTER (WHERE cp.comparable_type = 'rent') AS rent_count,
            count(cp.id) FILTER (WHERE cp.pinned_for_report = true)  AS pinned_count
        FROM appraisal_reports ar
        LEFT JOIN comparable_pool cp ON cp.report_id = ar.id
        WHERE ar.user_id = :uid
        GROUP BY ar.id
        ORDER BY ar.updated_at DESC
    """), {"uid": user_id}).mappings().all()
    return [dict(r) for r in rows]


def get_report_for_user(
    db: Session, report_id: uuid.UUID, user_id: int
) -> AppraisalReport | None:
    return (
        db.query(AppraisalReport)
        .filter(AppraisalReport.id == report_id, AppraisalReport.user_id == user_id)
        .first()
    )


def update_subject(db: Session, report_id: uuid.UUID, data: dict) -> None:
    report = db.get(AppraisalReport, report_id)
    if not report:
        return
    fields = (
        "title", "subject_address", "subject_city", "subject_area_sqm",
        "subject_floor", "subject_total_floors", "subject_construction",
        "subject_year", "subject_description", "valuation_date",
    )
    for f in fields:
        if f in data:
            setattr(report, f, data[f] if data[f] != "" else None)
    db.commit()


def delete_user_report(db: Session, report_id: uuid.UUID, user_id: int) -> None:
    report = (
        db.query(AppraisalReport)
        .filter(AppraisalReport.id == report_id, AppraisalReport.user_id == user_id)
        .first()
    )
    if report:
        db.delete(report)
        db.commit()


def finalize_user_report(db: Session, report_id: uuid.UUID, user_id: int) -> None:
    report = (
        db.query(AppraisalReport)
        .filter(AppraisalReport.id == report_id, AppraisalReport.user_id == user_id)
        .first()
    )
    if report and report.status == "draft":
        report.status = "finalized"
        db.commit()


# ── Pool operations ───────────────────────────────────────────────────────────

def add_to_pool(
    db: Session,
    listing_ids: list[int],
    comparable_type: str,
    report_id: uuid.UUID,
    user_id: int,
) -> int:
    if not listing_ids:
        return 0
    stmt = (
        pg_insert(ComparablePool)
        .values([{
            "listing_id": lid,
            "comparable_type": comparable_type,
            "report_id": report_id,
            "user_id": user_id,
        } for lid in listing_ids])
        .on_conflict_do_nothing(constraint="uq_pool_listing_ctype_report")
    )
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


def remove_from_pool(db: Session, pool_id: int) -> None:
    item = db.get(ComparablePool, pool_id)
    if item:
        db.delete(item)
        db.commit()


def clear_pool(
    db: Session, report_id: uuid.UUID, comparable_type: str | None = None
) -> None:
    q = db.query(ComparablePool).filter(ComparablePool.report_id == report_id)
    if comparable_type:
        q = q.filter_by(comparable_type=comparable_type)
    q.delete()
    db.commit()


def toggle_pin(db: Session, pool_id: int) -> bool:
    """Toggle pinned_for_report. Returns new value. Enforces MAX_PINNED."""
    item = db.get(ComparablePool, pool_id)
    if not item:
        return False
    if not item.pinned_for_report:
        pinned_count = (
            db.query(ComparablePool)
            .filter_by(
                comparable_type=item.comparable_type,
                report_id=item.report_id,
                pinned_for_report=True,
            )
            .count()
        )
        if pinned_count >= MAX_PINNED:
            return False
    item.pinned_for_report = not item.pinned_for_report
    db.commit()
    return item.pinned_for_report


def update_pool_adjustment(
    db: Session, pool_id: int, adjustment_pct: float | None, analyst_note: str
) -> None:
    item = db.get(ComparablePool, pool_id)
    if item:
        item.adjustment_pct = adjustment_pct
        item.analyst_note = analyst_note or None
        db.commit()


# ── Pool query + statistics ───────────────────────────────────────────────────

def get_pool_with_stats(
    db: Session, comparable_type: str, report_id: uuid.UUID
) -> dict:
    """Returns {rows, stats, pinned_count, total_count} for the given report + type."""
    rows = db.execute(text("""
        SELECT
            cp.id AS pool_id,
            cp.pinned_for_report,
            cp.adjustment_pct,
            cp.analyst_note,
            cp.added_at,
            l.id AS listing_id,
            l.ad_url,
            l.title,
            l.title_city_model,
            l.title_geo_2_model,
            l.location_raw,
            l.total_price,
            l.currency,
            l.price_per_sqm_model,
            l.area_sqm_model,
            l.floor_model,
            l.total_floors_model,
            l.construction_type_model,
            l.construction_year_model,
            l.published_date,
            l.last_seen_at,
            l.deal_type_normalized,
            l.property_type_raw
        FROM comparable_pool cp
        JOIN listings l ON l.id = cp.listing_id
        WHERE cp.comparable_type = :ctype
          AND cp.report_id = :rid
          AND l.status = 'active'
        ORDER BY cp.pinned_for_report DESC, cp.added_at DESC
    """), {"ctype": comparable_type, "rid": str(report_id)}).mappings().all()

    items = []
    for r in rows:
        r = dict(r)
        adj = float(r["adjustment_pct"]) if r["adjustment_pct"] is not None else 0.0
        ppsqm = float(r["price_per_sqm_model"]) if r["price_per_sqm_model"] else None
        r["adj_ppsqm"] = round(ppsqm * (1 + adj / 100), 0) if ppsqm is not None else None
        r["adjustment_pct"] = adj
        items.append(r)

    stats = _compute_stats(db, comparable_type, report_id)
    pinned_count = sum(1 for i in items if i["pinned_for_report"])
    return {
        "rows": items,
        "stats": stats,
        "pinned_count": pinned_count,
        "total_count": len(items),
    }


def _compute_stats(
    db: Session, comparable_type: str, report_id: uuid.UUID
) -> dict | None:
    row = db.execute(text("""
        SELECT
            count(*) AS n,
            round(min(l.price_per_sqm_model)::numeric, 0)  AS min_ppsqm,
            round(max(l.price_per_sqm_model)::numeric, 0)  AS max_ppsqm,
            round(avg(l.price_per_sqm_model)::numeric, 0)  AS mean_ppsqm,
            round(percentile_cont(0.25) WITHIN GROUP (ORDER BY l.price_per_sqm_model)::numeric, 0) AS p25,
            round(percentile_cont(0.50) WITHIN GROUP (ORDER BY l.price_per_sqm_model)::numeric, 0) AS median,
            round(percentile_cont(0.75) WITHIN GROUP (ORDER BY l.price_per_sqm_model)::numeric, 0) AS p75,
            round(min(l.area_sqm_model)::numeric, 0)       AS min_area,
            round(max(l.area_sqm_model)::numeric, 0)       AS max_area,
            round(avg(l.area_sqm_model)::numeric, 0)       AS mean_area,
            round(min(l.total_price)::numeric, 0)          AS min_price,
            round(max(l.total_price)::numeric, 0)          AS max_price,
            round(avg(l.total_price)::numeric, 0)          AS mean_price
        FROM comparable_pool cp
        JOIN listings l ON l.id = cp.listing_id
        WHERE cp.comparable_type = :ctype
          AND cp.report_id = :rid
          AND l.status = 'active'
          AND l.price_per_sqm_model IS NOT NULL
    """), {"ctype": comparable_type, "rid": str(report_id)}).mappings().first()

    if not row or not row["n"]:
        return None
    return dict(row)


# ── Excel export ──────────────────────────────────────────────────────────────

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="2563EB")
_PINNED_FILL = PatternFill("solid", fgColor="DCFCE7")
_SUBJECT_FILL = PatternFill("solid", fgColor="FEF3C7")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _autowidth(ws) -> None:
    for col in ws.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 45)


def _write_header(ws, cols: list[str]) -> None:
    ws.append(cols)
    for cell in ws[1]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER


def _floor_str(item: dict) -> str:
    f, t = item.get("floor_model"), item.get("total_floors_model")
    if f is None and t is None:
        return "—"
    return f"{f if f is not None else '?'}/{t}" if t else (str(f) if f is not None else "—")


def export_excel(db: Session, report: AppraisalReport) -> io.BytesIO:
    pool_sale = get_pool_with_stats(db, "sale", report.id)
    pool_rent = get_pool_with_stats(db, "rent", report.id)

    wb = Workbook()
    wb.remove(wb.active)

    def _subject_row(ws, ncols: int) -> None:
        if not (report and report.subject_address):
            return
        pad = [""] * ncols
        row_data = [
            "Обект",
            f"{report.subject_address or ''}\n{report.subject_city or ''}".strip(),
            float(report.subject_area_sqm) if report.subject_area_sqm else "",
            "", "",
            report.subject_construction or "",
            report.subject_year or "",
            f"{report.subject_floor or '?'}/{report.subject_total_floors or '?'}",
        ] + pad[8:]
        ws.append(row_data[:ncols])
        for cell in ws[ws.max_row]:
            cell.fill = _SUBJECT_FILL

    def _stats_row(ws, stats: dict | None, ncols: int) -> None:
        if not stats:
            return
        row = [
            "СТАТИСТИКИ", f"N={stats['n']}",
            f"Площ: {stats['min_area']}–{stats['max_area']} (ср.{stats['mean_area']})",
            "", "",
            f"Цена/кв.м: мин {stats['min_ppsqm']} | ср {stats['mean_ppsqm']} | макс {stats['max_ppsqm']}",
            f"Q25={stats['p25']} | Медиана={stats['median']} | Q75={stats['p75']}",
        ]
        ws.append((row + [""] * ncols)[:ncols])
        fill = PatternFill("solid", fgColor="EFF6FF")
        for cell in ws[ws.max_row]:
            cell.fill = fill

    def _write_sheet(ws, items, stats, cols, include_pin_col=True):
        all_cols = (["📌"] if include_pin_col else []) + cols
        _write_header(ws, all_cols)
        _subject_row(ws, len(all_cols))
        _stats_row(ws, stats, len(all_cols))
        for pos, item in enumerate(items, start=1):
            row = ([("✔" if item["pinned_for_report"] else "")] if include_pin_col else []) + [
                pos,
                f"{item.get('title_city_model') or ''} {item.get('title_geo_2_model') or ''}\n"
                f"{item.get('location_raw') or ''}".strip(),
                float(item["area_sqm_model"]) if item.get("area_sqm_model") else "",
                float(item["total_price"]) if item.get("total_price") else "",
                float(item["price_per_sqm_model"]) if item.get("price_per_sqm_model") else "",
                item.get("construction_type_model") or "",
                item.get("construction_year_model") or "",
                _floor_str(item),
                item["adjustment_pct"],
                item["adj_ppsqm"],
                item.get("analyst_note") or "",
                item["ad_url"],
            ]
            ws.append(row[:len(all_cols)])
            if item["pinned_for_report"]:
                for cell in ws[ws.max_row]:
                    cell.fill = _PINNED_FILL
        _autowidth(ws)

    sale_cols = ["№", "Адрес / Описание", "Площ (кв.м)", "Цена (EUR)",
                 "Цена/кв.м (EUR)", "Строит. тип", "Год.", "Ет./Обш.",
                 "Корекция (%)", "Кор. цена/кв.м", "Бележки", "URL"]
    rent_cols = ["№", "Адрес / Описание", "Площ (кв.м)", "Наем/мес (EUR)",
                 "Наем/кв.м/мес (EUR)", "Строит. тип", "Год.", "Ет./Обш.",
                 "Корекция (%)", "Кор. наем/кв.м", "Бележки", "URL"]

    if pool_sale["rows"]:
        ws = wb.create_sheet("Продажни сравними")
        _write_sheet(ws, pool_sale["rows"], pool_sale["stats"], sale_cols)

    if pool_rent["rows"]:
        ws = wb.create_sheet("Наемни сравними")
        _write_sheet(ws, pool_rent["rows"], pool_rent["stats"], rent_cols)

    if not wb.sheetnames:
        ws = wb.create_sheet("Без данни")
        ws.append(["Няма добавени сравними."])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
