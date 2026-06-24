from __future__ import annotations

import io
import uuid
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import AppraisalReport, ComparablePool

MAX_PINNED = 6   # max pinned-for-report per type (Word table limit)

# ── Word export constants ─────────────────────────────────────────────────────

_KNOB_TEXT = (
    "КСБ Аналитика е вписана със сертификат № 900200284 в Регистъра на Камарата на "
    "независимите оценители в България (КНОБ) и извършва оценителска дейност на "
    "територията на Република България за недвижими имоти, "
    "както и за финансови активи и финансови институции."
)
_BRAND_DARK = RGBColor(0x1E, 0x3A, 0x5F)
_BRAND_BLUE = RGBColor(0x25, 0x63, 0xEB)
_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
_MUTED      = RGBColor(0x64, 0x74, 0x8B)

# Column widths (cm) for the 11-column comparables table — total = 16.5 cm (A4 portrait)
_COMP_COL_WIDTHS = [0.7, 3.8, 1.2, 1.6, 1.6, 1.4, 0.9, 1.0, 1.0, 1.5, 1.8]

# ── Word helpers ──────────────────────────────────────────────────────────────

def _set_cell_bg(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for existing in tcPr.findall(qn("w:shd")):
        tcPr.remove(existing)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color.upper())
    tcPr.append(shd)


def _fill_cell(
    cell,
    text: str,
    *,
    bold: bool = False,
    pt: float = 9,
    color: RGBColor | None = None,
    align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT,
) -> None:
    para = cell.paragraphs[0]
    para.alignment = align
    run = para.add_run(str(text) if text is not None else "")
    run.bold = bold
    run.font.size = Pt(pt)
    if color:
        run.font.color.rgb = color


def _add_field(para, field: str, pt: float = 8) -> None:
    """Append a Word field code (PAGE / NUMPAGES) as a run in para."""
    run = para.add_run()
    run.font.size = Pt(pt)
    for kind, instr in [
        ("begin", None),
        ("instrText", field),
        ("separate", None),
        ("end", None),
    ]:
        if kind == "instrText":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = f" {instr} "
            run._r.append(el)
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            run._r.append(el)


def _para_border(para, position: str = "bottom", color: str = "1E3A5F", sz: int = 6) -> None:
    pPr = para._p.get_or_add_pPr()
    pBdr = pPr.find(qn("w:pBdr"))
    if pBdr is None:
        pBdr = OxmlElement("w:pBdr")
        pPr.append(pBdr)
    border = OxmlElement(f"w:{position}")
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), str(sz))
    border.set(qn("w:space"), "1")
    border.set(qn("w:color"), color)
    pBdr.append(border)


def _set_table_fixed(tbl, widths_cm: list[float]) -> None:
    """Force fixed layout and set column widths via tblGrid."""
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    # Total width
    total = sum(int(Cm(w) / 635) for w in widths_cm)
    for tag in [qn("w:tblW")]:
        for el in tblPr.findall(tag):
            tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(total))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    # tblGrid
    existing_grid = tbl._tbl.find(qn("w:tblGrid"))
    if existing_grid is not None:
        tbl._tbl.remove(existing_grid)
    tblGrid = OxmlElement("w:tblGrid")
    for w in widths_cm:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(int(Cm(w) / 635)))
        tblGrid.append(gc)
    # Insert tblGrid right after tblPr
    tblPr_idx = list(tbl._tbl).index(tblPr)
    tbl._tbl.insert(tblPr_idx + 1, tblGrid)


def _fmt(v, decimals: int = 0) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
        if decimals == 0:
            return f"{int(round(n)):,}".replace(",", " ")
        return f"{n:,.{decimals}f}".replace(",", " ")
    except Exception:
        return str(v)


def _floor_str_docx(item: dict) -> str:
    f, t = item.get("floor_model"), item.get("total_floors_model")
    if f is None and t is None:
        return "—"
    return f"{f if f is not None else '?'}/{t}" if t else (str(f) if f else "—")


def _write_comp_table(
    doc: Document,
    pinned: list[dict],
    ctype: str,
    report: AppraisalReport,
    stats: dict | None,
) -> None:
    ppsqm_lbl = "EUR/кв.м" if ctype == "sale" else "EUR/кв.м/мес"
    price_lbl = "Цена (EUR)" if ctype == "sale" else "Наем/мес (EUR)"

    headers = [
        "№", "Местоположение", "Площ\nкв.м", price_lbl,
        f"Цена/кв.м\n{ppsqm_lbl}", "Строит.\nтип", "Год.", "Ет./\nОбш.",
        "Кор.\n%", f"Кор.\n{ppsqm_lbl}", "Бележки",
    ]

    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    _set_table_fixed(tbl, _COMP_COL_WIDTHS)

    # ── Header row ────────────────────────────────────────────────
    hdr = tbl.rows[0]
    for cell, hdr_text in zip(hdr.cells, headers):
        _fill_cell(cell, hdr_text, bold=True, pt=8, color=_WHITE, align=WD_ALIGN_PARAGRAPH.CENTER)
        _set_cell_bg(cell, "1E3A5F")

    # ── Subject row (yellow) ──────────────────────────────────────
    if report.subject_address or report.subject_city:
        srow = tbl.add_row()
        subj_vals = [
            "Обект",
            f"{report.subject_address or ''} {report.subject_city or ''}".strip(),
            _fmt(report.subject_area_sqm), "", "",
            report.subject_construction or "—",
            str(report.subject_year) if report.subject_year else "—",
            f"{report.subject_floor or '?'}/{report.subject_total_floors or '?'}",
            "", "", "",
        ]
        for cell, val in zip(srow.cells, subj_vals):
            _fill_cell(cell, val, bold=(val == "Обект"), pt=8.5)
            _set_cell_bg(cell, "FEF3C7")

    # ── Data rows ─────────────────────────────────────────────────
    for pos, item in enumerate(pinned, start=1):
        adj = float(item.get("adjustment_pct") or 0)
        row = tbl.add_row()
        location = (
            f"{item.get('title_city_model') or ''} "
            f"{item.get('title_geo_2_model') or ''} "
            f"{item.get('location_raw') or ''}"
        ).strip()
        vals = [
            str(pos), location,
            _fmt(item.get("area_sqm_model")),
            _fmt(item.get("total_price")),
            _fmt(item.get("price_per_sqm_model")),
            item.get("construction_type_model") or "—",
            str(item["construction_year_model"]) if item.get("construction_year_model") else "—",
            _floor_str_docx(item),
            f"{adj:+.1f}%" if adj != 0 else "—",
            _fmt(item.get("adj_ppsqm")),
            item.get("analyst_note") or "",
        ]
        for cell, val in zip(row.cells, vals):
            _fill_cell(cell, val, pt=8.5)
        if item.get("pinned_for_report"):
            for cell in row.cells:
                _set_cell_bg(cell, "DCFCE7")

    # ── Stats row (light blue) ────────────────────────────────────
    if stats:
        srow = tbl.add_row()
        stats_vals = [
            "Статистики",
            f"N={stats['n']}",
            f"{_fmt(stats['min_area'])}–{_fmt(stats['max_area'])} кв.м",
            f"Ср. {_fmt(stats['mean_price'])} EUR",
            f"Мин {_fmt(stats['min_ppsqm'])} | Ср {_fmt(stats['mean_ppsqm'])} | Макс {_fmt(stats['max_ppsqm'])}",
            f"Медиана: {_fmt(stats['median'])}",
            "", "",
            f"Q25: {_fmt(stats['p25'])}",
            f"Q75: {_fmt(stats['p75'])}",
            "",
        ]
        for cell, val in zip(srow.cells, stats_vals):
            _fill_cell(cell, val, bold=(val.startswith("Статистики")), pt=8)
            _set_cell_bg(cell, "EFF6FF")


# ── Word / DOCX report generation ────────────────────────────────────────────

def generate_docx(db: Session, report: AppraisalReport) -> io.BytesIO:
    pool_sale = get_pool_with_stats(db, "sale", report.id)
    pool_rent = get_pool_with_stats(db, "rent", report.id)
    pinned_sale = [r for r in pool_sale["rows"] if r["pinned_for_report"]]
    pinned_rent = [r for r in pool_rent["rows"] if r["pinned_for_report"]]

    doc = Document()

    # ── Page setup (A4 portrait) ──────────────────────────────────
    sec = doc.sections[0]
    sec.page_width   = Cm(21)
    sec.page_height  = Cm(29.7)
    sec.left_margin  = Cm(2.5)
    sec.right_margin = Cm(2.0)
    sec.top_margin   = Cm(2.5)
    sec.bottom_margin = Cm(2.5)

    # ── Header ────────────────────────────────────────────────────
    sec.header.is_linked_to_previous = False   # register headerReference
    hp = sec.header.paragraphs[0]              # re-access for live reference
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT

    logo_path = Path("static/logo.png")
    if logo_path.exists():
        hp.add_run().add_picture(str(logo_path), height=Cm(1.1))
        hp.add_run("  ")

    r_co = hp.add_run("КСБ Аналитика")
    r_co.bold = True
    r_co.font.size = Pt(11)
    r_co.font.color.rgb = _BRAND_DARK

    r_sep = hp.add_run("  |  ДОКЛАД ЗА ОЦЕНКА НА НЕДВИЖИМ ИМОТ")
    r_sep.font.size = Pt(9)
    r_sep.font.color.rgb = _MUTED
    _para_border(hp, "bottom", "1E3A5F", 6)

    # ── Footer ────────────────────────────────────────────────────
    sec.footer.is_linked_to_previous = False   # register footerReference
    fp = sec.footer.paragraphs[0]              # re-access for live reference
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _para_border(fp, "top", "1E3A5F", 4)

    r_knob = fp.add_run(_KNOB_TEXT + "   |   стр. ")
    r_knob.font.size = Pt(7.5)
    r_knob.font.color.rgb = _MUTED
    _add_field(fp, "PAGE", pt=7.5)
    r_of = fp.add_run(" / ")
    r_of.font.size = Pt(7.5)
    _add_field(fp, "NUMPAGES", pt=7.5)

    # ── Cover page ────────────────────────────────────────────────
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    r = p.add_run("КСБ Аналитика")
    r.bold = True
    r.font.size = Pt(24)
    r.font.color.rgb = _BRAND_DARK

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("ДОКЛАД ЗА ОЦЕНКА НА НЕДВИЖИМ ИМОТ")
    r2.bold = True
    r2.font.size = Pt(14)
    r2.font.color.rgb = _BRAND_BLUE

    doc.add_paragraph()

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run(report.title or "Нов доклад")
    r3.bold = True
    r3.font.size = Pt(16)
    r3.font.color.rgb = _BRAND_DARK

    doc.add_paragraph()

    if report.subject_address or report.subject_city:
        pa = doc.add_paragraph()
        pa.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ra = pa.add_run(
            f"{report.subject_address or ''}, {report.subject_city or ''}".strip(", ")
        )
        ra.font.size = Pt(12)

    if report.valuation_date:
        pv = doc.add_paragraph()
        pv.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rv = pv.add_run(f"Дата на оценка: {report.valuation_date.strftime('%d.%m.%Y')}")
        rv.font.size = Pt(11)
        rv.font.color.rgb = _MUTED

    doc.add_page_break()

    # ── Section 1: Subject property ───────────────────────────────
    h1 = doc.add_heading("1. Описание на оценявания имот", level=1)
    if h1.runs:
        h1.runs[0].font.color.rgb = _BRAND_DARK

    tbl_subj = doc.add_table(rows=1, cols=2)
    tbl_subj.style = "Table Grid"
    tbl_subj.autofit = False
    tbl_subj._tbl.remove(tbl_subj.rows[0]._tr)

    def _subj_row(label, value):
        row = tbl_subj.add_row()
        _fill_cell(row.cells[0], label, bold=True, pt=10)
        _fill_cell(row.cells[1], value if value else "—", pt=10)

    _subj_row("Адрес", report.subject_address)
    _subj_row("Град", report.subject_city)
    _subj_row("Площ (кв.м)", _fmt(report.subject_area_sqm))
    _subj_row(
        "Етаж / Общо етажи",
        f"{report.subject_floor or '—'} / {report.subject_total_floors or '—'}",
    )
    _subj_row("Тип строителство", report.subject_construction)
    _subj_row("Година на строеж", str(report.subject_year) if report.subject_year else None)
    _subj_row(
        "Дата на оценка",
        report.valuation_date.strftime("%d.%m.%Y") if report.valuation_date else None,
    )

    if report.subject_description:
        doc.add_paragraph()
        pd = doc.add_paragraph()
        r_lbl = pd.add_run("Описание: ")
        r_lbl.bold = True
        r_lbl.font.size = Pt(10)
        r_desc = pd.add_run(report.subject_description)
        r_desc.font.size = Pt(10)

    doc.add_paragraph()

    # ── Section 2: Sales comparables ──────────────────────────────
    h2 = doc.add_heading("2. Пазарен подход — продажни сравними", level=1)
    if h2.runs:
        h2.runs[0].font.color.rgb = _BRAND_DARK

    if pinned_sale:
        _write_comp_table(doc, pinned_sale, "sale", report, pool_sale["stats"])
    else:
        p_no = doc.add_paragraph("Няма закачени (\U0001F4CC) продажни сравними за доклада.")
        p_no.runs[0].italic = True
        p_no.runs[0].font.size = Pt(10)

    # Concluded sales value
    if report.concluded_value_sales:
        doc.add_paragraph()
        p_cs = doc.add_paragraph()
        r_cs = p_cs.add_run(
            f"Стойност по пазарен подход: "
            f"{_fmt(report.concluded_value_sales)} {report.concluded_currency or 'EUR'}"
        )
        r_cs.bold = True
        r_cs.font.size = Pt(11)
        r_cs.font.color.rgb = _BRAND_DARK

    # ── Section 3: Rent comparables (if any) ──────────────────────
    if pinned_rent or pool_rent["total_count"] > 0:
        doc.add_paragraph()
        h3 = doc.add_heading("3. Доходен подход — наемни сравними", level=1)
        if h3.runs:
            h3.runs[0].font.color.rgb = _BRAND_DARK

        if pinned_rent:
            _write_comp_table(doc, pinned_rent, "rent", report, pool_rent["stats"])
        else:
            p_no2 = doc.add_paragraph("Няма закачени (\U0001F4CC) наемни сравними за доклада.")
            p_no2.runs[0].italic = True
            p_no2.runs[0].font.size = Pt(10)

        if report.concluded_value_income:
            doc.add_paragraph()
            p_ci = doc.add_paragraph()
            r_ci = p_ci.add_run(
                f"Стойност по доходен подход: "
                f"{_fmt(report.concluded_value_income)} {report.concluded_currency or 'EUR'}"
            )
            r_ci.bold = True
            r_ci.font.size = Pt(11)
            r_ci.font.color.rgb = _BRAND_DARK

    # ── Section 4 (or 3): Concluded value ────────────────────────
    conc_num = 4 if (pinned_rent or pool_rent["total_count"] > 0) else 3
    doc.add_paragraph()
    h_conc = doc.add_heading(f"{conc_num}. Заключение — оценена стойност", level=1)
    if h_conc.runs:
        h_conc.runs[0].font.color.rgb = _BRAND_DARK

    tbl_conc = doc.add_table(rows=1, cols=2)
    tbl_conc.style = "Table Grid"
    tbl_conc._tbl.remove(tbl_conc.rows[0]._tr)

    if report.concluded_value_sales:
        r = tbl_conc.add_row()
        _fill_cell(r.cells[0], "Пазарна стойност (пазарен подход)", bold=True, pt=10)
        _fill_cell(
            r.cells[1],
            f"{_fmt(report.concluded_value_sales)} {report.concluded_currency or 'EUR'}",
            pt=10,
        )

    if report.concluded_value_income:
        r = tbl_conc.add_row()
        _fill_cell(r.cells[0], "Пазарна стойност (доходен подход)", bold=True, pt=10)
        _fill_cell(
            r.cells[1],
            f"{_fmt(report.concluded_value_income)} {report.concluded_currency or 'EUR'}",
            pt=10,
        )

    if report.concluded_value_residual:
        r = tbl_conc.add_row()
        _fill_cell(r.cells[0], "Стойност на парцела (остатъчен метод)", bold=True, pt=10)
        _fill_cell(
            r.cells[1],
            f"{_fmt(report.concluded_value_residual)} {report.concluded_currency or 'EUR'}",
            pt=10,
        )

    if report.concluded_value:
        r = tbl_conc.add_row()
        _fill_cell(
            r.cells[0], "КРАЙНА ОЦЕНЕНА СТОЙНОСТ",
            bold=True, pt=11, color=_WHITE, align=WD_ALIGN_PARAGRAPH.CENTER,
        )
        _fill_cell(
            r.cells[1],
            f"{_fmt(report.concluded_value)} {report.concluded_currency or 'EUR'}",
            bold=True, pt=11, color=_WHITE,
        )
        _set_cell_bg(r.cells[0], "1E3A5F")
        _set_cell_bg(r.cells[1], "1E3A5F")
    elif not (report.concluded_value_sales or report.concluded_value_income):
        p_nv = doc.add_paragraph("Крайната оценена стойност не е въведена.")
        p_nv.runs[0].italic = True
        p_nv.runs[0].font.size = Pt(10)

    if report.appraiser_notes:
        doc.add_paragraph()
        p_an = doc.add_paragraph()
        r_an_lbl = p_an.add_run("Бележки на оценителя: ")
        r_an_lbl.bold = True
        r_an_lbl.font.size = Pt(10)
        r_an_txt = p_an.add_run(report.appraiser_notes)
        r_an_txt.font.size = Pt(10)

    # ── КНОБ statement in document body ──────────────────────────
    doc.add_paragraph()
    p_knob = doc.add_paragraph(_KNOB_TEXT)
    p_knob.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_knob.runs[0].font.size = Pt(8.5)
    p_knob.runs[0].font.color.rgb = _MUTED

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


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


def update_income_approach(
    db: Session,
    report_id: uuid.UUID,
    rent_per_sqm_month: float | None,
    cap_rate_pct: float | None,
    concluded_per_sqm: float | None,
    subject_area_sqm: float | None,
) -> None:
    report = db.get(AppraisalReport, report_id)
    if not report:
        return
    if rent_per_sqm_month is not None:
        report.annual_rent_estimate = round(rent_per_sqm_month * 12, 2)
    if cap_rate_pct is not None:
        report.capitalization_rate = round(cap_rate_pct / 100, 6)
    if concluded_per_sqm is not None:
        area = subject_area_sqm or float(report.subject_area_sqm or 0)
        report.concluded_value_income = round(concluded_per_sqm * area, 2) if area > 0 else round(concluded_per_sqm, 2)
    db.commit()


def update_residual_approach(
    db: Session,
    report_id: uuid.UUID,
    concluded_value_residual: float | None,
) -> None:
    report = db.get(AppraisalReport, report_id)
    if not report:
        return
    if concluded_value_residual is not None:
        report.concluded_value_residual = round(concluded_value_residual, 2)
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
