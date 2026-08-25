"""
Integration tests for app/services/comparable_service.py's update_* and
pool-statistics functions, run against the real `appraisal` DB inside a
transaction that's always rolled back (see tests/conftest.py::db_session).
"""
import uuid

import pytest
from docx import Document

from app.db.models import AppraisalReport, ComparablePool, Listing, User
from app.services import comparable_service


def _make_report(db_session, **overrides) -> AppraisalReport:
    report = AppraisalReport(title="Тест доклад", status="draft", **overrides)
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    return report


def _make_listing(db_session, *, ad_url: str, total_price, area_sqm_model, ppsqm=None) -> Listing:
    listing = Listing(
        ad_url=ad_url,
        total_price=total_price,
        area_sqm_model=area_sqm_model,
        price_per_sqm_model=ppsqm if ppsqm is not None else (
            round(float(total_price) / float(area_sqm_model), 2) if total_price and area_sqm_model else None
        ),
        currency="EUR",
        status="active",
    )
    db_session.add(listing)
    db_session.commit()
    db_session.refresh(listing)
    return listing


def test_update_sales_approach_sets_value_and_source(db_session):
    report = _make_report(db_session)
    comparable_service.update_sales_approach(db_session, report.id, 123456.789, source="avm")

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_sales) == pytest.approx(123456.79)
    assert refreshed.concluded_value_sales_source == "avm"


def test_update_sales_approach_noop_when_value_none(db_session):
    report = _make_report(db_session, concluded_value_sales=500.0, concluded_value_sales_source="manual")
    comparable_service.update_sales_approach(db_session, report.id, None)

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_sales) == pytest.approx(500.0)
    assert refreshed.concluded_value_sales_source == "manual"


def test_update_income_valuation_computes_annual_rent_cap_rate_and_concluded_value(db_session):
    report = _make_report(db_session, subject_area_sqm=80.0)
    comparable_service.update_income_valuation(
        db_session, report.id,
        rent_per_sqm_month=10.0, cap_rate_pct=6.5,
        method="direct", source="manual",
    )

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.annual_rent_estimate) == pytest.approx(120.0)          # 10 * 12
    assert float(refreshed.capitalization_rate) == pytest.approx(0.065)           # 6.5 / 100

    # concluded_value_income must be exactly area * the SAME direct_value_per_sqm
    # compute_income_valuation() itself would produce for these inputs -- the
    # whole point of the 2026-08-25 audit fix is that there is only one place
    # (this function, via compute_income_valuation) that ever derives this
    # number, so no other formula (JS or otherwise) can silently disagree.
    expected = comparable_service.compute_income_valuation(
        rent_per_sqm_month=10.0, sale_price_per_sqm=None,
        expenses_pct=20.0, vacancy_pct=8.0, cap_rate_pct=6.5,
        growth_pct=2.0, period_years=5, terminal_cap_rate_pct=7.5,
    )
    assert float(refreshed.concluded_value_income) == pytest.approx(expected["direct_value_per_sqm"] * 80.0, rel=1e-6)
    assert refreshed.income_valuation_source == "manual"
    assert refreshed.income_valuation_detail["method"] == "direct"
    assert refreshed.income_valuation_detail["assumptions_used"]["cap_rate_pct"] == 6.5
    assert refreshed.income_valuation_detail["assumptions_used"]["expenses_pct"] == 20.0   # default, not passed explicitly


def test_update_income_valuation_uses_explicit_area_over_subject_area(db_session):
    report = _make_report(db_session, subject_area_sqm=80.0)
    comparable_service.update_income_valuation(
        db_session, report.id,
        rent_per_sqm_month=10.0, cap_rate_pct=6.5,
        method="dcf", source="ai",
        subject_area_sqm=50.0,
    )
    refreshed = db_session.get(AppraisalReport, report.id)
    expected = comparable_service.compute_income_valuation(
        rent_per_sqm_month=10.0, sale_price_per_sqm=None,
        expenses_pct=20.0, vacancy_pct=8.0, cap_rate_pct=6.5,
        growth_pct=2.0, period_years=5, terminal_cap_rate_pct=7.5,
    )
    assert float(refreshed.concluded_value_income) == pytest.approx(expected["dcf_value_per_sqm"] * 50.0, rel=1e-6)
    assert refreshed.income_valuation_source == "ai"
    assert refreshed.income_valuation_detail["method"] == "dcf"


def test_update_income_valuation_noop_when_rent_or_cap_rate_missing(db_session):
    report = _make_report(db_session, subject_area_sqm=80.0, concluded_value_income=500.0)
    comparable_service.update_income_valuation(
        db_session, report.id,
        rent_per_sqm_month=None, cap_rate_pct=6.5,
        method="direct", source="manual",
    )
    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_income) == pytest.approx(500.0)   # untouched -- nothing to compute without rent


def test_update_residual_approach_sets_value(db_session):
    report = _make_report(db_session)
    comparable_service.update_residual_approach(db_session, report.id, 75000.0)

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_residual) == pytest.approx(75000.0)


def test_update_subject_ignored_for_unknown_report_id(db_session):
    # Should silently no-op, not raise, for a report_id that doesn't exist.
    comparable_service.update_sales_approach(db_session, uuid.uuid4(), 1000.0)


# ── compute_weighted_conclusion (pure function) ────────────────────────────────

@pytest.mark.parametrize("sales, income, residual, w_sales, w_income, w_residual, expected", [
    (100000, 90000, None, 60, 40, None, 96000.0),          # normalized weighted avg, two approaches
    (100000, None, None, 100, None, None, 100000.0),       # single approach, full weight
    (100000, 90000, 80000, 50, 30, 20, 93000.0),            # all three: (100k*50+90k*30+80k*20)/100
    (100000, 90000, None, 30, None, None, 100000.0),        # income has no weight -> excluded entirely
    (None, None, None, 100, 100, 100, None),                 # no values at all -> None
    (100000, 90000, None, 0, 40, None, 90000.0),             # zero weight excludes that approach
])
def test_compute_weighted_conclusion(sales, income, residual, w_sales, w_income, w_residual, expected):
    result = comparable_service.compute_weighted_conclusion(
        sales, income, residual, w_sales, w_income, w_residual,
    )
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_update_conclusion_computes_and_persists_weighted_value(db_session):
    report = _make_report(db_session, concluded_value_sales=100000, concluded_value_income=90000)
    comparable_service.update_conclusion(
        db_session, report.id,
        weight_sales_pct=60, weight_income_pct=40, weight_residual_pct=None,
        weighting_rationale="Пазарният подход има по-силна пряка пазарна опора.",
    )
    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value) == pytest.approx(96000.0)
    assert float(refreshed.weight_sales_pct) == pytest.approx(60)
    assert float(refreshed.weight_income_pct) == pytest.approx(40)
    assert refreshed.weight_residual_pct is None
    assert refreshed.weighting_rationale == "Пазарният подход има по-силна пряка пазарна опора."


def test_update_conclusion_with_no_weights_leaves_concluded_value_none(db_session):
    report = _make_report(db_session, concluded_value_sales=100000)
    comparable_service.update_conclusion(
        db_session, report.id,
        weight_sales_pct=None, weight_income_pct=None, weight_residual_pct=None,
        weighting_rationale=None,
    )
    refreshed = db_session.get(AppraisalReport, report.id)
    assert refreshed.concluded_value is None


def test_toggle_pin_flips_state_and_respects_max_pinned(db_session):
    report = _make_report(db_session)
    pool_ids = []
    for i in range(comparable_service.MAX_PINNED + 1):
        listing = _make_listing(
            db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=100000, area_sqm_model=100
        )
        item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
        db_session.add(item)
        db_session.commit()
        db_session.refresh(item)
        pool_ids.append(item.id)

    # Pin the first MAX_PINNED items -- all should succeed.
    for pid in pool_ids[:comparable_service.MAX_PINNED]:
        assert comparable_service.toggle_pin(db_session, pid) is True

    # The (MAX_PINNED + 1)th item should be rejected -- pool already at cap.
    assert comparable_service.toggle_pin(db_session, pool_ids[-1]) is False

    # Unpinning one of the already-pinned items should still work (toggle off).
    assert comparable_service.toggle_pin(db_session, pool_ids[0]) is False


def test_get_pool_with_stats_computes_percentiles_and_counts(db_session):
    report = _make_report(db_session)
    prices_and_areas = [(100000, 100), (150000, 100), (200000, 100)]  # ppsqm: 1000, 1500, 2000
    for price, area in prices_and_areas:
        listing = _make_listing(
            db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=price, area_sqm_model=area
        )
        db_session.add(ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id))
    db_session.commit()

    result = comparable_service.get_pool_with_stats(db_session, "sale", report.id)

    assert result["total_count"] == 3
    assert result["pinned_count"] == 0
    stats = result["stats"]
    assert stats["n"] == 3
    assert float(stats["min_ppsqm"]) == pytest.approx(1000)
    assert float(stats["max_ppsqm"]) == pytest.approx(2000)
    assert float(stats["median"]) == pytest.approx(1500)


def test_get_pool_with_stats_returns_none_stats_for_empty_pool(db_session):
    report = _make_report(db_session)
    result = comparable_service.get_pool_with_stats(db_session, "sale", report.id)
    assert result["total_count"] == 0
    assert result["stats"] is None


# ── generate_docx: purpose-differentiated front matter ─────────────────────────

def _docx_text(db_session, report) -> str:
    buf = comparable_service.generate_docx(db_session, report)
    doc = Document(buf)
    return "\n".join(p.text for p in doc.paragraphs)


@pytest.mark.parametrize("purpose, must_contain, must_not_contain", [
    ("market_opinion", "не следва да се използва за целите на финансова отчетност", "МСФО 13"),
    ("fair_value_ifrs", "МСФО 13", "чл. 72, ал. 2 от Търговския закон"),
    ("noncash_contribution", "чл. 72, ал. 2 от Търговския закон", "МСФО 13"),
])
def test_generate_docx_uses_purpose_specific_boilerplate(db_session, purpose, must_contain, must_not_contain):
    report = _make_report(db_session, report_purpose=purpose)
    text = _docx_text(db_session, report)
    assert must_contain in text
    assert must_not_contain not in text


def test_generate_docx_shows_appraiser_identity_when_set(db_session):
    user = User(
        email="appraiser-test@example.com", username="appraisertest",
        hashed_password="x", full_name="Иван Петров Иванов",
        appraiser_certificate_no="100102183",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    report = _make_report(db_session, user_id=user.id)
    text = _docx_text(db_session, report)
    assert "Иван Петров Иванов" in text
    assert "100102183" in text


def test_generate_docx_omits_appraiser_line_when_no_owner(db_session):
    report = _make_report(db_session)  # user_id stays NULL
    text = _docx_text(db_session, report)
    assert "Изготвил оценката" not in text


def test_generate_docx_always_includes_limiting_conditions_appendix(db_session):
    report = _make_report(db_session)
    text = _docx_text(db_session, report)
    assert "Ограничаващи условия и допускания" in text


def test_generate_docx_includes_legal_section_only_when_present(db_session):
    with_legal = _make_report(db_session, legal_description="Тестов правен текст за проверка.")
    without_legal = _make_report(db_session)

    text_with = _docx_text(db_session, with_legal)
    text_without = _docx_text(db_session, without_legal)

    assert "Правно и градоустройствено състояние" in text_with
    assert "Тестов правен текст за проверка." in text_with
    assert "Правно и градоустройствено състояние" not in text_without


# ── Tier 3: structured multi-factor comparable adjustments ─────────────────────

def test_update_pool_adjustment_factors_mode_derives_total_and_drops_zero_factors(db_session):
    report = _make_report(db_session)
    listing = _make_listing(db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=150000, area_sqm_model=80)
    item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    comparable_service.update_pool_adjustment(
        db_session, item.id, None, "note",
        adjustment_factors={"market": -5, "location": 7, "size": -8, "floor": 0, "condition": 0},
    )
    refreshed = db_session.get(ComparablePool, item.id)
    assert float(refreshed.adjustment_pct) == pytest.approx(-6.0)  # -5+7-8+0+0
    assert refreshed.adjustment_factors == {"market": -5, "location": 7, "size": -8}  # zeros dropped
    assert refreshed.analyst_note == "note"


def test_update_pool_adjustment_simple_mode_leaves_factors_untouched(db_session):
    report = _make_report(db_session)
    listing = _make_listing(db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=150000, area_sqm_model=80)
    item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    comparable_service.update_pool_adjustment(db_session, item.id, -3.5, "manual note")
    refreshed = db_session.get(ComparablePool, item.id)
    assert float(refreshed.adjustment_pct) == pytest.approx(-3.5)
    assert refreshed.adjustment_factors is None


def test_update_pool_adjustment_all_zero_factors_clears_to_none(db_session):
    report = _make_report(db_session)
    listing = _make_listing(db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=150000, area_sqm_model=80)
    item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    comparable_service.update_pool_adjustment(
        db_session, item.id, None, "", adjustment_factors={"market": 5},
    )
    comparable_service.update_pool_adjustment(
        db_session, item.id, None, "", adjustment_factors={"market": 0},
    )
    refreshed = db_session.get(ComparablePool, item.id)
    assert refreshed.adjustment_factors is None
    assert refreshed.adjustment_pct is None


def test_generate_docx_shows_factor_breakdown_table_only_for_pinned_items_with_factors(db_session):
    report = _make_report(
        db_session,
        submarket_rationale="Избрани са сравними от непосредствената околност.",
    )
    listing = _make_listing(db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=150000, area_sqm_model=80)
    item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)

    comparable_service.update_pool_adjustment(
        db_session, item.id, None, "", adjustment_factors={"market": -5, "location": 7, "size": -8},
    )
    comparable_service.toggle_pin(db_session, item.id)

    text = _docx_text(db_session, report)
    assert "Разбивка на корекциите по фактори" in text
    assert "Обосновка на съпоставимата зона" in text
    assert "Избрани са сравними от непосредствената околност." in text


def test_generate_docx_omits_factor_breakdown_table_when_no_pinned_factors(db_session):
    report = _make_report(db_session)
    listing = _make_listing(db_session, ad_url=f"https://imot.bg/test-{uuid.uuid4()}", total_price=150000, area_sqm_model=80)
    item = ComparablePool(listing_id=listing.id, comparable_type="sale", report_id=report.id)
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    comparable_service.toggle_pin(db_session, item.id)  # pinned, but no adjustment_factors

    text = _docx_text(db_session, report)
    assert "Разбивка на корекциите по фактори" not in text


def test_update_submarket_rationale_persists_text(db_session):
    report = _make_report(db_session)
    comparable_service.update_submarket_rationale(db_session, report.id, "Test rationale")
    refreshed = db_session.get(AppraisalReport, report.id)
    assert refreshed.submarket_rationale == "Test rationale"

    comparable_service.update_submarket_rationale(db_session, report.id, "")
    refreshed = db_session.get(AppraisalReport, report.id)
    assert refreshed.submarket_rationale is None
