"""
Integration tests for app/services/comparable_service.py's update_* and
pool-statistics functions, run against the real `appraisal` DB inside a
transaction that's always rolled back (see tests/conftest.py::db_session).
"""
import uuid

import pytest

from app.db.models import AppraisalReport, ComparablePool, Listing
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


def test_update_income_approach_computes_annual_rent_cap_rate_and_concluded_value(db_session):
    report = _make_report(db_session, subject_area_sqm=80.0)
    comparable_service.update_income_approach(
        db_session, report.id,
        rent_per_sqm_month=10.0, cap_rate_pct=6.5,
        concluded_per_sqm=1200.0, subject_area_sqm=None,
    )

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.annual_rent_estimate) == pytest.approx(120.0)          # 10 * 12
    assert float(refreshed.capitalization_rate) == pytest.approx(0.065)           # 6.5 / 100
    assert float(refreshed.concluded_value_income) == pytest.approx(96000.0)      # 1200 * 80 (subject_area_sqm)


def test_update_income_approach_uses_explicit_area_over_subject_area(db_session):
    report = _make_report(db_session, subject_area_sqm=80.0)
    comparable_service.update_income_approach(
        db_session, report.id,
        rent_per_sqm_month=None, cap_rate_pct=None,
        concluded_per_sqm=1000.0, subject_area_sqm=50.0,
    )
    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_income) == pytest.approx(50000.0)


def test_update_residual_approach_sets_value(db_session):
    report = _make_report(db_session)
    comparable_service.update_residual_approach(db_session, report.id, 75000.0)

    refreshed = db_session.get(AppraisalReport, report.id)
    assert float(refreshed.concluded_value_residual) == pytest.approx(75000.0)


def test_update_subject_ignored_for_unknown_report_id(db_session):
    # Should silently no-op, not raise, for a report_id that doesn't exist.
    comparable_service.update_sales_approach(db_session, uuid.uuid4(), 1000.0)


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
