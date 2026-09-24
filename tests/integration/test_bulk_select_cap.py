"""
Regression tests (2026-09-17): "select all matching pages" on /listings/
(get_all_listing_ids) had a docstring claiming "(max 5000)" but never
actually applied any LIMIT in its SQL -- a broad filter could hand back
every matching row (found live: 8,715 ids in one response for a real,
not-especially-narrow filter), which a plain <form> POST to
/comparables/add then happily accepted (Starlette's form-field cap only
applies to multipart bodies, not the urlencoded body a checkbox form
sends -- verified directly against starlette.formparsers), producing a
comparable_pool with thousands of rows: a real performance problem, since
every later pool render/htmx swap walks all of them.

Uses db_session (real DB, wrapped in a transaction that's always rolled
back -- see tests/conftest.py), so disposable rows need no manual cleanup.
"""
import uuid

from app.db.models import AppraisalReport, ComparablePool, Listing, User
from app.services import comparable_service, listing_service
from app.services.listing_service import SearchFilters

_TEST_SLUG = "bulk-select-cap-test-slug"


def _make_listing(db_session, ad_url_suffix: str) -> Listing:
    listing = Listing(
        ad_url=f"https://test.invalid/bulk-select-cap-{ad_url_suffix}",
        deal_type_normalized="sale",
        status="active",
        property_type_slug=_TEST_SLUG,
        total_price=100_000,
        area_sqm_model=50,
        price_per_sqm_model=2000,
    )
    db_session.add(listing)
    db_session.flush()
    return listing


def _make_report(db_session) -> AppraisalReport:
    report = AppraisalReport(title="Тест доклад", status="draft")
    db_session.add(report)
    db_session.flush()
    return report


def test_get_all_listing_ids_caps_but_reports_true_total(db_session, monkeypatch):
    monkeypatch.setattr(listing_service, "MAX_BULK_SELECT_IDS", 3)
    for i in range(5):
        _make_listing(db_session, f"cap-{i}")

    ids, total = listing_service.get_all_listing_ids(
        db_session, SearchFilters(property_types=[_TEST_SLUG]),
    )
    assert len(ids) == 3, "must stop at the cap, not return every match"
    assert total == 5, "the reported total must be the TRUE match count, not len(ids)"


def test_listing_ids_route_flags_capped_result(client, db_session, monkeypatch):
    monkeypatch.setattr(listing_service, "MAX_BULK_SELECT_IDS", 2)
    for i in range(4):
        _make_listing(db_session, f"route-cap-{i}")

    resp = client.get("/listings/ids", params={"property_types": [_TEST_SLUG]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ids"]) == 2
    assert data["total"] == 4
    assert data["capped"] is True


def test_add_to_pool_truncates_oversized_listing_id_list(db_session, monkeypatch):
    monkeypatch.setattr(comparable_service, "MAX_BULK_SELECT_IDS", 2)
    # Own user, not a hardcoded id: CI starts from an empty database.
    user = User(email="bulk-cap-test@example.com", username="bulkcaptest", hashed_password="x")
    db_session.add(user)
    db_session.flush()
    report = _make_report(db_session)
    listings = [_make_listing(db_session, f"add-cap-{i}") for i in range(5)]
    listing_ids = [listing.id for listing in listings]

    inserted = comparable_service.add_to_pool(
        db_session, listing_ids, "sale", report.id, user_id=user.id,
    )
    assert inserted == 2, "must silently cap even a maliciously/accidentally oversized request"
    pool_count = (
        db_session.query(ComparablePool)
        .filter_by(report_id=report.id, comparable_type="sale")
        .count()
    )
    assert pool_count == 2
