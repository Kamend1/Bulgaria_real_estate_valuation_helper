"""
Regression tests (2026-09-11, found during a stale-code/wrong-link audit):
get_cities_for_filter/get_quarters_for_filter/get_property_types_for_filter
(app/services/listing_service.py) counted ALL listings regardless of
status, while search_listings/get_all_listing_ids (which actually execute
the search these filters feed) both filter status='active' -- same bug
class as the home-page stats fix. A city/property-type could show a
nonzero sidebar count made up entirely of archived listings, then return
zero results when checked, because the count and the search disagreed on
what "current" means.

Uses db_session (real DB, wrapped in a transaction that's always rolled
back -- see tests/conftest.py), so disposable rows need no manual cleanup.
"""
from app.db.models import Listing, TaxonomyPropertyType
from app.services.listing_service import (
    get_cities_for_filter,
    get_property_types_for_filter,
    get_quarters_for_filter,
)

_TEST_CITY = "Регресионен Тестов Град"
_TEST_QUARTER = "Регресионен Тестов Квартал"
_TEST_SLUG = "regression_test_slug"


def _add_archived_listing(db_session, ad_url_suffix: str, **overrides):
    vals = dict(
        ad_url=f"https://test.invalid/filter-counts-{ad_url_suffix}",
        deal_type_normalized="sale",
        status="archived",
        title_city_model=_TEST_CITY,
        location_level_2_model=_TEST_QUARTER,
        geo_category="regional_city",
        property_type_slug=_TEST_SLUG,
    )
    vals.update(overrides)
    listing = Listing(**vals)
    db_session.add(listing)
    db_session.flush()
    return listing


def test_get_cities_for_filter_excludes_archived_listings(db_session):
    _add_archived_listing(db_session, "city-1")
    cities = get_cities_for_filter(db_session)
    assert _TEST_CITY not in [c for c, _n in cities]


def test_get_quarters_for_filter_excludes_archived_listings(db_session):
    _add_archived_listing(db_session, "quarter-1")
    quarters = get_quarters_for_filter(db_session, [_TEST_CITY])
    assert _TEST_QUARTER not in quarters


def test_get_property_types_for_filter_excludes_archived_listings(db_session):
    db_session.add(TaxonomyPropertyType(slug=_TEST_SLUG, display_name_bg="Тест", route_count=0))
    db_session.flush()
    _add_archived_listing(db_session, "ptype-1")

    types = get_property_types_for_filter(db_session)
    row = next((t for t in types if t[0] == _TEST_SLUG), None)
    assert row is not None
    assert row[2] == 0, "archived listing must not count toward the property-type total"
