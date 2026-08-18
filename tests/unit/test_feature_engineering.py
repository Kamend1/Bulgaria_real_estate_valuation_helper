"""
Unit tests for utils/feature_engineering/feature_engineering_utils.py — pure
functions with no I/O, run on every parsed listing before DB insertion.
Expected values were captured from the functions' own live output via a
scratch script (not independently derived), then encoded as regression
assertions.
"""
import pytest

from utils.feature_engineering.feature_engineering_utils import (
    correct_agricultural_area,
    map_geo_category,
    normalize_deal_type,
    parse_published_date,
)
from datetime import date


@pytest.mark.parametrize("raw, expected", [
    ("Публикувана в 13:42 на 31 яну, 2014 год.", date(2014, 1, 31)),
    ("Публикувана в 09:00 на 1 март, 2020 год.", date(2020, 3, 1)),
    ("Публикувана на 15 декември 2019", date(2019, 12, 15)),
    ("някакъв текст 2018 без дата", date(2018, 1, 1)),
    (None, None),
    ("", None),
    ("без никакви цифри", None),
])
def test_parse_published_date(raw, expected):
    assert parse_published_date(raw) == expected


def test_parse_published_date_falls_back_to_year_only_on_invalid_calendar_date():
    # 29 Feb 2021 doesn't exist (2021 isn't a leap year) -> date(2021, 2, 29)
    # raises ValueError internally, function falls back to year-only Jan 1.
    assert parse_published_date("Публикувана в 10:00 на 29 фев, 2021 год.") == date(2021, 1, 1)


@pytest.mark.parametrize("raw, expected", [
    ("Продава", "sale"),
    ("Дава под наем", "rent"),
    ("продава апартамент", "sale"),
    ("наем", "rent"),
    (None, "unknown"),
    ("", "unknown"),
    ("нещо друго", "unknown"),
])
def test_normalize_deal_type(raw, expected):
    assert normalize_deal_type(raw) == expected


@pytest.mark.parametrize("ptype, area, price, expected_area, expected_ppsqm", [
    ("ЗЕМЕДЕЛСКА ЗЕМЯ", 5.0, 10000.0, 5000.0, 2.0),        # decares -> sqm
    ("земеделска земя", 5.0, 10000.0, 5000.0, 2.0),        # lowercase variant also matches
    ("ЗЕМЕДЕЛСКА ЗЕМЯ", 600.0, 60000.0, 600.0, 100.0),     # area > 500 -> not converted
    ("Апартамент", 5.0, 10000.0, 5.0, 2000.0),             # non-agricultural -> untouched
    ("ЗЕМЕДЕЛСКА ЗЕМЯ", 5.0, None, 5.0, None),
    ("ЗЕМЕДЕЛСКА ЗЕМЯ", 0.005, 10000.0, 0.005, 2000000.0),  # area < 0.01 -> not converted
])
def test_correct_agricultural_area(ptype, area, price, expected_area, expected_ppsqm):
    result_area, result_ppsqm = correct_agricultural_area(ptype, area, price)
    assert result_area == pytest.approx(expected_area)
    if expected_ppsqm is None:
        assert result_ppsqm is None
    else:
        assert result_ppsqm == pytest.approx(expected_ppsqm)


def test_correct_agricultural_area_returns_none_none_when_area_missing():
    assert correct_agricultural_area("ЗЕМЕДЕЛСКА ЗЕМЯ", None, 10000.0) == (None, None)


@pytest.mark.parametrize("l1, l2, l3, city, geo2, expected", [
    ("софия", "лозенец", None, "софия", None, "sofia_center"),
    ("софия", "младост", None, "софия", None, "sofia_other"),
    ("софия", None, None, "софия", "център", "sofia_center"),
    (None, None, None, "пловдив", None, "large_regional_city"),
    (None, None, None, "русе", None, "regional_city"),
    ("несебър", None, None, None, None, "sea_resort"),
    ("банско", None, None, None, None, "mountain_resort"),
    ("област монтана", None, None, None, None, "small_city"),
    ("някакво си село", None, None, None, None, "small_city"),
    (None, None, None, None, None, "other_unknown"),
    ("гърция", None, None, None, None, "foreign"),
    (None, None, None, "атина", None, "foreign"),
])
def test_map_geo_category(l1, l2, l3, city, geo2, expected):
    assert map_geo_category(l1, l2, l3, city, geo2) == expected


def test_map_geo_category_sea_and_mountain_resorts_only_checked_via_location_levels_not_city():
    # Documenting actual (possibly surprising) behavior: a resort name in
    # title_city_model alone, with no location_level match, does NOT get
    # classified as sea_resort/mountain_resort -- only location_level_1/2
    # are checked against those sets, city only feeds the regional-city
    # and large-regional-city checks. Not asserting this is correct, just
    # guarding against a silent change.
    assert map_geo_category(None, None, None, "несебър", None) == "other_unknown"
    assert map_geo_category(None, None, None, "банско", None) == "other_unknown"
