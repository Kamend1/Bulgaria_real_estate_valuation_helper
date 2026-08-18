"""
Regression tests for Bulgarian number-to-words spelling
(utils/gis/engines/numwords_bg.py) — used to spell out cadastral
identifiers and areas in generated legal descriptions. Expected values
were captured from the function's own live output (already used to
generate real, checked legal text elsewhere in this app), not
independently derived — these are regression guards against an
accidental behavior change, not an independent grammar authority.
"""
import pytest

from utils.gis.engines import numwords_bg as nw


@pytest.mark.parametrize("n, expected", [
    (0, "нула"),
    (1, "един"),
    (5, "пет"),
    (10, "десет"),
    (11, "единадесет"),
    (15, "петнадесет"),
    (20, "двадесет"),
    (21, "двадесет и един"),
    (99, "деветдесет и девет"),
    (100, "сто"),
    (101, "сто един"),
    (111, "сто единадесет"),
    (200, "двеста"),
    (999, "деветстотин деветдесет и девет"),
    (1000, "хиляда"),          # not "една хиляда"
    (1001, "хиляда и един"),
    (1100, "хиляда и сто"),
    (2000, "две хиляди"),
    (2500, "две хиляди и петстотин"),
    (-5, "минус пет"),
])
def test_integer_words(n, expected):
    assert nw.integer_words(n) == expected


def test_integer_words_21000_thousands_agreement():
    # "двадесет и една хиляди" (not "хиляда") — flagging in case this ever
    # looks wrong to a native speaker; not asserting it's incorrect, this
    # is what the ported (already-in-production) function has always done.
    assert nw.integer_words(21000) == "двадесет и една хиляди"


@pytest.mark.parametrize("value, expected", [
    (0.0, "нула квадратни метра"),
    (1.0, "един квадратен метър"),       # singular for 1
    (2.0, "два квадратни метра"),
    (11.0, "единадесет квадратни метра"),  # NOT singular — the "ends in 1
                                            # but not 11" rule
    (21.0, "двадесет и един квадратен метър"),  # singular again for 21
    (100.0, "сто квадратни метра"),
    (1625.0, "хиляда шестстотин двадесет и пет квадратни метра"),
    (524.75, "петстотин двадесет и четири цяло и седемдесет и пет стотни квадратни метра"),
    (0.5, "нула цяло и петдесет стотни квадратни метра"),
])
def test_area_words(value, expected):
    assert nw.area_words(value) == expected


def test_area_words_rounds_99_999_cents_up_into_whole_number():
    # cents=100 after rounding must carry into the whole part, not render
    # as "... цяло и 100 стотни".
    assert nw.area_words(99.999) == "сто квадратни метра"


@pytest.mark.parametrize("identifier, expected", [
    ("68134.905.1462", "шест, осем, едно, три, четири, точка, девет, нула, пет, точка, едно, четири, шест, две"),
    ("15285.13.286.2", "едно, пет, две, осем, пет, точка, едно, три, точка, две, осем, шест, точка, две"),
    ("0", "нула"),
    ("", ""),
])
def test_spell_identifier(identifier, expected):
    assert nw.spell_identifier(identifier) == expected


@pytest.mark.parametrize("value, expected", [
    (1625.0, "1625"),      # whole -> no decimals
    (524.75, "524.75"),    # fractional -> 2 decimals
    (100, "100"),
    (100.0, "100"),
])
def test_format_area(value, expected):
    assert nw.format_area(value) == expected
