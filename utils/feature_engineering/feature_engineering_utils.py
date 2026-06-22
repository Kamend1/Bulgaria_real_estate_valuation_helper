"""
Feature engineering transformations extracted from notebook 03.
Called on every parsed listing dict before DB insertion.
"""

import re
from datetime import date


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

_PREFIXES = [
    "гр. ", "град ", "с. ", "село ", "к.к. ", "кк ", "курорт ",
    "в.з. ", "м-т ", "м. ", "местност ",
]


def normalize_bg_text(value: str | None) -> str | None:
    if not value:
        return None

    value = str(value).strip().lower()
    value = value.replace("ё", "е")
    value = value.replace("̀", "")   # combining grave accent
    value = value.replace("'", "").replace('"', "")
    value = " ".join(value.split())

    if value in {"", "none", "nan", "<na>"}:
        return None

    return value


def remove_geo_prefix(value: str | None) -> str | None:
    v = normalize_bg_text(value)
    if not v:
        return None

    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if v.startswith(prefix):
                v = v[len(prefix):].strip()
                changed = True

    return v if v else None


def normalize_settlement_name(value: str | None) -> str | None:
    return remove_geo_prefix(value)


# ============================================================
# 2. PUBLISHED DATE PARSING
# ============================================================

MONTHS_BG: dict[str, int] = {
    "яну": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "юни": 6,
    "юли": 7, "авг": 8, "сеп": 9, "окт": 10, "ное": 11, "дек": 12,
    "януари": 1, "февруари": 2, "март": 3, "април": 4,
    "август": 8, "септември": 9, "октомври": 10, "ноември": 11, "декември": 12,
}


def parse_published_date(published_raw: str | None) -> date | None:
    """
    Parses "Публикувана в 13:42 на 31 яну, 2014 год." → date(2014, 1, 31).
    Falls back to year-only (Jan 1) if full date unavailable.
    """
    if not published_raw:
        return None

    m = re.search(r"(\d{1,2})\s+([а-яА-Я]+),?\s+(\d{4})", published_raw)
    if m:
        day = int(m.group(1))
        month_raw = m.group(2).lower()
        year = int(m.group(3))
        month = MONTHS_BG.get(month_raw[:3]) or MONTHS_BG.get(month_raw)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Fallback: year only
    m = re.search(r"(\d{4})", published_raw)
    if m:
        year = int(m.group(1))
        if 2005 <= year <= 2035:
            return date(year, 1, 1)

    return None


# ============================================================
# 3. LOCATION PARSING
# ============================================================

def parse_location_levels(location_raw: str | None) -> dict:
    result: dict[str, str | None] = {
        "location_level_1": None,
        "location_level_2": None,
        "location_level_3": None,
        "location_level_1_model": None,
        "location_level_2_model": None,
        "location_level_3_model": None,
    }

    if not location_raw or str(location_raw).strip().lower() in {"none", "nan", ""}:
        return result

    parts = [p.strip() for p in str(location_raw).split(",")]

    for i, key in enumerate(["location_level_1", "location_level_2", "location_level_3"]):
        if i < len(parts) and parts[i]:
            result[key] = parts[i]
            result[f"{key}_model"] = normalize_bg_text(parts[i])

    return result


# ============================================================
# 4. GEO CATEGORY MAPPING
# ============================================================

_SOFIA_CENTER_AREAS = {
    "център", "идеален център", "докторски паметник", "яворов",
    "оборише", "оборище", "лозенец", "иван вазов",
    "зона б-5", "зона б-18", "зона б-19", "банишора",
    "сердика", "възраждане", "медицинска академия", "редута",
    "трите чучура", "докторска градина",
}

_LARGE_REGIONAL_CITIES = {"пловдив", "варна", "бургас"}

_REGIONAL_CITIES = {
    "русе", "стара загора", "плевен", "велико търново", "шумен",
    "добрич", "сливен", "хасково", "ямбол", "перник", "благоевград",
    "пазарджик", "враца", "видин", "монтана", "ловеч", "габрово",
    "търговище", "кърджали", "разград", "силистра", "смолян",
    "кюстендил", "vidin",
}

_SEA_RESORTS = {
    "несебър", "слънчев бряг", "свети влас", "созопол", "приморско",
    "обзор", "равда", "ахелой", "поморие", "равда", "черноморец",
    "китен", "лозенец", "синеморец", "царево", "ахтопол",
    "балчик", "каварна", "шабла", "шкорпиловци", "дюни",
    "елените", "св. константин и елена", "свети константин",
    "бяла", "голям дол",
}

_MOUNTAIN_RESORTS = {
    "банско", "боровец", "пампорово", "чепеларе", "велинград",
    "самоков", "ихтиман", "белково", "говедарци", "долна баня",
    "костенец", "якоруда", "разлог", "добринище", "сапарева баня",
    "широка лъка", "смолян",
}

_FOREIGN = {"гърция", "greece", "атина", "солун", "хале", "кипър"}


def _norm(v: str | None) -> str:
    return str(v or "").strip().lower()


def map_geo_category(
    level_1_model: str | None,
    level_2_model: str | None,
    level_3_model: str | None,
    title_city_model: str | None,
    title_geo_2_model: str | None = None,
) -> str:
    l1 = _norm(level_1_model)
    l2 = _norm(level_2_model)
    l3 = _norm(level_3_model)
    city = _norm(title_city_model)
    geo2 = _norm(title_geo_2_model)

    # Foreign
    if l1 in _FOREIGN or city in _FOREIGN:
        return "foreign"

    is_sofia = "софия" in l1 or "sofia" in city or "софия" in city

    if is_sofia:
        # Check any location level and title_geo_2 for center areas
        for candidate in (l2, l3, geo2):
            bare = remove_geo_prefix(candidate) or candidate
            if bare in _SOFIA_CENTER_AREAS or candidate in _SOFIA_CENTER_AREAS:
                return "sofia_center"
        return "sofia_other"

    # Sea resort — check both levels
    l1_bare = remove_geo_prefix(l1) or l1
    l2_bare = remove_geo_prefix(l2) or l2
    if l1_bare in _SEA_RESORTS or l2_bare in _SEA_RESORTS:
        return "sea_resort"

    # Mountain resort
    if l1_bare in _MOUNTAIN_RESORTS or l2_bare in _MOUNTAIN_RESORTS:
        return "mountain_resort"

    # Large regional city
    city_bare = remove_geo_prefix(city) or city
    l1_city = remove_geo_prefix(l1) or l1
    if city_bare in _LARGE_REGIONAL_CITIES or l1_city in _LARGE_REGIONAL_CITIES:
        return "large_regional_city"

    # Regional city
    if city_bare in _REGIONAL_CITIES or l1_city in _REGIONAL_CITIES:
        return "regional_city"

    # Oblast-level → small city
    if l1.startswith("област"):
        return "small_city"

    if l1:
        return "small_city"

    return "other_unknown"


# ============================================================
# 5. AREA CORRECTION (agricultural land entered in decares)
# ============================================================

_AGRICULTURAL_TYPES = {"земеделска земя", "ЗЕМЕДЕЛСКА ЗЕМЯ"}


def correct_agricultural_area(
    property_type_raw: str | None,
    area_sqm: float | None,
    total_price: float | None,
) -> tuple[float | None, float | None]:
    """Returns (area_sqm_corrected, price_per_sqm_corrected)."""
    if area_sqm is None or total_price is None:
        return area_sqm, None

    area = float(area_sqm)
    price = float(total_price)

    ptype = str(property_type_raw or "").strip()
    if ptype in _AGRICULTURAL_TYPES and 0.01 <= area <= 500:
        area = area * 1000  # decares → sqm

    ppsqm = round(price / area, 2) if area > 0 else None
    return area, ppsqm


# ============================================================
# 6. CONSTRUCTION ENGINEERING
# ============================================================

_LAND_TYPES = {"ПАРЦЕЛ", "ЗЕМЕДЕЛСКА ЗЕМЯ", "парцел", "земеделска земя"}


def engineer_construction(
    property_type_raw: str | None,
    construction_type: str | None,
    construction_year: int | float | None,
) -> dict:
    ptype = str(property_type_raw or "").strip()

    if ptype in _LAND_TYPES:
        return {
            "construction_type_model": "not_applicable_land",
            "construction_year_model": None,
        }

    year = None
    if construction_year is not None:
        try:
            y = int(construction_year)
            if 1850 <= y <= 2030:
                year = y
        except (ValueError, TypeError):
            pass

    return {
        "construction_type_model": construction_type,
        "construction_year_model": year,
    }


# ============================================================
# 7. FLOOR ENGINEERING
# ============================================================

_FLOOR_NOT_APPLICABLE = {
    "ПАРЦЕЛ", "ЗЕМЕДЕЛСКА ЗЕМЯ", "СОЛАРЕН ПАРК",
    "парцел", "земеделска земя", "соларен парк",
}


def engineer_floors(
    property_type_raw: str | None,
    floor: int | float | None,
    total_floors: int | float | None,
) -> dict:
    ptype = str(property_type_raw or "").strip()

    if ptype in _FLOOR_NOT_APPLICABLE:
        return {
            "floor_model": None,
            "total_floors_model": None,
            "floor_applicability": "not_applicable_non_building",
        }

    def _safe_int(v) -> int | None:
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "floor_model": _safe_int(floor),
        "total_floors_model": _safe_int(total_floors),
        "floor_applicability": "applicable",
    }


# ============================================================
# 8. PROPERTY TYPE SLUG EXTRACTION
# ============================================================

# Canonical slugs from taxonomy, sorted longest-first so greedy match wins.
_PROPERTY_SLUGS: tuple[str, ...] = tuple(sorted([
    "atelie-tavan", "biznes-imot", "chetiristaen", "dvustaen", "ednostaen",
    "etazh-ot-kashta", "garazh-parkomyasto", "hotel", "kashta", "magazin",
    "mezonet", "mnogostaen", "myasto", "ofis", "partsel",
    "promishleno-pomeshtenie", "sklad", "staya", "tristaen", "vila",
    "zavedenie", "zemedelska-zemya",
], key=len, reverse=True))

# Deal slugs that appear in the URL between ad-id and property-type.
_DEAL_SLUG_RE = re.compile(r"-(?:prodava|dava-pod-naem|naem)-(.+)")


def extract_property_type_slug(ad_url: str | None) -> str | None:
    """
    Extracts the canonical taxonomy slug from an imot.bg ad URL.
    URL format: /obiava-{id}-prodava-{property_slug}-{location_slug}
    """
    if not ad_url:
        return None
    m = _DEAL_SLUG_RE.search(ad_url)
    if not m:
        return None
    remainder = m.group(1)
    for slug in _PROPERTY_SLUGS:
        if remainder.startswith(slug + "-") or remainder == slug:
            return slug
    return None


# Display name mapping: slug → Bulgarian label used in UI.
PROPERTY_TYPE_DISPLAY: dict[str, str] = {
    "ednostaen":              "Едностаен апартамент",
    "dvustaen":               "Двустаен апартамент",
    "tristaen":               "Тристаен апартамент",
    "chetiristaen":           "Четиристаен апартамент",
    "mnogostaen":             "Многостаен апартамент",
    "mezonet":                "Мезонет",
    "atelie-tavan":           "Ателие / Таван",
    "staya":                  "Стая",
    "etazh-ot-kashta":        "Етаж от къща",
    "kashta":                 "Къща",
    "vila":                   "Вила",
    "myasto":                 "Място",
    "garazh-parkomyasto":     "Гараж / Паркомясто",
    "partsel":                "Парцел",
    "zemedelska-zemya":       "Земеделска земя",
    "ofis":                   "Офис",
    "magazin":                "Магазин",
    "sklad":                  "Склад",
    "zavedenie":              "Заведение",
    "hotel":                  "Хотел",
    "biznes-imot":            "Бизнес имот",
    "promishleno-pomeshtenie": "Промишлено помещение",
}


# ============================================================
# 9. DEAL TYPE NORMALIZATION
# ============================================================

def normalize_deal_type(deal_raw: str | None) -> str:
    if not deal_raw:
        return "unknown"
    d = str(deal_raw).strip().lower()
    if "продава" in d:
        return "sale"
    if "наем" in d or "дава под" in d:
        return "rent"
    return "unknown"


# ============================================================
# 10. MASTER FUNCTION
# ============================================================

def engineer_features(parsed_row: dict) -> dict:
    """
    Enriches one raw parsed_row dict with all engineered fields.
    Returns a new dict (does not mutate the input).
    """
    row = dict(parsed_row)

    # Deal type
    row["deal_type_normalized"] = normalize_deal_type(row.get("deal_raw"))

    # Property type taxonomy slug (from URL)
    row["property_type_slug"] = extract_property_type_slug(row.get("ad_url"))

    # Published date
    row["published_date"] = parse_published_date(row.get("published_raw"))

    # Area correction + price_per_sqm_model
    area, ppsqm = correct_agricultural_area(
        row.get("property_type_raw"),
        row.get("area_sqm"),
        row.get("total_price"),
    )
    row["area_sqm_model"] = area
    row["price_per_sqm_model"] = ppsqm

    # Geo normalization
    row["title_city_model"] = normalize_bg_text(row.get("title_city_raw"))
    row["title_geo_2_model"] = remove_geo_prefix(row.get("title_geo_2_raw"))

    location_fields = parse_location_levels(row.get("location_raw"))
    row.update(location_fields)

    row["geo_category"] = map_geo_category(
        row.get("location_level_1_model"),
        row.get("location_level_2_model"),
        row.get("location_level_3_model"),
        row.get("title_city_model"),
        row.get("title_geo_2_model"),
    )
    row["exclude_foreign"] = (row.get("geo_category") == "foreign")

    row.update(engineer_construction(
        row.get("property_type_raw"),
        row.get("construction_type"),
        row.get("construction_year"),
    ))

    row.update(engineer_floors(
        row.get("property_type_raw"),
        row.get("floor"),
        row.get("total_floors"),
    ))

    return row
