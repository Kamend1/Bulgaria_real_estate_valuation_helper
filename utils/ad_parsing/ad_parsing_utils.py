import re
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup


# ============================================================
# 1. HTML / TEXT HELPERS
# ============================================================

def read_html_file(html_path: str) -> str:
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


def html_to_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def html_to_lines(html: str) -> list[str]:
    soup = html_to_soup(html)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return lines


def lines_to_text(lines: list[str]) -> str:
    return "\n".join(lines)


def find_value_after_label(lines: list[str], label: str) -> str | None:
    for i, line in enumerate(lines):
        if line.strip().lower() == label.lower() and i + 1 < len(lines):
            return lines[i + 1].strip()

    return None


def find_first_line_containing(lines: list[str], keyword: str) -> str | None:
    keyword_lower = keyword.lower()

    for line in lines:
        if keyword_lower in line.lower():
            return line

    return None


def get_block_after_label(
    lines: list[str],
    start_label: str,
    stop_labels: set[str],
) -> str | None:
    start_idx = None

    for i, line in enumerate(lines):
        if line.strip().lower() == start_label.lower():
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    block_lines = []

    for line in lines[start_idx:]:
        if line in stop_labels:
            break

        block_lines.append(line)

    if not block_lines:
        return None

    return "\n".join(block_lines).strip()


# ============================================================
# 2. URL / AD ID PARSER
# ============================================================

def parse_ad_url(ad_url: str) -> dict:
    parsed = urlparse(ad_url)

    ad_url_path = parsed.path
    ad_url_slug = ad_url_path.strip("/")

    # Examples:
    # obiava-1a176294551327342-prodava-ednostaen-apartament-grad-sofiya-lozenets
    # obiava-8567-zhilishtna-sgrada-grad-sofiya-levski-v
    match = re.match(r"obiava-([^-]+)-?", ad_url_slug)

    ad_id = match.group(1) if match else None

    return {
        "ad_id": ad_id,
        "ad_url": ad_url,
        "ad_url_path": ad_url_path,
        "ad_url_slug": ad_url_slug,
    }


# ============================================================
# 3. PRICE / VAT / AREA PARSERS
# ============================================================

def clean_number_string(value: str) -> str:
    return (
        value
        .replace("\xa0", " ")
        .replace(" ", "")
        .replace(",", ".")
        .strip()
    )


def parse_number(value: str) -> float | None:
    try:
        return float(clean_number_string(value))
    except Exception:
        return None


def parse_price(text: str) -> dict:
    """
    Extracts total price.

    Handles:
    115 000 €
    115 000 EUR
    224 920.45 лв.
    """

    eur_patterns = [
        r"(\d[\d\s]{2,}(?:[.,]\d+)?)\s*(?:€|EUR)",
        r"(?:€|EUR)\s*(\d[\d\s]{2,}(?:[.,]\d+)?)",
    ]

    for pattern in eur_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = parse_number(match.group(1))

            return {
                "total_price": value,
                "currency": "EUR",
                "price_raw": match.group(0),
            }

    bgn_patterns = [
        r"(\d[\d\s]{2,}(?:[.,]\d+)?)\s*(?:лв\.?|BGN)",
        r"(?:лв\.?|BGN)\s*(\d[\d\s]{2,}(?:[.,]\d+)?)",
    ]

    for pattern in bgn_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = parse_number(match.group(1))

            return {
                "total_price": value,
                "currency": "BGN",
                "price_raw": match.group(0),
            }

    return {
        "total_price": None,
        "currency": None,
        "price_raw": None,
    }


def parse_vat_status(text: str) -> str:
    lower = text.lower()

    if "без ддс" in lower:
        return "without_vat"

    if (
        "с ддс" in lower
        or "вкл. ддс" in lower
        or "включен ддс" in lower
        or "включено ддс" in lower
    ):
        return "with_vat"

    if "не се начислява ддс" in lower:
        return "no_vat"

    return "unknown"


def parse_area_sqm(text: str) -> dict:
    """
    Extracts area.

    Handles:
    46 кв.м
    46 кв.м.
    46.5 кв.м
    """

    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*кв\.?\s*м\.?",
        r"(\d+(?:[.,]\d+)?)\s*m2",
        r"(\d+(?:[.,]\d+)?)\s*sqm",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return {
                "area_sqm": parse_number(match.group(1)),
                "area_raw": match.group(0),
            }

    return {
        "area_sqm": None,
        "area_raw": None,
    }


def calculate_price_per_sqm(total_price: float | None, area_sqm: float | None) -> float | None:
    if total_price is None or area_sqm is None or area_sqm <= 0:
        return None

    return round(total_price / area_sqm, 2)


# ============================================================
# 4. FLOOR PARSER
# ============================================================

def parse_floor_from_lines(lines: list[str]) -> dict:
    for i, line in enumerate(lines):
        if line.strip() == "Етаж:" and i + 1 < len(lines):
            raw = lines[i + 1].strip()

            lower = raw.lower()

            total_match = re.search(r"(?:от|/)\s*(\d+)", raw)
            total_floors = int(total_match.group(1)) if total_match else None

            if "партер" in lower:
                return {
                    "floor": 0,
                    "total_floors": total_floors,
                    "floor_raw": raw,
                }

            if "сутерен" in lower:
                return {
                    "floor": -1,
                    "total_floors": total_floors,
                    "floor_raw": raw,
                }

            floor_match = re.search(r"(\d+)", raw)
            floor = int(floor_match.group(1)) if floor_match else None

            return {
                "floor": floor,
                "total_floors": total_floors,
                "floor_raw": raw,
            }

    return {
        "floor": None,
        "total_floors": None,
        "floor_raw": None,
    }

# ============================================================
# 5. CONSTRUCTION PARSER
# ============================================================

CONSTRUCTION_TYPES = [
    "Тухла",
    "Панел",
    "ЕПК",
    "Гредоред",
    "Ново строителство",
    "Старо строителство",
    "Монолит",
    "Стоманобетон",
    "Сглобяема конструкция",
]


def parse_construction(text: str) -> dict:
    """
    Extracts construction type and construction year.

    Handles:
    Тухла, 2005
    Панел, 1980
    ЕПК, 1985
    Ново Строителство
    """

    construction_type = None

    for ctype in CONSTRUCTION_TYPES:
        if ctype.lower() in text.lower():
            construction_type = ctype
            break

    year_match = re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", text)
    construction_year = int(year_match.group(1)) if year_match else None

    construction_raw = None

    if construction_type and construction_year:
        construction_raw = f"{construction_type}, {construction_year}"
    elif construction_type:
        construction_raw = construction_type
    elif construction_year:
        construction_raw = str(construction_year)

    return {
        "construction_raw": construction_raw,
        "construction_type": construction_type,
        "construction_year": construction_year,
    }


# ============================================================
# 6. DESCRIPTION / FEATURES PARSER
# ============================================================
def parse_title_fields(title: str | None) -> dict:
    if not title:
        return {
            "deal_raw": None,
            "property_type_raw": None,
            "title_city_raw": None,
            "title_geo_2_raw": None,
            "title_area_sqm": None,
            "title_price_eur": None,
            "title_price_bgn": None,
            "seller_name_raw": None,
            "title_ad_id": None,
        }

    result = {
        "deal_raw": None,
        "property_type_raw": None,
        "title_city_raw": None,
        "title_geo_2_raw": None,
        "title_area_sqm": None,
        "title_price_eur": None,
        "title_price_bgn": None,
        "seller_name_raw": None,
        "title_ad_id": None,
    }

    # Deal + property type + location
    # Продава 1-СТАЕН в град София, Лагера - ...
    match = re.search(
        r"^(Продава|Дава под наем)\s+(.+?)\s+в\s+(.+?)\s+-",
        title,
        flags=re.IGNORECASE,
    )

    if match:
        result["deal_raw"] = match.group(1).strip()
        result["property_type_raw"] = match.group(2).strip()

        location_part = match.group(3).strip()
        location_parts = [p.strip() for p in location_part.split(",") if p.strip()]

        if len(location_parts) >= 1:
            result["title_city_raw"] = location_parts[0]

        if len(location_parts) >= 2:
            result["title_geo_2_raw"] = location_parts[1]

    # Area
    area_match = re.search(r"-\s*(\d+(?:[.,]\d+)?)\s*кв\.м", title)
    if area_match:
        result["title_area_sqm"] = parse_number(area_match.group(1))

    # EUR price
    eur_match = re.search(r"/\s*(\d[\d\s]*(?:[.,]\d+)?)\s*€", title)
    if eur_match:
        result["title_price_eur"] = parse_number(eur_match.group(1))

    # BGN price
    bgn_match = re.search(r"»\s*(\d[\d\s]*(?:[.,]\d+)?)\s*лв", title)
    if bgn_match:
        result["title_price_bgn"] = parse_number(bgn_match.group(1))

    # Seller between BGN price and :: imot.bg
    seller_match = re.search(r"лв\.,\s*(.*?)\s*::\s*imot\.bg", title)
    if seller_match:
        result["seller_name_raw"] = seller_match.group(1).strip()

    # Ad id
    ad_match = re.search(r"Обява\s+(\d+)", title)
    if ad_match:
        result["title_ad_id"] = ad_match.group(1)

    return result


def clean_multiline_text(value: str | None) -> str | None:
    if not value:
        return None

    lines = [
        line.strip()
        for line in value.splitlines()
        if line.strip()
    ]

    return " ".join(lines)


def parse_description(lines: list[str]) -> str | None:
    start_labels = {
        "Описание на имота:",
        "Описание:",
    }

    stop_labels = {
        "Особености",
        "Особености:",
        "За контакт:",
        "За контакти:",
        "Брокер:",
        "Агенция:",
        "Изпрати e-mail:",
        "Изпрати е-mail:",
        "Местоположение:",
        "Имотът се предлага от:",
        "Виж на Картата",
        "Виж карта",
        "Принтирай",
    }

    start_idx = None

    for i, line in enumerate(lines):
        if line.strip() in start_labels:
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    desc_lines = []

    for line in lines[start_idx:]:
        if line.strip() in stop_labels:
            break
        desc_lines.append(line)

    description = "\n".join(desc_lines).strip()

    return description if description else None


def parse_features(lines: list[str]) -> dict:
    start_labels = {
        "Особености",
        "Особености:",
    }

    stop_labels = {
        "За контакт:",
        "За контакти:",
        "Брокер:",
        "Агенция:",
        "Още оферти от брокера:",
        "Още оферти:",
        "Адрес:",
        "Изпрати e-mail:",
        "Изпрати е-mail:",
        "Местоположение:",
        "Имотът се предлага от:",
        "Принтирай",
    }

    start_idx = None

    for i, line in enumerate(lines):
        if line.strip() in start_labels:
            start_idx = i + 1
            break

    if start_idx is None:
        return {
            "features_raw": None,
            "features_pipe": None,
            "features_count": 0,
        }

    feature_lines = []

    for line in lines[start_idx:]:
        clean_line = line.strip()

        if clean_line in stop_labels:
            break

        if clean_line:
            feature_lines.append(clean_line)

    # Remove duplicates while preserving order
    seen = set()
    features = []

    for feature in feature_lines:
        if feature not in seen:
            seen.add(feature)
            features.append(feature)

    return {
        "features_raw": "\n".join(features) if features else None,
        "features_pipe": "|".join(features) if features else None,
        "features_count": len(features),
    }


# ============================================================
# 7. LISTING TYPE / LOCATION
# ============================================================

def classify_listing(text: str) -> str:
    if "Жилищна сграда" in text or "В сградата се продават:" in text:
        return "new_building_project"

    if "Описание на имота:" in text or "Описание:" in text:
        return "single_property_listing"

    return "unknown"


def parse_location(lines: list[str]) -> str | None:
    location = find_value_after_label(lines, "Местоположение:")

    if location:
        return location

    # Fallback: line after "в" in title-ish section is too fragile.
    return None


def parse_published_raw(lines: list[str]) -> str | None:
    return find_first_line_containing(lines, "Публикувана")


def parse_views(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if line == "Обявата е посетена" and i + 1 < len(lines):
            if lines[i + 1].isdigit():
                return int(lines[i + 1])

    return None


# ============================================================
# 8. ENGINE FUNCTION
# ============================================================

def parse_imot_listing(
    html_path: str,
    ad_url: str,
    include_raw_text_preview: bool = True,
) -> dict:
    """
    Main engine function.

    Input:
    - html_path: saved listing HTML file
    - ad_url: original listing URL

    Output:
    - one structured dictionary, ready for DataFrame
    """

    html = read_html_file(html_path)
    lines = html_to_lines(html)
    text = lines_to_text(lines)

    url_data = parse_ad_url(ad_url)

    title = lines[0] if lines else None
    title_data = parse_title_fields(title)

    price_data = parse_price(text)
    area_data = parse_area_sqm(text)
    floor_data = parse_floor_from_lines(lines)
    construction_data = parse_construction(text)
    feature_data = parse_features(lines)
    description_raw = parse_description(lines)
    description_clean = clean_multiline_text(description_raw)

    total_price = price_data["total_price"]
    area_sqm = area_data["area_sqm"]

    price_per_sqm = calculate_price_per_sqm(
        total_price=total_price,
        area_sqm=area_sqm,
    )

    listing_type = classify_listing(text)

    result = {
        **url_data,

        "html_path": html_path,

        "listing_type": listing_type,
        "title": title,
        **title_data,

        "total_price": total_price,
        "currency": price_data["currency"],
        "price_raw": price_data["price_raw"],
        "vat_status": parse_vat_status(text),

        "area_sqm": area_sqm,
        "area_raw": area_data["area_raw"],
        "price_per_sqm": price_per_sqm,

        "floor": floor_data["floor"],
        "total_floors": floor_data["total_floors"],
        "floor_raw": floor_data["floor_raw"],

        "construction_raw": construction_data["construction_raw"],
        "construction_type": construction_data["construction_type"],
        "construction_year": construction_data["construction_year"],

        "location_raw": parse_location(lines),

        "description_raw": description_raw,
        "description_clean": description_clean,

        "features_raw": feature_data["features_raw"],
        "features_pipe": feature_data["features_pipe"],
        "features_count": feature_data["features_count"],

        "published_raw": parse_published_raw(lines),
        "views": parse_views(lines),

        "training_eligible": (
            listing_type == "single_property_listing"
            and total_price is not None
            and area_sqm is not None
        ),
    }

    if include_raw_text_preview:
        result["raw_text_preview"] = text[:3000]

    return result