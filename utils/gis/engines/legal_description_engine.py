"""
Generates a legal-document-ready Bulgarian description of a land plot (or,
when the subject is a specific building, of that building) — the
"при граници/съседи: ..." clause standard in Bulgarian notarial deeds and
appraisal reports, formatted per the ЗКИР-standard convention (numeral +
digit-spelled identifier, numeral + word-spelled area) used in the user's
own chsi-app project (github.com/Kamend1/chsi-app) — see
numwords_bg.py's docstring.

What AGKK's data can and can't give this text directly:
  - Parcel/building identifier, area, neighbour identifiers: yes, live,
    verified.
  - A human-readable settlement/place name for the parcel's admin unit:
    NO — the Administrative_Unit service's queryable layers (checked
    live) carry only numeric/coded identifiers, no Bulgarian place-name
    field. Pass `settlement_name` in from the appraisal report's own
    subject fields instead.
  - The official КККР approval order number (АГКК's "Заповед за
    одобрение на КККР ..." clause, standard in chsi-app's descriptions):
    NOT available from the free INSPIRE service — omitted rather than
    fabricated.
"""
from __future__ import annotations

from datetime import datetime, timezone

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.connectors.agkk_client import fetch_building_by_cadastral_id
import utils.gis.engines.numwords_bg as nw
from utils.gis.engines.parcel_engine import get_parcel_with_neighbours
from utils.gis.models.schemas import LegalDescription, NeighbourParcel


def _spelled(cadastral_id: str) -> str:
    return f"{cadastral_id} /{nw.spell_identifier(cadastral_id)}/"


def _area_clause(area_sqm: float | None) -> str:
    if area_sqm is None:
        return "неустановена по АГКК площ"
    return f"{nw.format_area(area_sqm)} кв.м /{nw.area_words(float(area_sqm))}/"


def _format_neighbour_list(neighbours: list[NeighbourParcel]) -> str:
    if not neighbours:
        return "съседни имоти не са установени по данни на АГКК"

    parts = [f"имот с идентификатор {_spelled(n.cadastral_id)}" for n in neighbours]
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts[:-1]) + " и " + parts[-1]


def generate_legal_description(
    cadastral_id: str,
    settlement_name: str | None = None,
    cache: GisCache | None = None,
) -> LegalDescription:
    """Builds the LegalDescription for a *parcel* — use
    generate_building_description() instead when the subject is a
    specific building. `settlement_name` is optional free text (e.g.
    "гр. София, район Триадица, кв. Лозенец") — supply it from the
    appraisal report's own subject fields; AGKK's own admin-unit code is
    kept in the structured output regardless, but is not resolved to a
    name here (see module docstring)."""
    parcel, neighbours = get_parcel_with_neighbours(cadastral_id, cache=cache)

    settlement_clause = f", находящ се в {settlement_name}" if settlement_name else ""
    neighbours_clause = _format_neighbour_list(neighbours)

    text_bg = (
        f"Поземлен имот с идентификатор {_spelled(parcel.cadastral_id)} по кадастралната карта "
        f"и кадастралните регистри на Агенцията по геодезия, картография и кадастър "
        f"(АГКК){settlement_clause}, целият с площ {_area_clause(parcel.area_sqm)}, "
        f"при граници (съседи): {neighbours_clause}."
    )

    return LegalDescription(
        cadastral_id=parcel.cadastral_id,
        area_sqm=parcel.area_sqm,
        admin_unit_code=parcel.admin_unit_code,
        neighbours=neighbours,
        text_bg=text_bg,
        generated_at=datetime.now(timezone.utc),
        sources=[parcel.source],
    )


def generate_building_description(
    building_cadastral_id: str,
    parcel_id: str,
    settlement_name: str | None = None,
    cache: GisCache | None = None,
) -> LegalDescription:
    """Builds a building-led LegalDescription: "СГРАДА ... разположена в
    поземлен имот с идентификатор ..." — matches chsi-app's
    describe_building() template. Use this instead of
    generate_legal_description() whenever the user has pointed at a
    specific building (a 4-segment cadastral id), since a building sitting
    on a plot is legally a different subject than the plot itself."""
    building = fetch_building_by_cadastral_id(building_cadastral_id, cache=cache)

    settlement_clause = f", с адрес: {settlement_name}" if settlement_name else ""
    floors_clause = (
        f"{building.floors_above_ground} /{nw.integer_words(building.floors_above_ground)}/"
        if building.floors_above_ground is not None
        else "няма данни"
    )
    use_clause = building.current_use or "няма данни"

    text_bg = (
        f"СГРАДА с идентификатор {_spelled(building.cadastral_id)} "
        f"по кадастралната карта и кадастралните регистри на Агенцията по геодезия, "
        f"картография и кадастър (АГКК){settlement_clause}, "
        f"с предназначение: {use_clause}, "
        f"брой надземни етажи: {floors_clause}, "
        f"застроена площ: {_area_clause(building.area_sqm)}, "
        f"която сграда е разположена в поземлен имот с идентификатор {_spelled(parcel_id)}."
    )

    return LegalDescription(
        cadastral_id=building.cadastral_id,
        area_sqm=building.area_sqm,
        admin_unit_code=None,
        neighbours=[],
        text_bg=text_bg,
        generated_at=datetime.now(timezone.utc),
        sources=[building.source],
    )
