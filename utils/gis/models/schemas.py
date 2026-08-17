"""
Pydantic models for the cadastral/zoning/legal-description pipeline.

CadastralIdentifier validates the Bulgarian ПИ (поземлен имот) identifier
format: EKATTE.masiv.imot[.building_no], e.g. "68134.1234.567" for a
parcel or "15285.13.286.2" for a building on that parcel — both formats
verified live against AGKK's INSPIRE service during development.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CADASTRAL_ID_PATTERN = re.compile(r"^\d{5}\.\d+\.\d+(\.\d+)?$")


class CadastralIdentifier(BaseModel):
    """EKATTE.masiv.imot[.building_unit] — the optional 4th segment
    identifies a building sitting on the parcel (AGKK's own convention,
    confirmed live: building "15285.13.286.2" sits on parcel
    "15285.13.286"), not an individual apartment/СОС — see
    engines/building_engine.py's module docstring for that distinction."""

    raw: str

    @field_validator("raw")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not CADASTRAL_ID_PATTERN.match(v):
            raise ValueError(
                f"'{v}' is not a valid cadastral identifier "
                "(expected EKATTE.masiv.imot, e.g. 68134.1234.567)"
            )
        return v

    @property
    def ekatte(self) -> str:
        return self.raw.split(".")[0]

    @property
    def masiv(self) -> str:
        return self.raw.split(".")[1]

    @property
    def imot(self) -> str:
        return self.raw.split(".")[2]

    @property
    def parcel_id(self) -> str:
        """The 3-segment parcel id, even when `raw` is a 4-segment building id."""
        return ".".join(self.raw.split(".")[:3])


class Coordinates(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    crs: str = "EPSG:4326"


class SourceMeta(BaseModel):
    name: str
    endpoint: str
    queried_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["ok", "cached", "not_found", "error"] = "ok"
    detail: str | None = None


class ParcelGeometry(BaseModel):
    cadastral_id: str
    area_sqm: float | None = None
    admin_unit_code: str | None = None
    centroid: Coordinates
    geometry_geojson: dict
    native_crs: str = "EPSG:4258"
    source: SourceMeta


class BuildingInfo(BaseModel):
    """From AGKK's BU.Building layer. Note area_sqm here is *derived*
    (reprojected + computed via spatial_engine), unlike ParcelGeometry's,
    because the Building layer has no ready-made area attribute — see
    engines/building_engine.py."""

    cadastral_id: str
    parcel_id: str
    area_sqm: float | None = None
    floors_above_ground: int | None = None
    dwellings: int | None = None
    building_units: int | None = None
    current_use: str | None = None
    building_nature: str | None = None
    condition: str | None = None
    construction_date: str | None = None
    centroid: Coordinates | None = None
    source: SourceMeta


class NeighbourParcel(BaseModel):
    cadastral_id: str
    label: str | None = None
    area_sqm: float | None = None


class ZoningInfo(BaseModel):
    zone_code: str | None = None
    zone_description: str | None = None
    max_density_pct: float | None = Field(None, description="Плътност на застрояване, %")
    max_kint: float | None = Field(None, description="Кинт — интензивност на застрояване")
    max_height_m: float | None = Field(None, description="Максимална височина, м")
    min_landscaping_pct: float | None = Field(None, description="Минимално озеленяване, %")
    plan_name: str | None = None
    confidence: Literal["exact_match", "nearest_zone", "not_found", "unavailable"] = "unavailable"
    source: SourceMeta | None = None


class LegalDescription(BaseModel):
    """The generated legal-document-ready description of a land plot,
    including its boundary neighbours — see engines/legal_description_engine.py."""

    cadastral_id: str
    area_sqm: float | None
    admin_unit_code: str | None
    neighbours: list[NeighbourParcel]
    text_bg: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sources: list[SourceMeta]


class DevelopmentPlanDocument(BaseModel):
    """A single attachment on a NAG Sofia development-plan case (Скица –
    предложение, Заявление, ИПР, ИПРЗ, РУП, ...). `url` is a real,
    downloadable ViewAttachment link — confirmed live to serve the actual
    PDF."""

    name: str
    url: str


class DevelopmentPlanRecord(BaseModel):
    """One case from NAG Sofia's development-plan (ПУП) case registry —
    see connectors/nag_sofia_client.py. This is NOT a zoning-parameters
    lookup (no Kint/density/height here) — it's "has there been recent
    planning activity referencing this cadastral identifier," with links
    to the actual case documents."""

    reference: str  # e.g. "САГ26-ГР00-1696/22.07.2026 г. - Заявление за ..."
    office: str | None = None
    scope_text: str  # district/quarter/УПИ/cadastral-id text, as published
    procedure_type: str | None = None
    documents: list[DevelopmentPlanDocument] = Field(default_factory=list)


class ParcelReport(BaseModel):
    cadastral_id: str
    area_sqm: float | None
    centroid: Coordinates
    buildings: list[BuildingInfo] = Field(default_factory=list)
    zoning: ZoningInfo
    legal_description: LegalDescription | None = None
    sources: list[SourceMeta]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
