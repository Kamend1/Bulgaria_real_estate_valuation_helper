"""
Unit tests for the two new specialist tools added in Phase 15 Tier 3
(2026-09-15): orchestrator_graph._get_zoning_info_fn (backs the new
"zoning" specialist) and _get_pool_quality_detail_fn (backs the new
"comp_quality" specialist). Both are pure data-shaping wrappers around
existing services (gis_service.get_cadastre_panel_data,
comparable_service.get_pool_with_stats) -- these tests mock those services
directly rather than hitting AGKK/isofmap.bg or a real DB, mirroring
tests/integration/test_legal_document_search.py's mocked-external pattern
(the closest existing analog for a new specialist tool, per the Tier 3
research pass).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.llm import orchestrator_graph as og


class TestGetZoningInfoFn:
    def test_unavailable_reason_surfaced_when_lookup_fails(self):
        report = SimpleNamespace(subject_cadastral_id="")
        with patch("app.services.gis_service.get_cadastre_panel_data", return_value={"ok": False, "reason": "missing_cadastral_id"}):
            tool = og._get_zoning_info_fn(report)
            result = tool()
        assert result == {"available": False, "reason": "missing_cadastral_id", "detail": None}

    def test_success_shapes_zoning_and_parcel_fields(self):
        report = SimpleNamespace(subject_cadastral_id="68134.1000.100")
        zoning = SimpleNamespace(
            confidence="exact_match", zone_code="Жм", zone_description="Жилищна зона средно застрояване",
            max_density_pct=60.0, max_kint=1.2, max_height_m=15.0, min_landscaping_pct=30.0,
            plan_name="ОУП София",
        )
        parcel = SimpleNamespace(cadastral_id="68134.1000.100", area_sqm=450.0)
        plan = SimpleNamespace(reference="САГ26-ГР00-1", scope_text="кв. Оборище", procedure_type="ИПРЗ")
        data = {
            "ok": True, "zoning": zoning, "parcel": parcel, "total_building_area_sqm": 320.0,
            "development_plans": [plan],
        }
        with patch("app.services.gis_service.get_cadastre_panel_data", return_value=data):
            tool = og._get_zoning_info_fn(report)
            result = tool()
        assert result["available"] is True
        assert result["zoning"]["zone_code"] == "Жм"
        assert result["zoning"]["max_kint"] == 1.2
        assert result["parcel"]["cadastral_id"] == "68134.1000.100"
        assert result["development_plans"] == [{"reference": "САГ26-ГР00-1", "scope_text": "кв. Оборище", "procedure_type": "ИПРЗ"}]

    def test_success_with_no_zoning_or_parcel_does_not_crash(self):
        report = SimpleNamespace(subject_cadastral_id="68134.1000.100")
        data = {"ok": True, "zoning": None, "parcel": None, "total_building_area_sqm": None, "development_plans": []}
        with patch("app.services.gis_service.get_cadastre_panel_data", return_value=data):
            tool = og._get_zoning_info_fn(report)
            result = tool()
        assert result["available"] is True
        assert result["zoning"] is None
        assert result["parcel"] is None


class TestGetPoolQualityDetailFn:
    def _fake_pool(self, rows):
        return {"rows": rows, "stats": {"median": 1200}, "pinned_count": sum(1 for r in rows if r["pinned_for_report"]), "total_count": len(rows)}

    def test_only_pinned_comparables_are_returned(self, monkeypatch):
        rows = [
            {"listing_id": 1, "pinned_for_report": True, "title_city_model": "софия", "title_geo_2_model": "център",
             "area_sqm_model": 65, "price_per_sqm_model": 1300, "adjustment_pct": -5, "adjustment_factors": {"size": -5},
             "adj_ppsqm": 1235, "floor_model": 2, "construction_type_model": "Тухла", "analyst_note": None},
            {"listing_id": 2, "pinned_for_report": False, "title_city_model": "софия", "title_geo_2_model": "център",
             "area_sqm_model": 500, "price_per_sqm_model": 900, "adjustment_pct": None, "adjustment_factors": None,
             "adj_ppsqm": None, "floor_model": None, "construction_type_model": None, "analyst_note": None},
        ]
        monkeypatch.setattr(og, "get_pool_with_stats", lambda db, ctype, rid: self._fake_pool(rows))
        report = SimpleNamespace(id="r1")
        tool = og._get_pool_quality_detail_fn(MagicMock(), report)
        result = tool(comparable_type="sale")
        assert len(result["pinned_comparables"]) == 1
        assert result["pinned_comparables"][0]["listing_id"] == 1
        assert result["pinned_comparables"][0]["adjustment_factors"] == {"size": -5}
        assert result["pinned_count"] == 1

    def test_empty_pool_returns_empty_list_not_error(self, monkeypatch):
        monkeypatch.setattr(og, "get_pool_with_stats", lambda db, ctype, rid: self._fake_pool([]))
        report = SimpleNamespace(id="r1")
        tool = og._get_pool_quality_detail_fn(MagicMock(), report)
        result = tool(comparable_type="rent")
        assert result["pinned_comparables"] == []
