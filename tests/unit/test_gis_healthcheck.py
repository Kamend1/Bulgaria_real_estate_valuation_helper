"""
Unit tests for utils/gis/healthcheck.py (Phase 14 Tier 1.2) -- mocked HTTP
only, no real network calls to the government GIS endpoints.
"""
import requests

from utils.gis import healthcheck


def test_check_gis_connectors_reports_ok_on_2xx(monkeypatch):
    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(healthcheck.requests, "get", lambda *a, **k: FakeResponse())
    results = healthcheck.check_gis_connectors()
    assert len(results) == 3
    for r in results:
        assert r["ok"] is True
        assert r["status_code"] == 200
        assert r["error"] is None


def test_check_gis_connectors_reports_down_on_5xx(monkeypatch):
    class FakeResponse:
        status_code = 503

    monkeypatch.setattr(healthcheck.requests, "get", lambda *a, **k: FakeResponse())
    results = healthcheck.check_gis_connectors()
    for r in results:
        assert r["ok"] is False
        assert r["status_code"] == 503


def test_check_gis_connectors_survives_connection_error(monkeypatch):
    def raise_error(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(healthcheck.requests, "get", raise_error)
    results = healthcheck.check_gis_connectors()
    assert len(results) == 3
    for r in results:
        assert r["ok"] is False
        assert r["status_code"] is None
        assert "no route to host" in r["error"]


def test_check_gis_connectors_handles_partial_failure(monkeypatch):
    calls = {"n": 0}

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise requests.Timeout("timed out")
        class FakeResponse:
            status_code = 200
        return FakeResponse()

    monkeypatch.setattr(healthcheck.requests, "get", flaky_get)
    results = healthcheck.check_gis_connectors()
    assert [r["ok"] for r in results] == [True, False, True]
