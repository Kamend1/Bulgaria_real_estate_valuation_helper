"""
Smoke tests for critical unauthenticated routes -- a safety net against
regressions, not full coverage. Uses the real `client` TestClient fixture
(see tests/conftest.py). Note: app/main.py's home() route and the
attach_current_user middleware both open their own DB session directly
(not via the get_db FastAPI dependency), so they always hit the real DB
regardless of the test's dependency override -- fine for these read-only
checks, but it means authenticated flows aren't exercised here (would
require a real, committed user).
"""
import re

import pytest


def _extract_csrf_token(html: str) -> str:
    m = re.search(r'name="csrf-token" content="([^"]+)"', html)
    assert m, "csrf-token meta tag not found in response HTML"
    return m.group(1)


def test_home_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "csrf-token" in resp.text


def test_login_page_renders(client):
    resp = client.get("/auth/login")
    assert resp.status_code == 200
    assert "csrf-token" in resp.text


def test_register_page_renders(client):
    resp = client.get("/auth/register")
    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/help", "/legal/privacy-policy", "/legal/terms"])
def test_static_info_pages_render(client, path):
    resp = client.get(path)
    assert resp.status_code == 200


def test_login_post_with_bad_credentials_returns_401_not_csrf_rejection(client):
    csrf_token = _extract_csrf_token(client.get("/auth/login").text)
    resp = client.post(
        "/auth/login",
        data={"login": "nonexistent@example.com", "password": "wrong-password"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 401
    assert "Невалиден" in resp.text


def test_login_post_without_csrf_token_is_rejected(client):
    resp = client.post(
        "/auth/login",
        data={"login": "nonexistent@example.com", "password": "wrong-password"},
    )
    assert resp.status_code == 403


def test_comparables_page_redirects_unauthenticated_user_to_login(client):
    resp = client.get("/comparables/", follow_redirects=False)
    assert resp.status_code in (302, 401)
