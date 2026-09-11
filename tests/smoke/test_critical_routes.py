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


def test_home_page_stats_exclude_archived_listings(client):
    """Regression test (2026-09-11): app/main.py's home() used to count
    `FROM listings` with no status filter, so archived (sold/removed)
    listings inflated the "Обяви за продажба"/"Обяви за наем" stat cards --
    caught when a user noticed the homepage total exactly matched
    active+archived, not the real active count. Uses a disposable archived
    row against the real DB (home() opens its own session, bypassing the
    test's dependency override) -- cleaned up in a finally block."""
    from sqlalchemy import text as sa_text
    from app.db.session import db_session

    test_url = "https://test.invalid/regression-home-stats-archived-1"
    with db_session() as db:
        before_sale_count = db.execute(
            sa_text("SELECT count(*) FROM listings WHERE status = 'active' AND deal_type_normalized = 'sale'")
        ).scalar()

    try:
        with db_session() as db:
            db.execute(sa_text("""
                INSERT INTO listings (ad_url, deal_type_normalized, status, price_per_sqm_model)
                VALUES (:url, 'sale', 'archived', 999999)
            """), {"url": test_url})

        html = client.get("/").text
        m = re.search(r'stat-value">([\d\s]+)</div>\s*<div class="stat-label">Обяви за продажба', html)
        assert m, "sale_count stat card not found on home page"
        shown_sale_count = int(m.group(1).replace(" ", "").replace("\xa0", ""))
        assert shown_sale_count == before_sale_count
    finally:
        with db_session() as db:
            db.execute(sa_text("DELETE FROM listings WHERE ad_url = :url"), {"url": test_url})


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
