"""
CSRF protection — synchronizer token pattern tied to the session cookie
(SessionMiddleware). No third-party dependency; the app already has a
signed session, so a token stored there and compared against what the
client submits back is all this needs.

Token delivery to the client happens two ways, both driven by
`get_or_create_csrf_token`:
  - `app/templates/base.html` puts it in a <meta> tag and sets it as an
    `hx-headers` value on <body> — covers every htmx request app-wide,
    including into content swapped in after the initial page load (htmx
    resolves `hx-headers` inheritance per-request, not at DOM-load time,
    so this isn't defeated by HTMX partial swaps).
  - A small script in base.html injects a hidden `csrf_token` input into
    every plain <form> at load time — covers classic full-page POSTs,
    which have no JS/htmx involved to attach a header.

Enforcement is centralized in one middleware (see app/main.py's
`csrf_protect`) rather than per-route, so a new POST route can't
accidentally ship without protection.
"""
from __future__ import annotations

import secrets

from starlette.requests import Request

SESSION_KEY = "csrf_token"


def get_or_create_csrf_token(request: Request) -> str:
    token = request.session.get(SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[SESSION_KEY] = token
    return token


def verify_csrf_token(request: Request, submitted: str | None) -> bool:
    expected = request.session.get(SESSION_KEY)
    if not expected or not submitted:
        return False
    return secrets.compare_digest(expected, submitted)
