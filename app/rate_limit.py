"""
Shared slowapi Limiter instance. Its own module (not app/main.py) so
app/routers/auth.py can import it without a circular import — main.py
imports the auth router, so the router can't import back from main.py.
"""
from slowapi import Limiter
from starlette.requests import Request


def _client_ip(request: Request) -> str:
    """Same X-Forwarded-For-aware logic as auth.py's own helper — kept as
    a standalone copy here (not imported from auth.py) to avoid a circular
    import in the other direction (auth.py imports `limiter` from here)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip)
