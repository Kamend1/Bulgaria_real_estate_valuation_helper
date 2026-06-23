from typing import Optional

from fastapi import HTTPException, Request

from app.db.models import User


def get_current_user(request: Request) -> Optional[User]:
    """Returns the authenticated user or None (does not enforce auth)."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> User:
    """FastAPI dependency — raises 401 if not authenticated."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Необходима е идентификация")
    return user


def current_user_is_admin(request: Request) -> bool:
    user = get_current_user(request)
    return user is not None and user.role == "admin"
