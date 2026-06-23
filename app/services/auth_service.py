from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.db.models import User, UserConsent

PRIVACY_VERSION = "1.0"
TERMS_VERSION = "1.0"

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username.strip()).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def authenticate(db: Session, login: str, password: str) -> Optional[User]:
    """Accept email or username as login identifier."""
    user = get_user_by_email(db, login) or get_user_by_username(db, login)
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(
    db: Session,
    *,
    email: str,
    username: str,
    password: str,
    full_name: str = "",
    role: str = "user",
) -> User:
    user = User(
        email=email.lower().strip(),
        username=username.strip(),
        hashed_password=hash_password(password),
        full_name=full_name.strip() or None,
        role=role,
        is_active=True,
    )
    db.add(user)
    db.flush()  # get user.id without committing
    return user


def record_consent(
    db: Session,
    user_id: int,
    consent_type: str,
    version: str,
    ip_address: Optional[str] = None,
) -> UserConsent:
    consent = UserConsent(
        user_id=user_id,
        consent_type=consent_type,
        accepted=True,
        version=version,
        ip_address=ip_address,
    )
    db.add(consent)
    return consent


def revoke_consent(db: Session, consent_id: int, user_id: int) -> bool:
    consent = db.query(UserConsent).filter(
        UserConsent.id == consent_id,
        UserConsent.user_id == user_id,
        UserConsent.revoked_at.is_(None),
    ).first()
    if not consent:
        return False
    consent.revoked_at = datetime.now(timezone.utc)
    return True


def get_active_consents(db: Session, user_id: int) -> dict:
    """Returns {consent_type: UserConsent} for the latest active consent of each type."""
    rows = (
        db.query(UserConsent)
        .filter(UserConsent.user_id == user_id, UserConsent.revoked_at.is_(None))
        .order_by(UserConsent.accepted_at.desc())
        .all()
    )
    seen = {}
    for c in rows:
        if c.consent_type not in seen:
            seen[c.consent_type] = c
    return seen


def update_last_login(db: Session, user: User) -> None:
    user.last_login_at = datetime.now(timezone.utc)
