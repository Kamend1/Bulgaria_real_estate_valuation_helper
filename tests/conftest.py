"""
Shared pytest fixtures.

DB tests run against the real `appraisal` database (no separate test DB —
the `appraiser` role has no CREATEDB privilege, and setting one up would
need elevated credentials nobody's granted). Safety comes from wrapping
every test in a transaction that's always rolled back at teardown, even
if the code under test calls session.commit() — see `db_session` below.
This is SQLAlchemy's own documented pattern for exactly this situation
("Joining a Session into an External Transaction (such as for test
suites)"), not a improvised workaround.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.base import engine
from app.db.session import get_db


@pytest.fixture
def db_session():
    connection = engine.connect()
    trans = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with the get_db dependency overridden to use the
    same rolled-back session, so smoke tests hitting real routes don't
    touch the real database either."""
    from app.main import app

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
