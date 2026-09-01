"""
Integration tests for app/services/avm_retrain_service.py (Phase 14 Tier
2.1) -- real DB for AvmModel rows (rolled back per test, see conftest.py),
mocked subprocess.run (never actually spawn scripts.train_avm_model) and
mocked _count_eligible_rows (deterministic, independent of the real
corpus's current row counts).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.db.models import AvmModel
from app.services import avm_retrain_service


@pytest.fixture(autouse=True)
def _maintainer_key_set(monkeypatch):
    monkeypatch.setattr(avm_retrain_service.settings, "r2_maintainer_access_key_id", "test-key")
    monkeypatch.setattr(avm_retrain_service.settings, "avm_auto_retrain_growth_pct", 20.0)


@pytest.fixture(autouse=True)
def _never_spawn_a_real_training_subprocess(monkeypatch):
    # Safety net, not just a convenience: the real DB (see conftest.py's
    # rollback-based db_session) already has real, committed AvmModel rows
    # for every segment from actual training runs earlier in this project.
    # A test that mocks _count_eligible_rows to a large number without ALSO
    # mocking subprocess.run would launch a real `python -m
    # scripts.train_avm_model --push-to-r2` against production data --
    # exactly what happened once while writing this test file. Default to
    # a harmless success here; individual tests override via patch.object
    # when they need to assert on the call or simulate a failure.
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(avm_retrain_service.subprocess, "run", MagicMock(return_value=fake_proc))


def _make_active_model(db_session, segment: str, training_row_count: int) -> AvmModel:
    # A partial unique index (uq_avm_models_active_per_segment) enforces at
    # most one is_active=True row per segment -- and the real ambient DB
    # already has one for every segment (real past training runs). Deactivate
    # it within this test's own rolled-back transaction first.
    db_session.query(AvmModel).filter(
        AvmModel.segment == segment, AvmModel.is_active.is_(True)
    ).update({"is_active": False})
    m = AvmModel(
        segment=segment, algorithm="lightgbm", feature_columns=[], hyperparams={},
        training_row_count=training_row_count, min_row_threshold=300,
        model_path="s3://fake/model.joblib", is_active=True,
    )
    db_session.add(m)
    db_session.commit()
    return m


def test_skips_entirely_when_no_maintainer_key(db_session, monkeypatch):
    monkeypatch.setattr(avm_retrain_service.settings, "r2_maintainer_access_key_id", "")
    results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    assert results == []


def test_skips_segment_with_no_active_model(db_session):
    # Every segment already has a real, committed active AvmModel in the
    # ambient DB from actual past training runs -- deactivate residential's
    # within this test's own (rolled-back) transaction so "no active model"
    # is genuinely true for it here, without touching real data.
    db_session.query(AvmModel).filter(
        AvmModel.segment == "residential", AvmModel.is_active.is_(True)
    ).update({"is_active": False})
    db_session.commit()

    with patch.object(avm_retrain_service, "_count_eligible_rows", return_value=10_000):
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    residential = next(r for r in results if r["segment"] == "residential")
    assert residential["action"] == "skipped"
    assert "няма активен модел" in residential["detail"]


def test_skips_segment_below_growth_threshold(db_session):
    _make_active_model(db_session, "residential", training_row_count=1000)
    with patch.object(avm_retrain_service, "_count_eligible_rows", return_value=1100):  # +10%, below 20% threshold
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    residential = next(r for r in results if r["segment"] == "residential")
    assert residential["action"] == "skipped"


def test_retrains_segment_above_growth_threshold(db_session):
    _make_active_model(db_session, "residential", training_row_count=1000)
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(avm_retrain_service, "_count_eligible_rows", return_value=1300), \
         patch.object(avm_retrain_service.subprocess, "run", return_value=fake_proc) as mock_run:
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    residential = next(r for r in results if r["segment"] == "residential")
    assert residential["action"] == "retrained"
    args = mock_run.call_args.args[0]
    assert "--segment" in args and "residential" in args
    assert "--push-to-r2" in args


def test_reports_failed_when_subprocess_returns_nonzero(db_session):
    _make_active_model(db_session, "residential", training_row_count=1000)
    fake_proc = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(avm_retrain_service, "_count_eligible_rows", return_value=1300), \
         patch.object(avm_retrain_service.subprocess, "run", return_value=fake_proc):
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    residential = next(r for r in results if r["segment"] == "residential")
    assert residential["action"] == "failed"
    assert "boom" in residential["detail"]
