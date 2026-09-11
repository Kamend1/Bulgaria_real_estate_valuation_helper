"""
Integration tests for app/services/avm_retrain_service.py -- retrains
unconditionally on every successful scrape (2026-09-11, following a real
incident where the previous growth-percentage gate silently skipped every
segment because a single scrape's growth never came close to the
threshold). subprocess.run is always mocked (never actually spawn
scripts.train_avm_model against the real corpus).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.services import avm_retrain_service
from utils.ml.avm_features import SEGMENT_PROPERTY_TYPES


@pytest.fixture(autouse=True)
def _maintainer_key_set(monkeypatch):
    monkeypatch.setattr(avm_retrain_service.settings, "r2_maintainer_access_key_id", "test-key")


@pytest.fixture(autouse=True)
def _never_spawn_a_real_training_subprocess(monkeypatch):
    # Safety net, not just a convenience -- see git history for the real
    # incident this guards against (an unmocked subprocess.run launched a
    # genuine `train_avm_model --push-to-r2` against production data while
    # writing an earlier version of this test file).
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(avm_retrain_service.subprocess, "run", MagicMock(return_value=fake_proc))


def test_skips_entirely_when_no_maintainer_key(db_session, monkeypatch):
    monkeypatch.setattr(avm_retrain_service.settings, "r2_maintainer_access_key_id", "")
    results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    assert results == []


def test_retrains_every_segment_when_maintainer_key_present(db_session):
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(avm_retrain_service.subprocess, "run", return_value=fake_proc) as mock_run:
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)

    assert {r["segment"] for r in results} == set(SEGMENT_PROPERTY_TYPES)
    assert all(r["action"] == "retrained" for r in results)
    assert mock_run.call_count == len(SEGMENT_PROPERTY_TYPES)
    for call in mock_run.call_args_list:
        args = call.args[0]
        assert "--push-to-r2" in args
        assert "--segment" in args


def test_reports_skipped_when_train_script_prints_skip_notice(db_session):
    # train_avm_model.py itself prints "SKIPPED" and exits 0 when a segment
    # is below --min-rows -- that must surface as "skipped", not "retrained".
    fake_proc = MagicMock(returncode=0, stdout="  SKIPPED — below minimum row threshold, no model trained.", stderr="")
    with patch.object(avm_retrain_service.subprocess, "run", return_value=fake_proc):
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    assert all(r["action"] == "skipped" for r in results)


def test_reports_failed_when_subprocess_returns_nonzero(db_session):
    fake_proc = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(avm_retrain_service.subprocess, "run", return_value=fake_proc):
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)
    assert all(r["action"] == "failed" for r in results)
    assert all("boom" in r["detail"] for r in results)


def test_one_segment_failing_does_not_stop_the_others(db_session):
    calls = {"n": 0}

    def fake_run(args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return MagicMock(returncode=1, stdout="", stderr="network error")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(avm_retrain_service.subprocess, "run", side_effect=fake_run):
        results = avm_retrain_service.maybe_retrain_avm_models(db_session)

    assert len(results) == len(SEGMENT_PROPERTY_TYPES)
    actions = [r["action"] for r in results]
    assert "failed" in actions
    assert "retrained" in actions
