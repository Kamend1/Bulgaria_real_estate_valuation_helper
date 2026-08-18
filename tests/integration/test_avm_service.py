"""
Integration tests for app/services/avm_service.py's graceful-degradation
paths -- the ones that don't need a real trained joblib pipeline on disk,
just DB state (or lack of it). Run against the real `appraisal` DB inside a
rolled-back transaction (see tests/conftest.py::db_session).
"""
import pytest

from app.db.models import AppraisalReport, AvmModel
from app.services import avm_service


def _make_report(db_session, **overrides) -> AppraisalReport:
    report = AppraisalReport(title="Тест доклад", status="draft", **overrides)
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    return report


def test_predict_sales_value_reports_missing_fields(db_session):
    report = _make_report(db_session)  # no subject_* fields set at all
    result = avm_service.predict_sales_value(db_session, report)

    assert result["ok"] is False
    assert result["reason"] == "missing_fields"
    assert set(result["missing_fields"]) == {
        "subject_area_sqm", "subject_city", "subject_property_type", "subject_geo_category",
    }


def test_predict_sales_value_reports_unsupported_property_type(db_session):
    report = _make_report(
        db_session,
        subject_area_sqm=80.0,
        subject_city="софия",
        subject_geo_category="sofia_other",
        subject_property_type="partsel",  # land -- intentionally excluded from all segments
    )
    result = avm_service.predict_sales_value(db_session, report)

    assert result["ok"] is False
    assert result["reason"] == "unsupported_property_type"


def _deactivate_all_models(db_session, segment: str) -> None:
    # The real DB already has an active model per segment from earlier live
    # training runs -- deactivate them within this test's (always
    # rolled-back) transaction so the "no active model" path is genuinely
    # exercised without touching real, already-committed data.
    db_session.query(AvmModel).filter_by(segment=segment).update({"is_active": False})
    db_session.commit()


def test_predict_sales_value_reports_no_model_when_segment_has_no_active_model(db_session):
    _deactivate_all_models(db_session, "hospitality")
    report = _make_report(
        db_session,
        subject_area_sqm=500.0,
        subject_city="софия",
        subject_geo_category="sofia_other",
        subject_property_type="hotel",  # resolves to "hospitality" segment
    )
    result = avm_service.predict_sales_value(db_session, report)

    assert result["ok"] is False
    assert result["reason"] == "no_model"
    assert result["segment"] == "hospitality"


def test_get_active_model_meta_returns_none_when_no_row_exists(db_session):
    _deactivate_all_models(db_session, "hospitality")
    assert avm_service.get_active_model_meta(db_session, "hospitality") is None


def test_predict_sales_value_reports_model_fetch_failed_when_r2_unreachable(db_session, monkeypatch):
    # Simulates a real active model row (R2-key-style model_path, per Phase
    # 5) whose R2 fetch fails -- e.g. missing/bad credentials, R2 outage.
    # Should degrade to a distinct "model_fetch_failed" reason, not crash
    # and not get lumped in with a genuine prediction-error.
    _deactivate_all_models(db_session, "hospitality")
    db_session.add(AvmModel(
        segment="hospitality",
        algorithm="lightgbm",
        feature_columns=["area_sqm"],
        hyperparams={},
        target_transform="raw",
        training_row_count=1000,
        min_row_threshold=300,
        model_path="avm-models/hospitality/faketest/model.joblib",
        quantile_low_path="avm-models/hospitality/faketest/q_low.joblib",
        quantile_high_path="avm-models/hospitality/faketest/q_high.joblib",
        is_active=True,
    ))
    db_session.commit()

    def _raise_fetch_error():
        raise RuntimeError("R2 credentials not configured")

    monkeypatch.setattr(avm_service.r2_client, "get_models_read_client", _raise_fetch_error)

    report = _make_report(
        db_session,
        subject_area_sqm=500.0,
        subject_city="софия",
        subject_geo_category="sofia_other",
        subject_property_type="hotel",
    )
    result = avm_service.predict_sales_value(db_session, report)

    assert result["ok"] is False
    assert result["reason"] == "model_fetch_failed"
    assert result["segment"] == "hospitality"
