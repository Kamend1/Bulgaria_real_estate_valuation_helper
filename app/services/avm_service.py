"""
Serving layer for the segment-aware AVM. Loads the active avm_models
pipelines per segment (cached in-process) and predicts a sales-approach
value for an AppraisalReport's subject property.
"""
from __future__ import annotations

import io
import logging
import uuid

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AppraisalReport, AvmModel
from app.services import r2_client
from utils.ml import avm_features, text_features

logger = logging.getLogger(__name__)

# Sanity-guard band: a prediction more than 3x (or less than 0.3x) the
# cohort median for the same property type + geo category is clamped rather
# than trusted outright — guards against wild extrapolation on subject
# properties that sit far outside the training distribution.
_CLAMP_LOW_MULT = 0.3
_CLAMP_HIGH_MULT = 3.0

_PIPELINE_CACHE: dict[uuid.UUID, dict] = {}


def get_active_model_meta(db: Session, segment: str) -> AvmModel | None:
    return (
        db.query(AvmModel)
        .filter_by(segment=segment, is_active=True)
        .first()
    )


def _load_joblib_from_r2(client, key: str):
    """Fetches an object from R2 straight into memory and unpickles it --
    no local file is ever written. `key` is one of avm_models' *_path
    columns, which store R2 object keys (e.g.
    "avm-models/hospitality/20260818_120000/model.joblib"), not local
    filesystem paths."""
    obj = client.get_object(Bucket=settings.r2_models_bucket_name, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    return joblib.load(buf)


def _load_pipelines(meta: AvmModel) -> dict:
    """Fetches all pipeline artifacts for one AvmModel from R2, caching the
    result in-process (_PIPELINE_CACHE) so each active model is only ever
    fetched once per process lifetime, not once per prediction request."""
    cached = _PIPELINE_CACHE.get(meta.id)
    if cached is not None:
        return cached

    client = r2_client.get_models_read_client()
    pipelines = {
        "point": _load_joblib_from_r2(client, meta.model_path),
        "q_low": _load_joblib_from_r2(client, meta.quantile_low_path),
        "q_high": _load_joblib_from_r2(client, meta.quantile_high_path),
    }
    if meta.companion_model_path:
        pipelines["companion_point"] = _load_joblib_from_r2(client, meta.companion_model_path)
        pipelines["companion_q_low"] = _load_joblib_from_r2(client, meta.companion_quantile_low_path)
        pipelines["companion_q_high"] = _load_joblib_from_r2(client, meta.companion_quantile_high_path)
    if meta.text_transformer_path:
        pipelines["text_transformer"] = _load_joblib_from_r2(client, meta.text_transformer_path)
    _PIPELINE_CACHE[meta.id] = pipelines
    return pipelines


def _predict_one_catboost(model, row, use_log: bool) -> float:
    """Same as _predict_one but for a raw CatBoostRegressor (no sklearn
    Pipeline wrapper) — needs categoricals fillna'd first, see
    utils.ml.avm_features.prep_for_catboost."""
    cb_row = avm_features.prep_for_catboost(row)
    pred = float(model.predict(cb_row)[0])
    if use_log:
        pred = float(np.expm1(pred))
    return max(pred, 1.0)


def _predict_one(pipeline, row, use_log: bool) -> float:
    """Predicts a single row and inverse-transforms back to raw EUR/sqm if
    the pipeline was trained on log1p(target) — see AvmModel.target_transform."""
    pred = float(pipeline.predict(row)[0])
    if use_log:
        pred = float(np.expm1(pred))
    return max(pred, 1.0)


def _cohort_median_ppsqm(db: Session, property_type_slug: str, geo_category: str) -> float | None:
    value = db.execute(text("""
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_sqm_model)
        FROM listings
        WHERE training_eligible = TRUE
          AND deal_type_normalized = 'sale'
          AND property_type_slug = :slug
          AND geo_category = :geo
          AND price_per_sqm_model IS NOT NULL
    """), {"slug": property_type_slug, "geo": geo_category}).scalar()
    return float(value) if value is not None else None


def predict_sales_value(db: Session, report: AppraisalReport) -> dict:
    """
    Returns a dict always containing "ok". On success: ppsqm_point/low/high,
    total_point/low/high, model_trained_at, metrics, clamped. On failure:
    "reason" is one of "missing_fields" | "unsupported_property_type" |
    "no_model" | "model_fetch_failed" | "prediction_error".
    """
    missing = avm_features.missing_subject_fields(report)
    if missing:
        return {"ok": False, "reason": "missing_fields", "missing_fields": missing}

    segment = avm_features.segment_for_property_type(report.subject_property_type)
    if segment is None:
        return {"ok": False, "reason": "unsupported_property_type"}

    meta = get_active_model_meta(db, segment)
    if meta is None:
        return {"ok": False, "reason": "no_model", "segment": segment}

    try:
        structured_cols = [c for c in meta.feature_columns if not c.startswith(text_features.TEXT_FEATURE_PREFIX)]
        row = avm_features.build_feature_row(db, report, structured_cols)
    except ValueError:
        return {"ok": False, "reason": "missing_fields", "missing_fields": missing}

    try:
        pipelines = _load_pipelines(meta)
    except Exception:
        # Separate from the prediction try-block below on purpose -- an R2
        # outage / bad credentials / missing object should surface as a
        # distinct, explainable degradation ("model temporarily
        # unavailable"), not get lumped in with a genuine inference bug.
        logger.exception("AVM model fetch from R2 failed for report %s, segment %s", report.id, segment)
        return {"ok": False, "reason": "model_fetch_failed", "segment": segment}

    try:
        if meta.text_transformer_path:
            text_row = text_features.transform_tfidf_svd(
                pd.Series([report.subject_description or ""]), pipelines["text_transformer"]
            )
            row = pd.concat([row.reset_index(drop=True), text_row.reset_index(drop=True)], axis=1)

        use_log = meta.target_transform == "log1p"
        ppsqm_point = _predict_one(pipelines["point"], row, use_log)
        ppsqm_low = _predict_one(pipelines["q_low"], row, use_log)
        ppsqm_high = _predict_one(pipelines["q_high"], row, use_log)

        blended = False
        if meta.blend_weight is not None and "companion_point" in pipelines:
            w = float(meta.blend_weight)
            cb_point = _predict_one_catboost(pipelines["companion_point"], row, use_log)
            cb_low = _predict_one_catboost(pipelines["companion_q_low"], row, use_log)
            cb_high = _predict_one_catboost(pipelines["companion_q_high"], row, use_log)
            ppsqm_point = w * ppsqm_point + (1 - w) * cb_point
            ppsqm_low = w * ppsqm_low + (1 - w) * cb_low
            ppsqm_high = w * ppsqm_high + (1 - w) * cb_high
            blended = True
    except Exception:
        logger.exception("AVM prediction failed for report %s, segment %s", report.id, segment)
        return {"ok": False, "reason": "prediction_error"}

    if ppsqm_low > ppsqm_high:
        ppsqm_low, ppsqm_high = ppsqm_high, ppsqm_low

    clamped = False
    cohort_median = _cohort_median_ppsqm(db, report.subject_property_type, report.subject_geo_category)
    if cohort_median:
        lo_bound, hi_bound = cohort_median * _CLAMP_LOW_MULT, cohort_median * _CLAMP_HIGH_MULT
        new_point = min(max(ppsqm_point, lo_bound), hi_bound)
        clamped = new_point != ppsqm_point
        ppsqm_point = new_point
        ppsqm_low = min(max(ppsqm_low, lo_bound), hi_bound)
        ppsqm_high = min(max(ppsqm_high, lo_bound), hi_bound)

    area = float(report.subject_area_sqm)
    return {
        "ok": True,
        "segment": segment,
        "ppsqm_point": round(ppsqm_point),
        "ppsqm_low": round(ppsqm_low),
        "ppsqm_high": round(ppsqm_high),
        "total_point": round(ppsqm_point * area),
        "total_low": round(ppsqm_low * area),
        "total_high": round(ppsqm_high * area),
        "model_trained_at": meta.trained_at,
        "metrics": meta.metrics,
        "clamped": clamped,
        "blended": blended,
    }
