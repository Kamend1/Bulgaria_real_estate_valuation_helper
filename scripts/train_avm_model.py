"""
Train segment-aware AVM (automated valuation model) pipelines directly from
the `listings` table and register them in `avm_models`.

Ports the pipeline proven in
notebooks/04_residential_real_estate_regression_ML_analysis.ipynb (LightGBM
over a median-impute / one-hot ColumnTransformer) to read live DB data
instead of a parquet snapshot, and repeats it independently for each of the
5 asset-class segments defined in utils/ml/avm_features.py.

Usage (from project root):
    python -m scripts.train_avm_model                  # train all 5 segments
    python -m scripts.train_avm_model --segment office  # retrain one segment
    python -m scripts.train_avm_model --min-rows 200    # override the guard

A segment whose training_eligible row count falls below --min-rows is
skipped entirely (no model written, no avm_models row) rather than being
trained and silently activated with too little data.
"""
import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import bindparam, text

from app.config import settings
from app.db.base import engine
from app.db.models import AvmModel
from app.db.session import db_session
from utils.ml.avm_features import CATEGORICAL_COLS, NUMERIC_COLS, SEGMENT_PROPERTY_TYPES

# Per-segment LightGBM hyperparameters from RandomizedSearchCV tuning
# (25-40 candidates x 3-fold CV, run against cleaned live-DB data — see the
# AVM Model Diagnostics report, 2026-08-07). Each segment was tuned in
# whichever target space (raw vs log1p) won its own data/target comparison,
# so SEGMENT_LGBM_PARAMS and SEGMENT_USE_LOG_TARGET below must be read
# together per segment, not mixed across segments.
SEGMENT_LGBM_PARAMS = {
    "residential": {
        "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127,
        "max_depth": -1, "min_child_samples": 10, "subsample": 1.0,
        "colsample_bytree": 0.9, "reg_lambda": 0.1, "reg_alpha": 0.0,
    },
    "office": {
        "n_estimators": 800, "learning_rate": 0.02, "num_leaves": 15,
        "max_depth": 12, "min_child_samples": 10, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 1.0,
    },
    "retail": {
        "n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 31,
        "max_depth": 6, "min_child_samples": 10, "subsample": 1.0,
        "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 1.0,
    },
    "industrial": {
        "n_estimators": 800, "learning_rate": 0.02, "num_leaves": 63,
        "max_depth": 6, "min_child_samples": 10, "subsample": 0.9,
        "colsample_bytree": 1.0, "reg_lambda": 0.1, "reg_alpha": 0.1,
    },
    "hospitality": {
        "n_estimators": 300, "learning_rate": 0.02, "num_leaves": 31,
        "max_depth": 8, "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 1.0,
    },
}

# Whether each segment's model was tuned/validated on log1p(price_per_sqm)
# rather than the raw value. Only residential won on the raw target — for
# the other four, log1p gave a (small) but consistent MAE edge once the data
# was cleaned. Must stay in sync with SEGMENT_LGBM_PARAMS above and with
# AvmModel.target_transform written at the end of train_segment().
SEGMENT_USE_LOG_TARGET = {
    "residential": False,
    "office": True,
    "retail": True,
    "industrial": True,
    "hospitality": True,
}

# Implausibly small area (sqm) below which a row is almost certainly a
# parse error, not a real listing (e.g. a Halkidiki villa priced at
# EUR 310,000 with area_sqm_model = 1.00 -> "price/sqm" of EUR 310,000).
# Segment-specific because office/retail legitimately include small units
# more often than residential does.
MIN_AREA_BY_SEGMENT = {
    "residential": 15, "office": 8, "retail": 5, "industrial": 15, "hospitality": 20,
}

# Percentile bounds used to trim the remaining extreme tails of the target
# after the hard area/foreign filters below.
TARGET_TRIM_BOUNDS = (0.005, 0.995)

MIN_STRATUM_SIZE = 20   # strata smaller than this get folded into OTHER_SMALL_STRATA


def _load_segment_df(slugs: list[str]) -> pd.DataFrame:
    query = text("""
        SELECT
            area_sqm_model              AS area_sqm,
            views,
            features_count,
            construction_year_model,
            floor_model,
            total_floors_model,
            property_type_raw,
            title_city_model,
            title_geo_2_model,
            geo_category,
            construction_type_model,
            features_pipe,
            price_per_sqm_model         AS price_per_sqm
        FROM listings
        WHERE training_eligible = TRUE
          AND deal_type_normalized = 'sale'
          AND property_type_slug IN :slugs
          AND price_per_sqm_model IS NOT NULL
    """).bindparams(bindparam("slugs", expanding=True))
    return pd.read_sql(query, engine, params={"slugs": slugs})


def _clean_segment_df(df: pd.DataFrame, segment: str) -> tuple[pd.DataFrame, dict]:
    """
    Excludes foreign listings and implausibly small areas (both confirmed
    parse errors / out-of-scope rows), then trims the remaining extreme
    tails of the target. See the AVM Model Diagnostics report for the
    root-cause investigation this is based on.
    """
    before = len(df)
    min_area = MIN_AREA_BY_SEGMENT.get(segment, 10)
    cleaned = df[(df["geo_category"] != "foreign") & (df["area_sqm"] > min_area)].copy()

    lo_q, hi_q = TARGET_TRIM_BOUNDS
    lo, hi = cleaned["price_per_sqm"].quantile([lo_q, hi_q])
    cleaned = cleaned[(cleaned["price_per_sqm"] >= lo) & (cleaned["price_per_sqm"] <= hi)].copy()

    after = len(cleaned)
    stats = {
        "before": before, "after": after, "removed": before - after,
        "removed_pct": round((before - after) / before * 100, 2) if before else 0,
    }
    return cleaned, stats


def _feature_col_name(feature: str) -> str:
    return "feature_" + (
        feature.lower()
        .replace(" ", "_").replace("-", "_").replace(",", "")
        .replace("/", "_").replace("(", "").replace(")", "")
    )


def _add_binary_feature_cols(df: pd.DataFrame, min_count: int) -> list[str]:
    """Recomputes amenity dummy columns from features_pipe for this segment's
    own data (office/retail/industrial amenities barely overlap with
    residential ones, so this can't be a fixed list)."""
    exploded = (
        df["features_pipe"].dropna().astype(str).str.split("|").explode().str.strip()
    )
    exploded = exploded[exploded != ""]
    counts = exploded.value_counts()
    selected = counts[counts >= min_count].index.tolist()

    binary_cols = []
    for feat in selected:
        col = _feature_col_name(feat)
        binary_cols.append(col)
        df[col] = (
            df["features_pipe"].fillna("").astype(str).str.split("|")
            .apply(lambda feats: int(feat in [f.strip() for f in feats]))
        )
    return binary_cols


def _build_strata(df: pd.DataFrame) -> pd.Series:
    strata_raw = df["geo_category"].astype(str) + "__" + df["property_type_raw"].astype(str)
    counts = strata_raw.value_counts()
    valid = counts[counts >= MIN_STRATUM_SIZE].index
    return strata_raw.where(strata_raw.isin(valid), "OTHER_SMALL_STRATA")


def _build_preprocessor(binary_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("bin", "passthrough", binary_cols),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def _evaluate(y_true, y_pred) -> dict:
    errors = y_true - y_pred
    abs_pct_errors = np.abs(errors / y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(abs_pct_errors.mean() * 100),
        "within_10_pct": float((abs_pct_errors <= 0.10).mean() * 100),
        "within_20_pct": float((abs_pct_errors <= 0.20).mean() * 100),
        "within_30_pct": float((abs_pct_errors <= 0.30).mean() * 100),
    }


def _fit_pipeline(binary_cols: list[str], X_train, y_train, lgbm_params: dict, **lgbm_kwargs) -> Pipeline:
    pipeline = Pipeline(steps=[
        ("preprocessor", _build_preprocessor(binary_cols)),
        ("model", LGBMRegressor(
            **lgbm_params, random_state=42, n_jobs=-1, verbosity=-1, **lgbm_kwargs,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def _predict_raw(pipeline: Pipeline, X, use_log_target: bool) -> np.ndarray:
    """Predicts and inverse-transforms back to raw EUR/sqm if the pipeline
    was fit on log1p(target)."""
    pred = pipeline.predict(X)
    if use_log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 1, None)


def train_segment(segment: str, min_rows: int) -> None:
    print(f"\n{'='*60}\nSegment: {segment}")
    slugs = SEGMENT_PROPERTY_TYPES[segment]

    df = _load_segment_df(slugs)
    raw_row_count = len(df)
    print(f"training_eligible rows: {raw_row_count} (threshold: {min_rows})")

    if raw_row_count < min_rows:
        print(f"  SKIPPED — below minimum row threshold, no model trained.")
        return

    df, clean_stats = _clean_segment_df(df, segment)
    row_count = len(df)
    print(f"  cleaned: {clean_stats} -> {row_count} rows")

    if row_count < min_rows:
        print(f"  SKIPPED — below minimum row threshold after cleaning, no model trained.")
        return

    min_feature_count = max(20, round(0.02 * row_count))
    binary_cols = _add_binary_feature_cols(df, min_feature_count)
    print(f"  amenity features selected (min_count={min_feature_count}): {len(binary_cols)}")

    df["strata"] = _build_strata(df)

    try:
        train_val_df, test_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["strata"]
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["strata"]
        )
    except ValueError as exc:
        print(f"  SKIPPED — stratified split failed ({exc}).")
        return

    model_feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + binary_cols
    target_col = "price_per_sqm"

    X_train, y_train = train_df[model_feature_cols], train_df[target_col]
    X_val, y_val = val_df[model_feature_cols], val_df[target_col]
    X_test, y_test = test_df[model_feature_cols], test_df[target_col]

    lgbm_params = SEGMENT_LGBM_PARAMS[segment]
    use_log_target = SEGMENT_USE_LOG_TARGET[segment]
    y_train_fit = np.log1p(y_train) if use_log_target else y_train
    print(f"  target transform: {'log1p' if use_log_target else 'raw'}")

    print("  fitting point-estimate model...")
    point_pipeline = _fit_pipeline(binary_cols, X_train, y_train_fit, lgbm_params, objective="regression")

    print("  fitting quantile models (p10 / p90)...")
    q_low_pipeline = _fit_pipeline(binary_cols, X_train, y_train_fit, lgbm_params, objective="quantile", alpha=0.1)
    q_high_pipeline = _fit_pipeline(binary_cols, X_train, y_train_fit, lgbm_params, objective="quantile", alpha=0.9)

    val_metrics = _evaluate(y_val, _predict_raw(point_pipeline, X_val, use_log_target))
    print(f"  validation: MAE={val_metrics['mae']:.1f}  R2={val_metrics['r2']:.3f}  "
          f"within10%={val_metrics['within_10_pct']:.1f}%")

    test_metrics = _evaluate(y_test, _predict_raw(point_pipeline, X_test, use_log_target))
    print(f"  test:       MAE={test_metrics['mae']:.1f}  R2={test_metrics['r2']:.3f}  "
          f"within10%={test_metrics['within_10_pct']:.1f}%")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = Path(settings.avm_models_dir) / segment / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    q_low_path = model_dir / "q_low.joblib"
    q_high_path = model_dir / "q_high.joblib"
    joblib.dump(point_pipeline, model_path)
    joblib.dump(q_low_pipeline, q_low_path)
    joblib.dump(q_high_pipeline, q_high_path)
    print(f"  saved to {model_dir}")

    with db_session() as session:
        previously_active = (
            session.query(AvmModel)
            .filter_by(segment=segment, is_active=True)
            .all()
        )
        for old in previously_active:
            old.is_active = False

        session.add(AvmModel(
            id=uuid.uuid4(),
            segment=segment,
            algorithm="lightgbm",
            feature_columns=model_feature_cols,
            hyperparams=lgbm_params,
            metrics={"validation": val_metrics, "test": test_metrics},
            training_row_count=row_count,
            min_row_threshold=min_rows,
            model_path=str(model_path),
            quantile_low_path=str(q_low_path),
            quantile_high_path=str(q_high_path),
            is_active=True,
            target_transform="log1p" if use_log_target else "raw",
        ))

    print(f"  activated new {segment} model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train segment-aware AVM models")
    parser.add_argument(
        "--segment", choices=list(SEGMENT_PROPERTY_TYPES.keys()), default=None,
        help="Train only this segment (default: train all 5)",
    )
    parser.add_argument(
        "--min-rows", type=int, default=300,
        help="Minimum training_eligible rows required to train+activate a segment (default: 300)",
    )
    args = parser.parse_args()

    segments = [args.segment] if args.segment else list(SEGMENT_PROPERTY_TYPES.keys())

    print("AVM Training")
    print(f"Database: {settings.database_url.split('@')[-1]}")
    print(f"Segments: {', '.join(segments)}")

    for segment in segments:
        train_segment(segment, min_rows=args.min_rows)

    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
