# Track F — CatBoost+blend productionization design (draft, not applied)

**Status: design only.** Nothing here has touched the real repo. Written
so a decision to proceed can move straight to implementation instead of
starting design from zero. Depends on Round 2's findings
(`ROUND2_findings.md`) and the frozen per-segment CatBoost hyperparameters
found there.

## Design choice: one `avm_models` row per segment, with optional companion fields

The current schema enforces exactly one active `AvmModel` row per segment
(`uq_avm_models_active_per_segment` partial unique index). Rather than
relaxing that to "one active row per (segment, algorithm)" — which would
ripple into `avm_service.get_active_model_meta` and the admin UI's
per-segment model listing — keep **one row represents one segment's
serving config**, and let that row optionally describe a blend via
nullable companion fields. `blend_weight IS NULL` (or `1.0`) means
"single model, unchanged" — residential stays exactly as it is today,
zero behavior change there.

## 1. Migration `0013_avm_blend_companion.py`

```python
def upgrade():
    op.add_column("avm_models", sa.Column("companion_algorithm", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_model_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_quantile_low_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("companion_quantile_high_path", sa.Text(), nullable=True))
    op.add_column("avm_models", sa.Column("blend_weight", sa.Numeric(4, 3), nullable=True))
    # blend_weight = weight on THIS row's (primary) algorithm; companion gets (1 - blend_weight)
```

Companion shares the primary row's `target_transform` — Round 2 tuned both
algorithms on the same target space per segment, no need for a second
transform field.

## 2. `app/db/models.py` — add the same 5 columns to `AvmModel`

Mechanical, mirrors the migration.

## 3. `scripts/train_avm_model.py`

New frozen config (from `ROUND2_findings.md`):

```python
SEGMENT_CATBOOST_PARAMS = {
    "office":      {"iterations": 800, "depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 3,  "bagging_temperature": 1.0, "random_strength": 10},
    "retail":      {"iterations": 500, "depth": 6,  "learning_rate": 0.08, "l2_leaf_reg": 3,  "bagging_temperature": 1.0, "random_strength": 5},
    "industrial":  {"iterations": 500, "depth": 6,  "learning_rate": 0.05, "l2_leaf_reg": 1,  "bagging_temperature": 1.0, "random_strength": 10},
    "hospitality": {"iterations": 800, "depth": 8,  "learning_rate": 0.02, "l2_leaf_reg": 20, "bagging_temperature": 0.0, "random_strength": 5},
    # residential deliberately absent — Round 2 found no benefit there
}
SEGMENT_BLEND_WEIGHT = {  # weight on LightGBM; None = no blend
    "residential": None, "office": 0.30, "retail": 0.70, "industrial": 0.40, "hospitality": 0.30,
}
```

In `train_segment()`, after the existing LightGBM point/quantile fit+save
block: if `SEGMENT_BLEND_WEIGHT[segment]` is not `None`, additionally fit
a CatBoost point pipeline + 2 quantile models (`max_ctr_complexity=1`,
same `use_log_target` as the primary, `SEGMENT_CATBOOST_PARAMS[segment]`)
on the identical train split, save 3 more `.joblib` files under
`models/avm/<segment>/<timestamp>/catboost_*.joblib`, and populate the new
companion_* + blend_weight fields on the same `AvmModel` insert.

No retuning happens here — hyperparameters are frozen from Round 2, same
philosophy as the LightGBM tuning adopted in Round 1 (re-tune periodically,
not on every retrain — CatBoost's ~90min/segment tuning cost makes this
non-negotiable, not just a nicety).

## 4. `utils/ml/avm_features.py` — one new helper

`build_feature_row()` already returns categoricals as raw strings (one-hot
happens inside the LightGBM sklearn `Pipeline`, not in this function) —
so it's already 90% suitable as CatBoost input. Just needs NaN-safe
categoricals:

```python
def prep_for_catboost(row: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    row = row.copy()
    for c in categorical_cols:
        row[c] = row[c].fillna("missing").astype(str)
    return row
```

## 5. `app/services/avm_service.py`

- `_load_pipelines(meta)`: if `meta.companion_model_path` is set, also
  `joblib.load` the 3 companion files into the cached dict under
  `"companion_point"` / `"companion_q_low"` / `"companion_q_high"`.
- `predict_sales_value()`: after computing primary
  `ppsqm_point/low/high` exactly as today, if `meta.blend_weight` is not
  `None`:
  ```python
  cb_row = avm_features.prep_for_catboost(row, avm_features.CATEGORICAL_COLS)
  cb_point = _predict_one(pipelines["companion_point"], cb_row, use_log)
  cb_low   = _predict_one(pipelines["companion_q_low"],  cb_row, use_log)
  cb_high  = _predict_one(pipelines["companion_q_high"], cb_row, use_log)
  w = float(meta.blend_weight)
  ppsqm_point = w * ppsqm_point + (1 - w) * cb_point
  ppsqm_low   = w * ppsqm_low   + (1 - w) * cb_low
  ppsqm_high  = w * ppsqm_high  + (1 - w) * cb_high
  ```
  (before the existing cohort-median clamp step, so blended values still
  get the same sanity-guard treatment as single-model ones today.)
- Result dict gains an informational `"blended": bool` field so the AVM
  panel can optionally show "модел: LightGBM + CatBoost blend" instead of
  just "LightGBM" — small template change in
  `comparables/_avm_panel.html`, not required for correctness.

## 6. `requirements.txt`

`catboost==1.2.10` moves from experimental-only to a real pinned
dependency — it becomes a production import path once any segment has a
companion model, not just an experiment tool.

## 7. Admin UI (`/admin/avm`, `app/templates/admin/avm.html`)

Minor addition: show blend weight + companion algorithm in the per-segment
card when present, so it's visible which segments are running a blend vs.
single-model — otherwise the retrain button and metrics table need no
structural change (they already read `AvmModel` rows generically).

## What this does NOT change

- Residential's serving path — zero behavior change, `blend_weight IS
  NULL` short-circuits the new blending branch entirely.
- The admin retrain trigger's UX (`POST /admin/avm/retrain`) — same
  interface, `train_segment()` just does more work internally for the 4
  blended segments.
- Anything about the sales-approach save flow, subject form, or report
  export — those only ever touch the final blended value, agnostic to how
  it was produced.

## Estimated implementation effort (once decided)

Roughly a half-day: 1 migration, ~40 lines in `train_avm_model.py`, ~20
lines in `avm_service.py`, 1 new small helper in `avm_features.py`, minor
template tweak. The bulk of the actual risk is testing (retrain all 4
blended segments, verify predictions end-to-end through `/comparables/`
the same way Round 1's changes were verified) — not code volume.
