# Round 2 — CatBoost vs. production LightGBM, and blending

**Status: COMPLETE AND APPLIED TO PRODUCTION (2026-08-10).** Migration
0013 (`companion_algorithm`, `companion_model_path`,
`companion_quantile_low_path`, `companion_quantile_high_path`,
`blend_weight` on `avm_models`), `scripts/train_avm_model.py` (fits +
saves a frozen-hyperparameter CatBoost companion for the 4 blended
segments), `app/services/avm_service.py` (blends primary + companion
predictions at inference), `catboost==1.2.10` now a real dependency in
`requirements.txt`. Verified end-to-end through the actual app
(`/comparables/` AVM panel shows "LightGBM + CatBoost (комбиниран)" for
blended segments, residential unaffected, DOCX export unaffected,
`/admin/avm` renders correctly).

**Live production retrain results** (single train/test split, not the
5-fold CV means below — some variance vs. the CV numbers is expected,
direction is what matters and it's consistent everywhere):

| Segment | LightGBM alone | Blend | Δ MAE |
|---|---|---|---|
| residential | MAE=276.8, R²=0.788 | *(no blend, by design)* | — |
| office | MAE=573.9, R²=0.460 | MAE=560.5, R²=0.487 | −2.3% |
| retail | MAE=543.5, R²=0.515 | MAE=538.6, R²=0.525 | −0.9% |
| industrial | MAE=247.2, R²=0.517 | MAE=238.1, R²=0.534 | −3.7% |
| hospitality | MAE=383.4, R²=0.358 | MAE=360.4, R²=0.413 | −6.0% |

The original CV-based investigation below is preserved as-is for the
methodology and reasoning; treat this section as the final outcome.

---

**Original findings (CV-based investigation):** All 5 segments finished. Total runtime 28,267s
(7.85 hours) — far longer than the original ~2-3h estimate, because
CatBoost's per-segment tuning cost was consistently ~88-93 minutes
**regardless of segment size** (5,362s / 5,545s / 5,589s / 5,351s / 4,849s
for residential/office/retail/industrial/hospitality respectively) —
confirms the fixed-overhead theory from the isolation testing, not a
row-count-driven cost. Nothing from this round has been applied to
production yet — this file is the findings, not a decision record.

**MLflow:** 20 runs backfilled across all 5 segments (`lightgbm_production`,
`catboost_tuned`, `best_blend`, `equal_blend_50_50` × 5), tagged `round=2`,
`backfilled=true`.

**Methodology:** LightGBM uses the exact production hyperparameters from
Round 1 (fixed reference point, not retuned here). CatBoost tuned via
RandomizedSearchCV (25-40 candidates × 3-fold, `max_ctr_complexity=1` to
keep fit time bounded). Both evaluated via proper 5-fold out-of-fold (OOF)
predictions; blend weight grid-searched on the full OOF set afterward
(never inside a single fold) — not optimistically biased by weight-fitting
leakage.

## Full results, all 5 segments

| Segment | LightGBM MAE / R² | CatBoost MAE / R² | Best blend MAE / R² (w_lgbm) | vs. LightGBM alone |
|---|---|---|---|---|
| residential | 272.9 / 0.792 | 303.8 / 0.754 (worse) | 272.8 / 0.792 (w=0.95) | **no gain** |
| office | 535.7 / 0.512 | 516.7 / 0.540 (better) | **512.9 / 0.547** (w=0.30) | **-4.2% MAE** |
| retail | 536.5 / 0.512 | 547.1 / 0.502 (worse) | **533.3 / 0.516** (w=0.70) | -0.6% MAE |
| industrial | 245.0 / 0.470 | 243.3 / 0.468 (~tie) | **240.5 / 0.481** (w=0.40) | -1.8% MAE |
| hospitality | 381.5 / 0.331 | 373.9 / 0.350 (better) | **372.0 / 0.357** (w=0.30) | -2.5% MAE |

## The pattern is clean and matches the working theory

**Residential (141K rows) is the one segment where CatBoost adds nothing**
— the grid search found w_lgbm=0.95, i.e. "don't blend." **All four smaller
segments (1.1K-7.4K rows) benefit from blending**, and CatBoost alone beats
LightGBM alone in 3 of those 4 (office, industrial, hospitality — only
retail has CatBoost losing outright, though the blend still helps there
too).

This matches the theory from the residential-only interim finding: at
141K rows, one-hot encoding already has enough samples per category to
estimate well, so CatBoost's native categorical handling (its main
structural advantage) has nothing to fix. At 1K-7K rows with hundreds of
distinct neighborhoods, one-hot sparsity is a real problem — that's
exactly where CatBoost's ordered target statistics earn their keep.

Gains are real but modest — roughly 0.6% to 4.2% MAE reduction, not a
transformative jump. Office sees the largest and most consistent
improvement (both CatBoost alone and the blend win clearly there).

## Operational cost — the real constraint on adoption

CatBoost tuning cost ~90 minutes per segment with no relationship to row
count — this is fixed categorical-handling overhead in this specific
Windows environment (isolation-tested and confirmed in Round 2's own prep
work: numeric-only fit 0.8s vs. any-categorical fit 14-23s on identical
tiny data). Adopting CatBoost+blend for the 4 non-residential segments
would add ~6 hours to any full retrain-with-retuning cycle if CatBoost
were retuned every time.

**This should not block adoption, but should shape how it's adopted:**
freeze the tuned CatBoost hyperparameters found this round (same pattern
as Round 1's LightGBM tuning — reuse the found params on every routine
retrain, only re-run the expensive RandomizedSearchCV periodically, not on
every scrape-triggered retrain).

## Preliminary recommendation

- **Residential:** no change. Production LightGBM already wins outright;
  CatBoost and blending add nothing here.
- **Office, retail, industrial, hospitality:** blending LightGBM + CatBoost
  (with the per-segment weights found here) beats LightGBM alone. Worth
  productionizing — but this is a real **architecture change**, not a
  parameter tweak: `avm_service.py` would need to load and combine two
  models' predictions per segment instead of one, `avm_models`/training
  script need a second model type + blend weight concept. Bigger lift than
  Round 1's changes (which kept the single-model architecture intact).

**Not yet decided:** whether to actually build this. Flagged to the user
as a scope decision before implementation starts.

## Tuned CatBoost hyperparameters found (for reuse if adopted)

| Segment | iterations | depth | learning_rate | l2_leaf_reg | bagging_temperature | random_strength |
|---|---|---|---|---|---|---|
| residential | 500 | 10 | 0.10 | 1 | 0.3 | 10 |
| office | 800 | 10 | 0.05 | 3 | 1.0 | 10 |
| retail | 500 | 6 | 0.08 | 3 | 1.0 | 5 |
| industrial | 500 | 6 | 0.05 | 1 | 1.0 | 10 |
| hospitality | 800 | 8 | 0.02 | 20 | 0.0 | 5 |

All tuned with `max_ctr_complexity=1`; all except residential use the
`log1p` target (same choice as their Round 1 LightGBM models).
