# Round 1 — Data cleaning, target-transform, scaling, hyperparameter tuning

**Status:** complete. Applied to production (`scripts/train_avm_model.py`,
migrations 0010–0012). Full interactive version: `report.html` (published
as an Artifact — "AVM Model Diagnostics"). This file is the portable
markdown record of the same findings.

**MLflow:** 25 runs backfilled (`avm-round3-<segment>` experiments — reused
the same per-segment experiments for continuity — tagged `round=1`,
`backfilled=true`): 4 data/target configs + 1 tuned-final run, × 5 segments.

**Methodology:** 5-fold CV (mean ± std across folds), replacing the
production models' original single 60/20/20 stratified split.

---

## Root cause (not generic "outliers")

1. **Area-parsing failures.** 8 residential listings have
   `area_sqm_model = 1.00` exactly — including a Halkidiki villa priced at
   €310,000, reading as €310,000/m². Six more of the top-15 residential
   listings by price/m² share the identical `area = 1.00` signature — a
   parser fallback bug, not market variance.
2. **1,054 foreign listings** (mostly Greek Halkidiki vacation homes,
   already tagged `geo_category = 'foreign'` by the existing pipeline but
   never excluded from training) mixed into the Bulgarian residential set.
3. **Combined effect:** residential price/m² skewness dropped from **87.3**
   (pre-cleaning) to **0.77** (post-cleaning) — confirming the "long tail"
   was mostly contamination, not genuine market skew. Same pattern in every
   segment (retail 7.1→1.3, industrial 47.3→1.8, office 4.5→0.9,
   hospitality 2.0→1.5).

## Why cross-validation, not a single split

Residential's original single-split validation R² was 0.11 vs. test R² of
0.60 — same model, wildly different scores depending on which fold a
handful of extreme rows landed in. Switching to 5-fold CV confirmed this:
fold-to-fold R² std on uncleaned data was **±0.121**; on cleaned data it
collapsed to **±0.005**. That variance collapse is the strongest evidence
that cleaning, not the model, was the real problem.

## Results by segment (5-fold CV)

| Segment | Config | MAE (EUR/m²) | R² | ±10% |
|---|---|---|---|---|
| residential | A — raw data, raw target (≈old production) | 353.0 ± 2.5 | 0.345 ± 0.121 | 35.7% |
| residential | B — clean data, raw target (**winner**) | 304.5 ± 1.7 | 0.754 ± 0.005 | 38.1% |
| residential | Tuned LightGBM on B | **272.9 ± 1.7** | **0.792 ± 0.005** | 43.3% |
| office | A — raw baseline | 593.5 ± 22.5 | 0.421 ± 0.072 | 23.8% |
| office | D — clean, log, scaled (**winner**) | 536.0 ± 14.1 | 0.504 ± 0.058 | 24.9% |
| office | Tuned LightGBM on D | **535.7 ± 11.0** | **0.509 ± 0.045** | 23.6% |
| retail | A — raw baseline | 635.3 ± 17.1 | 0.366 ± 0.066 | 17.7% |
| retail | C — clean, log target (**winner**) | 535.9 ± 7.2 | 0.514 ± 0.007 | 17.7% |
| retail | Tuned LightGBM on C | **536.5 ± 7.1** | **0.512 ± 0.006** | 17.9% |
| industrial | A — raw baseline | 434.1 ± 81.1 | **−0.285 ± 0.449** (broken) | 9.8% |
| industrial | D — clean, log, scaled (**winner**) | 249.6 ± 12.0 | 0.465 ± 0.037 | 12.3% |
| industrial | Tuned LightGBM on D | **244.9 ± 9.5** | **0.471 ± 0.031** | 12.3% |
| hospitality | A — raw baseline | 434.2 ± 22.0 | 0.343 ± 0.111 | 12.6% |
| hospitality | C — clean, log target (**winner**) | 394.7 ± 12.1 | 0.310 ± 0.045 | 13.3% |
| hospitality | Tuned LightGBM on C | **381.4 ± 19.9** | **0.334 ± 0.051** | 13.9% |

(Full A/B/C/D comparison per segment is in `results.json` / `report.html`.)

## What each lever actually contributed

- **Cleaning (foreign exclusion + area floor + 0.5%/99.5% target trim):**
  the dominant lever everywhere. Rescued industrial from a literally broken
  model (negative R²) to usable (R²=0.47). Fixed residential's CV
  instability. Meaningful gain in every segment.
- **Log-transform:** mixed, inconsistent, small. Won by a hair in 4/5
  segments (office, retail, industrial, hospitality) but the margin over
  the raw-target config was noise-level once data was cleaned — the
  perceived skew was mostly the outlier contamination itself, not real
  market skew.
- **Feature scaling:** confirmed non-issue for LightGBM (as expected for
  tree models) — configs C vs. D differed only at noise level in all 5
  segments.
- **Hyperparameter tuning:** meaningful only for residential (304.5→272.9,
  because 141K rows gives deep trees/more estimators real room to pay
  off). Close to a wash for the other four segments.

## Recommendations applied to production

1. Cleaning step folded into `scripts/train_avm_model.py` — **done**.
2. Per-segment tuned hyperparameters adopted, each paired with the target
   transform it was actually validated on (residential=raw,
   office/retail/industrial/hospitality=log1p) — **done**, via
   `SEGMENT_LGBM_PARAMS` / `SEGMENT_USE_LOG_TARGET` + `target_transform`
   column on `avm_models` (migration 0012).
3. Verified end-to-end against the live app (`/comparables/` AVM panel,
   both raw- and log1p-target segments) before considering this closed.

## Caveats

- These are imot.bg **asking** prices, not closed transaction prices —
  some MAE floor is structural.
- Office/industrial/hospitality are still small-sample (1,098–2,778
  cleaned rows) — CV std bands are wide enough that routine retraining
  will show real number movement that isn't model regression, just
  small-sample noise.
- One random seed per config — relative ranking (cleaning >> tuning >>
  transform/scaling) is robust, exact numbers would shift a few points
  under a different seed.
