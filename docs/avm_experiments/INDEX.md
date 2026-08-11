# AVM experimentation — index

All experiment work for the segment-aware AVM (imot.bg appraisal helper),
across 3 rounds plus 3 parallel tracks (E/F/G). This directory is a
session scratchpad — nothing here is in the repo except what's explicitly
noted as "applied to production."

`HANDOFF.md` has the blow-by-blow history (including a real bug that cost
2.5h of compute once, found and fixed) if you want the story. This file is
the current-state map.

## Status at a glance

| Round/Track | Topic | Status | Applied to production? |
|---|---|---|---|
| 1 | Data cleaning, target-transform, scaling, LightGBM tuning | ✅ complete | ✅ yes |
| 2 | CatBoost vs. LightGBM, blending | ✅ complete (all 5 segments) | ✅ yes (2026-08-10) |
| 3 | Text features — Phase 1 (pilot) + Phase 2 (rollout) both complete, all 5 segments | ✅ complete | ✅ yes (2026-08-10) |
| E | Geocoding + spatial clustering + geo_cluster feature test | ✅ complete (E1-E4) | ❌ no — E4 result too small to justify |
| F | CatBoost+blend productionization design | ✅ design complete, not implemented | ❌ not yet |
| G | Manual LLM-extraction feasibility pilot | ✅ complete | n/a (research) |

**Round 1:** cleaning (foreign-listing exclusion, area floor, target
trim) + per-segment LightGBM tuning. Biggest single lever across the
whole project — rescued industrial from a broken model, fixed
residential's CV instability. **Live in production.**

**Round 2:** CatBoost + blending. No gain for residential (skip). Real,
modest gain (0.6-4.2% MAE) for office/retail/industrial/hospitality via
blending — but CatBoost's ~90min/segment tuning cost and the architecture
change (two models blended per segment) mean this needs a deliberate
go/no-go, not a rubber stamp. Design ready in
`TRACK_F_blend_productionization_design.md`.

**Round 3:** text features on `description_clean`. **TF-IDF+SVD-15 wins
outright** on both tested segments (residential MAE 272.9→261.2,
hospitality 381.5→371.2) — and it's nearly free in production (seconds to
fit, vs. CatBoost's 90min or MiniLM's 2.5h+). MiniLM (sentence-transformer
embeddings) **lost to TF-IDF on both segments** despite costing far more
compute — e5-base/bge-m3 skipped on that basis, see `ROUND3_findings.md`
for the reasoning. Best cost/benefit result of any round so far.

## Where things live

**Findings documents (read these, not the raw JSON, for the actual conclusions)**
- `ROUND1_findings.md` — complete, applied to production
- `ROUND2_findings.md` — complete, decision pending
- `ROUND3_findings.md` — complete, recommendation ready
- `ROUND3_trackG_manual_pilot.md` — manual LLM-extraction feasibility + corrected attribute schema
- `TRACK_F_blend_productionization_design.md` — Round 2 productionization design (not applied)
- `PLAN_round3_text_features.md` / `PLAN_parallel_and_next_phases.md` — original plans (superseded by the findings docs where they overlap)
- `HANDOFF.md` — narrative history + the MAX_PATH bug postmortem
- `report.html` — Round 1's interactive report, published as an Artifact ("AVM Model Diagnostics")

**MLflow** (`sqlite:///mlflow.db` in this folder, local, no server)
```
mlflow ui --backend-store-uri sqlite:///<this-folder>/mlflow.db
```
All 3 rounds logged (Round 1/2 backfilled after the fact via
`backfill_mlflow.py`, tagged `backfilled=true`; Round 3 logged live).

**Raw results (JSON) — reference only, findings docs are the summary**
- `results.json` (Round 1), `results_catboost_blend.json` (Round 2, all 5 segments),
  `results_phase1_pilot.json` (Round 3, all configs including MiniLM),
  `geocode_cache.json` (Track E1), `geo_clusters.json` (Track E3)

**Code**
- `run_experiments.py`, `run_catboost_blend.py`, `phase1_pilot.py` — Round 1/2/3 drivers
- `text_features.py` — Round 3 toolkit. FastText intentionally skipped (see original brainstorm reasoning).
- `geocode_neighborhoods.py`, `track_e2_e3.py` — Track E
- `mlflow_setup.py`, `backfill_mlflow.py` — shared tracking infra

**Cache**
- `C:\Users\kamen.dimitrov\avm_embed_cache\` — sentence-transformer embeddings, moved out of this deeply-nested folder after a MAX_PATH bug (see `HANDOFF.md`)
- `mlruns/` — pre-sqlite-migration leftover, harmless, ignore

## Open decisions (both independent, both can happen any time)

1. **Round 2 productionization** — adopt CatBoost+blend for office/retail/industrial/hospitality? Design ready, ~half-day implementation estimate.
2. **Round 3 Phase 2** — roll TF-IDF+SVD-15 out to the 3 untested segments (office/retail/industrial), then productionize alongside/instead of deciding on (1). Given TF-IDF's near-zero production cost, this is the more clearly-worth-it one of the two.

Track E4 (test `geo_cluster_id` as a feature, on top of the TF-IDF winner)
is **complete** — all 5 segments tested, deltas all under 1% (noise-level).
See `TRACK_E4_findings.md`. Not recommended for production.
