# Track E4 — geo_cluster_id feature (spatial KMeans, K=25)

**Status: COMPLETE (2026-08-07). Recommendation: do NOT productionize.**

## What was tested

Whether `geo_cluster_id` (Track E3's K=25 spatial KMeans clustering over
geocoded neighborhoods, Track E1) adds predictive value on top of the
**current production feature set** — cleaned data, tuned per-segment
LightGBM, TF-IDF+SVD-15 text where the segment uses it (all except office).
5-fold CV, TF-IDF refit per fold (matches how it's actually validated/
deployed), `geo_cluster_id` computed once from the external geocode lookup
(no leakage risk — it's not fit on price data).

## Results

| Segment | Baseline MAE | +geo_cluster MAE | Delta MAE | Coverage |
|---|---|---|---|---|
| residential | 261.2 | 260.4 | **−0.28%** | 61.2% |
| office | 535.7 | 535.3 | **−0.07%** | 60.9% |
| retail | 526.1 | 525.3 | **−0.15%** | 61.6% |
| industrial | 229.9 | 229.6 | **−0.13%** | 63.7% |
| hospitality | 371.2 | 368.9 | **−0.62%** | 86.2% |

R² moved by ≤0.003 in every segment (residential 0.808→0.809, office
0.512→0.511, retail 0.529→0.529, industrial 0.490→0.492, hospitality
0.352→0.355).

## Conclusion

Every segment landed inside noise (all deltas under 1%, most under 0.3%).
Even hospitality — which has by far the best geocoding coverage (86.2% vs.
~61-64% elsewhere) and so gave the feature its best shot — only moved
0.62%. This matches the pre-experiment prediction: `title_geo_2_model`
(already a categorical feature) and, where present, CatBoost's native
categorical handling already capture most of the location signal that a
derived spatial cluster would add. The coverage gap (36-39% of listings in
4/5 segments resolve to "unknown" cluster) further caps any ceiling this
feature could have reached.

**Recommendation: do not add `geo_cluster_id` to production.** The
implementation cost (new migration, geocoding pipeline becoming a live
production dependency, retraining, serving-layer lookup) isn't justified
by a sub-1% MAE change that's indistinguishable from CV noise. This closes
out Track E — E1/E2/E3 (geocoding + clustering) were worthwhile
infrastructure to build and test, but the resulting feature doesn't earn
its way into the model.

Track D (LLM structured attribute extraction) remains unstarted and would
need an infrastructure decision (no LLM API key currently configured in
this project) before it could be tested the same rigorous way.
