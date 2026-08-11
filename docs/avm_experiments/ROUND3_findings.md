# Round 3 — Text features (Phase 1 pilot findings)

**Status: COMPLETE AND APPLIED TO PRODUCTION (2026-08-10).** Migration
0014 (`text_transformer_path` on `avm_models`), `utils/ml/text_features.py`
(production TF-IDF+SVD-15 module), `scripts/train_avm_model.py` (fits +
saves the transformer for residential/retail/industrial/hospitality —
office excluded per the finding below), `app/services/avm_service.py`
(transforms `report.subject_description` and merges it into the feature
row at inference). For the 3 segments that are also blended (Round 2:
retail/industrial/hospitality), text columns were added symmetrically to
both LightGBM and CatBoost — see the design note in
`scripts/train_avm_model.py`'s `SEGMENT_USE_TEXT_FEATURES` comment for why
this is a reasoned simplification, not a re-validated blend weight.

**A real bug was found and fixed during verification** (not in the
production code — in the curl-based test harness): passing Cyrillic text
through `curl --data-urlencode` via Git Bash on Windows silently corrupted
it into `U+FFFD` replacement characters before it ever reached the app,
making early smoke tests show identical predictions regardless of
description content. Switching the test harness to Python's `requests`
(proper UTF-8 handling) resolved it immediately — verified predictions
genuinely differ by description content once real text reaches the model
(residential: no description → 1,752 EUR/m², a renovated/park-view
description → 1,857 EUR/m², a "needs renovation" description → 1,756).
Worth remembering for any future ad-hoc testing with Cyrillic payloads in
this environment.

**Final production numbers** (single train/test split):

| Segment | Round 1 (tuned) | Round 2 (+blend) | Round 3 (+text, final) |
|---|---|---|---|
| residential | 272.9 | *(no blend)* | **266.6** |
| office | 573.9 | 560.5 | 560.5 *(unchanged — no text)* |
| retail | 543.5 | 538.6 | **525.0** |
| industrial | 247.2 | 238.1 | **236.9** |
| hospitality | 383.4 | 360.4 | **353.5** |

Verified end-to-end through the actual app: descriptions change
predictions meaningfully, office is provably unaffected (identical to the
Round-2-only numbers), blend + text compound correctly (hospitality
improved further with both vs. either alone), DOCX export and
`/admin/avm` both unaffected.

---

**Original findings (Phase 1/2 pilot investigation):**

**Status: Phase 1 pilot COMPLETE** (residential + hospitality, the two
segment-size extremes, per the original staging plan).
**e5-base / bge-m3: SKIPPED by decision** — see reasoning below. Phase 2
(rolling TF-IDF out to all 5 segments) not yet started.

**MLflow:** all Phase 1 pilot runs logged live during execution (no
backfill needed here, unlike Rounds 1/2 — `phase1_pilot.py` calls
`log_mlflow_run` inline for every config).

## Full results

| Segment | Config | MAE | R² | ±10% |
|---|---|---|---|---|
| residential | baseline (no text) | 272.9 | 0.792 | 43.3% |
| residential | keyword flags (B3) | 267.5 | 0.799 | 44.0% |
| residential | **TF-IDF + SVD-15 (A1, best overall)** | **261.2** | **0.808** | 45.0% |
| residential | MiniLM + PCA-15 (A3, B1) | 270.7 | 0.796 | 43.6% |
| residential | MiniLM + text-residual (A3, B2) | 268.1 | 0.799 | 44.0% |
| hospitality | baseline (no text) | 381.5 | 0.331 | 13.9% |
| hospitality | keyword flags (B3) | 378.0 | 0.342 | 14.1% |
| hospitality | **TF-IDF + SVD-15 (A1, best overall)** | **371.2** | **0.352** | 15.1% |
| hospitality | MiniLM + PCA-15 (A3, B1) | 380.8 | 0.318 | 14.7% |
| hospitality | MiniLM + text-residual (A3, B2) | 398.5 (worse than baseline) | 0.336 | 13.6% |

(TF-IDF raw vs. normalized tied exactly in both segments — the Bulgarian
text normalization step made no measurable difference here.)

## The headline finding: cheapest method won, most expensive method lost

**TF-IDF beats MiniLM on both segments**, despite MiniLM costing
enormously more compute (2.73h for residential's embedding pass alone,
vs. seconds for TF-IDF). On hospitality, MiniLM's residual-stacking
variant is actually *worse than doing nothing* (398.5 vs. 381.5 baseline
MAE) — the 384-dim embedding overfits on only ~1,000 rows.

**Working explanation:** TF-IDF's vocabulary is fit directly on this
corpus's real-estate-specific terms (ремонт, тухла, гледка, etc.) — every
dimension is tied to a term that plausibly moves price. MiniLM is a
general-purpose multilingual *paraphrase* model — it's good at "these two
sentences mean the same thing," which is a different skill from "this
specific word predicts price." Compressing 384 generic-semantic
dimensions down to 15 via PCA may be discarding exactly the
price-relevant signal that TF-IDF's domain-fit SVD concentrates directly.

## Decision: skip e5-base and bge-m3

Both are larger and slower than MiniLM (more parameters, in bge-m3's case
also longer max sequence length) — there's no structural reason to expect
either to reverse a pattern that's about *representation type*
(domain-specific lexical vs. generic semantic), not raw model size or
embedding quality. Spending several more hours per model to test a
hypothesis the data already argues against isn't a good trade. If this
conclusion ever gets revisited, the bar should be "there's a specific new
reason to expect the pattern to flip" (e.g. bge-m3's longer context
window mattering for unusually long descriptions), not "let's just check."

## Recommendation

**TF-IDF is worth productionizing; MiniLM is not.** Rationale:
- Real, consistent gain on both tested segments (residential -4.3% MAE,
  hospitality -2.7% MAE) — smaller than Round 1's cleaning fix, but real
  and free of Round 2's operational cost problem.
- **Trivially cheap in production** — a `TfidfVectorizer` + `TruncatedSVD`
  fit takes seconds, not the ~90min/segment CatBoost tuning cost or the
  multi-hour MiniLM embedding cost. This is the best cost/benefit ratio of
  anything tested across all 3 rounds.
- Keyword flags alone (B3) also show a smaller but real, essentially-free
  gain — could be included alongside TF-IDF for near-zero extra cost, or
  treated as redundant with it (TF-IDF likely already captures most of
  what the keyword flags capture, since the keywords were themselves
  chosen by TF-IDF-adjacent frequency analysis).

## Phase 2 — COMPLETE (2026-08-10): office/retail/industrial validated

Same 5-fold CV discipline, TF-IDF only (the established winner — no need
to re-litigate keyword/MiniLM on 3 more segments given Phase 1's
conclusive result). Took 4.2 minutes total for all 3 segments.

| Segment | Baseline MAE | TF-IDF MAE | R² | Δ MAE |
|---|---|---|---|---|
| office | 535.7 | 534.9 | 0.512→0.510 | **−0.1% (negligible)** |
| retail | 536.5 | 526.1 | 0.512→0.529 | −1.9% |
| industrial | 245.0 | 229.9 | 0.470→0.490 | **−6.2% (best of all 5 segments)** |

**Full 5-segment picture:** residential −4.3%, office **flat**, retail
−1.9%, industrial −6.2%, hospitality −2.7%.

**Office is the one segment where TF-IDF doesn't help** — 534.9 vs. 535.7
is noise, not signal. Everywhere else the gain is real and consistent.
Possible explanation: office listings' descriptions may carry less price-
relevant variance than other segments' (condition/amenity language present
but perhaps less differentiating for office space specifically), though
this wasn't specifically investigated — noted as an open question, not a
settled explanation.

**Recommendation:** adopt TF-IDF+SVD-15 for residential, retail,
industrial, hospitality. Skip it for office — no benefit, no reason to
add the complexity even though the cost is low.

## Track E4 (spatial cluster feature) — now unblocked

Round 3's winning method is now known (TF-IDF), so the plan's condition
for testing `geo_cluster_id` (from Track E3) as an additional feature is
satisfied. Test config: baseline + TF-IDF (established winner) +
geo_cluster_id, vs. baseline + TF-IDF alone, same 2 pilot segments first.
