# Parallel tracks (while Phase 1 pilot's MiniLM embeds residential) + what's next

**Constraint driving this plan:** the running Phase 1 pilot is now CPU-bound
(MiniLM embedding 141,074 residential descriptions, ~2.5-3h). Anything
launched alongside it should be network-bound or pure code/design work, not
another CPU-heavy training job — otherwise both slow down and the timing
data from either becomes harder to trust (same reasoning as not running
Round 3 alongside Round 2 earlier).

## Track E — Geocoding + spatial clustering (revives an untouched brainstorm item)

From the original brainstorm: clustering was flagged as promising but
blocked on not having lat/lng — only text neighborhood names. Confirmed
network access works (Nominatim/OpenStreetMap, tested, 200 OK).

- **E1** — Geocode all unique `(title_city_model, title_geo_2_model)` pairs
  (a few thousand, not 260K — one geocode per distinct neighborhood, not
  per listing). Nominatim rate limit is 1 req/sec, so ~3,000 pairs ≈ 50
  min. Cache to `geocode_cache.json` (query → lat/lon), safe to resume if
  interrupted.
- **E2** — Coverage check: what % resolved successfully, spot-check a
  handful against known Sofia neighborhoods for sanity.
- **E3** — KMeans/DBSCAN on the resolved coordinates → `geo_cluster_id`.
  Cheap once coordinates exist (seconds, not hours). Deliberately spatial,
  not price-based — avoids the target-leakage risk flagged when this was
  first brainstormed.
- **E4** — Once Phase 1 pilot tells us whether text features are worth
  keeping, test `geo_cluster_id` as an *additional* engineered feature the
  same way (baseline vs. baseline+cluster, 5-fold OOF, on residential +
  hospitality first).

**Status: starting E1 now** — network-bound, doesn't touch CPU meaningfully.

## Track F — CatBoost+blend productionization (Round 2's pending decision)

Not deploying anything — user explicitly deferred this decision until
Round 3 is in. But the code changes can be **designed and drafted** now
(pure writing, no compute), so it's ready to review/apply quickly once a
decision is made instead of starting from zero:

- Schema: `avm_models` needs a way to represent "this segment blends two
  models" — either a second `AvmModel` row per segment with a
  `blend_weight` column, or a `companion_model_id` self-reference. Leaning
  toward the former (simpler, keeps `AvmModel` rows uniform).
- `scripts/train_avm_model.py`: fit + save both LightGBM and CatBoost per
  blended segment (residential stays LightGBM-only — no change there).
- `app/services/avm_service.py`: `predict_sales_value` needs to load and
  combine two pipelines' predictions per segment when a blend is active,
  using the frozen per-segment weight from Round 2 (§ROUND2_findings.md).
- New dependency: `catboost` would need to move from experimental into
  `requirements.txt` for real this time.

**Status: will draft as a reviewable diff, not applied.**

## Track G — LLM structured extraction (Track D from the original Round 3 plan)

- **D1 (schema design)** — no API needed, can do now.
- **D2 (pilot validation)** — originally scoped as prompting an LLM API
  over a sample. No LLM API key is configured in this project. Cheaper
  substitute available right now: I read a batch of real
  `description_clean` samples directly and manually assess how
  consistently the target attributes (renovation, view, proximity
  mentions) can actually be extracted — a feasibility proxy that costs
  nothing and doesn't need any new infrastructure decision. If it looks
  promising, *then* the API-key question becomes worth raising.

**Status: will do the manual D1+D2 proxy now.**

## Blocked until Phase 1 pilot finishes

- **Phase 2** (roll the winning text method out to all 5 segments) —
  don't know which method wins yet.
- **e5-base / bge-m3 decision** — want real MiniLM timing on the full
  141K-row residential set before committing to two larger, slower models
  on the same data (this is exactly why Phase 1 was staged instead of
  launched all at once).
- **Final Round 3 recommendation** — needs the full pilot picture, not
  partial data.
