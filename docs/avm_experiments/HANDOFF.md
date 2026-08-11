# HANDOFF — read this first in a new session

**Written:** 2026-08-09, right before the machine was shut down mid-run.
**Everything below the working directory note is self-contained** — a
fresh Claude Code session with no memory of the prior conversation should
be able to resume from this file alone.

## Where everything lives

```
C:\Users\KAMEN~1.DIM\AppData\Local\Temp\claude\d--Users-kamen-dimitrov-Desktop-SOFTUNI-AI-and-ML-upskill-program-Machine-learning-BG-real-estate-appraisal-helper\0df62287-de7b-4829-b0c4-b2a318cb3e2c\scratchpad\avm_experiments\
```

**Caveat:** this is a session-scoped temp path from a previous Claude Code
session — it is NOT this new session's own scratchpad. It should still
exist on disk (Windows doesn't auto-clean `AppData\Local\Temp` eagerly),
but there's a real risk it gets cleaned by the OS or a disk-cleanup tool
at some point. If you're reading this in a new session: verify the path
above still exists before trusting anything else in this file. If it's
gone, the raw JSON results are lost but the `.md` findings docs
summarizing them are the important part anyway, and this file + `INDEX.md`
describe what to redo.

The actual project (git repo, real code) is at:
```
d:\Users\kamen.dimitrov\Desktop\SOFTUNI\AI_and_ML_upskill_program\Machine_learning\BG_real_estate_appraisal_helper
```
All experiment scripts `sys.path.insert()` that project root and import
production modules (`app.db.base`, `scripts.train_avm_model`,
`utils.ml.avm_features`) directly — nothing here duplicates production
logic except where explicitly noted (e.g. one query in `phase1_pilot.py`
adds a `description_clean` column production doesn't need).

**Start by reading `INDEX.md`** in this same folder — it's the full map of
all 3 rounds, what's done, what's pending. This file only covers the
in-flight state at the moment of shutdown.

## What was mid-flight when the machine shut down

### 1. Track E1 — geocoding: COMPLETE (finished before shutdown)

`geocode_neighborhoods.py` ran to completion: **3,736/4,352 pairs resolved
(85.9%)**. `geocode_cache.json` has the full result. Nothing to resume
here.

**Track E2/E3 also done now** (`track_e2_e3.py`, output saved conceptually
here since it's fast to rerun if needed — `geo_clusters.json` has the
actual cluster assignments):
- Neighborhood-count coverage 85.8%, but **listing-weighted coverage only
  63.6%** — 36.4% of listings sit in neighborhoods that failed to geocode.
  The failures cluster around large, definitely-real Plovdiv/Varna
  districts (Христо Смирненски, Кършияка, Тракия, Виница, Бриз) — the
  simple `"{neighborhood}, {city}"` query format doesn't match them well.
  If Track E is pursued further, fixing the query format (try alternate
  spellings / a "ж.к." prefix / geo_1+geo_2+geo_3 from `location_raw`
  instead of just city+neighborhood) would meaningfully improve this
  before spending time on clustering quality.
- Sanity check caught 2 genuine bad geocodes: "запад, град монтана" →
  Croatia, "република 2, град софия" → Moldova. Both are real Nominatim
  mismatches (not legitimately-foreign listings) and should be excluded
  if this dataset gets used for real feature engineering.
- KMeans, K=25 chosen by silhouette (0.402). `geo_clusters.json` has the
  cluster assignments + centers.
- **E4 (test `geo_cluster_id` as an engineered feature) is still not
  done** — needs the Phase 1 pilot's winning text method decided first,
  per the original plan, so it's tested in the same experimental pass
  rather than as a one-off.

### 2. Round 3 Phase 1 pilot — MiniLM results were LOST to a real bug, now fixed

`phase1_pilot.py --methods keyword tfidf minilm` ran to completion but
**crashed while caching the MiniLM embeddings**, after already spending
**2.57 hours** computing them for residential (141K rows) and 75s for
hospitality. The crash discarded the in-memory results before they could
be used for the CV evaluation — a real, costly bug, not a data problem.

**Root cause:** `embed_cache/` sat one directory level deeper than
`avm_experiments/` itself, which was already close to Windows' 260-char
`MAX_PATH` limit given how long this temp path is. The extra nesting
tipped `residential_minilm.npy`'s full path over the limit; `np.save`
failed with a bare `FileNotFoundError`, not an obvious path-length error,
which is why it wasn't caught by the earlier smoke test (which never
exercised the caching code path — it called the embedding function
directly).

**Already fixed in `phase1_pilot.py`:**
1. `EMBED_CACHE_DIR` moved to `C:\Users\kamen.dimitrov\avm_embed_cache\` —
   short, flat, well clear of any length limit.
2. The `np.save` call is now wrapped in `try/except OSError` — a future
   cache-write failure will log a warning and continue with the in-memory
   result, never crash the run or lose completed computation again.

**What's safely saved from that run** (in `results_phase1_pilot.json`):
`baseline`, `keyword_flags`, `tfidf_raw`, `tfidf_norm` for both
`residential` and `hospitality`. **Do not recompute these** — rerun with
only `--methods minilm` to avoid redundant work (baseline gets recomputed
anyway since the script always includes it as the reference point, which
is fine/cheap and doubles as a consistency check):

```
cd "d:\Users\kamen.dimitrov\Desktop\SOFTUNI\AI_and_ML_upskill_program\Machine_learning\BG_real_estate_appraisal_helper"
python "<the long temp path above>\phase1_pilot.py" --methods minilm
```
Expect ~2.5-3h again for the embedding computation (residential dominates;
hospitality is seconds). The MiniLM model itself is already downloaded to
the local HuggingFace cache from the earlier run, so no re-download.

**Results already in hand (from the completed part of the crashed run) —
worth knowing before spending more compute:**

| Segment | Config | MAE | R² | ±10% |
|---|---|---|---|---|
| residential | baseline (no text) | 272.9 | 0.792 | 43.3% |
| residential | keyword flags | 267.5 | 0.799 | 44.0% |
| residential | **tfidf_raw/norm** | **261.2** | **0.808** | 45.0% |
| hospitality | baseline | 381.5 | 0.331 | 13.9% |
| hospitality | keyword flags | 378.0 | 0.342 | 14.1% |
| hospitality | **tfidf_raw/norm** | **371.2** | **0.352** | 15.1% |

**This alone is a meaningful finding regardless of what MiniLM shows**:
TF-IDF (cheapest possible text method) already beats the tuned
text-free baseline on both segments. Worth keeping in mind when deciding
whether the much more expensive MiniLM/e5-base/bge-m3 runs are worth it
at all, or whether TF-IDF alone is the pragmatic production answer.

## Everything else — stable, no action needed to "resume," just next steps

- **Round 1** (data cleaning, LightGBM tuning): complete, already applied
  to production. Nothing to do.
- **Round 2** (CatBoost + blend): complete, all 5 segments, findings in
  `ROUND2_findings.md`. **Decision pending**: whether to productionize.
  Design is ready in `TRACK_F_blend_productionization_design.md`
  (~half-day implementation estimate once decided) — not yet applied to
  the real repo.
- **Track G** (manual LLM-extraction feasibility pilot): complete, no
  further action unless you want to pursue a real API-based pilot (needs
  an LLM API key decision first — none configured in this project).
  Findings + a corrected attribute schema in `ROUND3_trackG_manual_pilot.md`.
- **MLflow**: `sqlite:///mlflow.db` in this folder, 48 runs across all 3
  rounds as of last check. `mlflow ui --backend-store-uri sqlite:///<this-folder>/mlflow.db`
  to browse. `backfill_mlflow.py` re-runnable any time (dedups by run
  name) to pick up new results as they land.

## Suggested priority order when resuming

1. Resume geocoding (cheap, ~43 min, just let it run).
2. Rerun `phase1_pilot.py --methods minilm` (the expensive one, ~2.5-3h —
   worth starting early in a session so it can run while doing other
   things).
3. While that runs: Track E2/E3 once geocoding finishes (fast, minutes).
4. Once MiniLM results are in: decide e5-base/bge-m3 (bigger, unknown
   timing) vs. calling it here with TF-IDF as the practical winner.
5. Independent of all the above, whenever convenient: decide on Round 2's
   CatBoost+blend productionization (design is ready, just needs a go/no-go).
