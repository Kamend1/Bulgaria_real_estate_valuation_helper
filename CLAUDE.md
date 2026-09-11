# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bulgarian residential real estate ML project: scrapes imot.bg, parses listings, and builds a LightGBM regression model to predict price per square meter. The workflow is split across four sequential Jupyter notebooks.

## Running the notebooks

Notebooks live in `notebooks/` and must be run from that directory so that `Path.cwd().parent` resolves correctly to the project root. Launch Jupyter from that directory:

```
cd notebooks
jupyter notebook
```

Notebooks add the project root to `sys.path` themselves — no install step needed.

## Notebook pipeline (run in order)

| Notebook | Purpose |
|---|---|
| `01_imot_bg_scraper_tool.ipynb` | Fetches imot.bg sitemap, builds taxonomy CSVs (`valid_deal_types.csv`, `valid_geo_paths.csv`, `valid_property_types.csv`) |
| `02_imot_bg_query_tool.ipynb` | Crawls listing URLs in parallel, downloads raw HTML, parses listings, saves to `data/parsed_sales_runs/<run_id>/` |
| `03_EDA_and_feature_engineering.ipynb` | EDA, feature engineering, produces `imot_ml_ready.parquet` |
| `04_residential_real_estate_regression_ML_analysis.ipynb` | LightGBM regression: `price_per_sqm` prediction, GridSearchCV, error analysis |

In notebooks 03 and 04, set `LATEST_RUN` manually to the name of the latest timestamped run directory under `data/parsed_sales_runs/`:

```python
LATEST_RUN = "parsed_sales_full_20260608_081646"
```

## Architecture

### `utils/fetch_data/fetch_data_utils.py`
Web scraping layer. Key components:
- `ScrapeSelection` — builds imot.bg search URLs from deal type × geo path × property type combinations
- `collect_listing_urls_for_routes_parallel_streaming` — parallel crawl across routes with immediate CSV checkpointing (crash-resumable)
- `download_and_parse_listing_batch_streaming` — parallel download + parse with buffered CSV flush every 300 rows
- `load_valid_*` functions — load taxonomy CSVs produced by notebook 01

### `utils/ad_parsing/ad_parsing_utils.py`
HTML parsing layer. Key components:
- `parse_imot_listing` — main engine: reads saved HTML, returns one structured dict per listing
- Parsers for price (EUR/BGN), area (кв.м), floor, construction type/year, features, description, title metadata
- `classify_listing` — distinguishes `single_property_listing` from `new_building_project`

### Data layout

```
data/
  taxonomy/               # Valid selectors for scraping (from notebook 01)
  raw_listing_html/       # Downloaded HTML files, named by URL hash
  parsed_sales_runs/
    parsed_sales_full_<timestamp>/
      route_results.csv   # One row per crawled route (resumable checkpoint)
      page_results.csv    # One row per tested page
      listing_urls_raw.csv
      listing_urls_unique.csv
      download_manifest.csv
      parsed_listings.csv
      parsed_listings.parquet
      imot_ml_ready.parquet  # Feature-engineered, ML-ready (from notebook 03)
```

The `data/` directory is gitignored.

## Key implementation details

- imot.bg pages are **Windows-1251 encoded** — all HTML fetches decode with `content.decode("windows-1251", errors="replace")`
- Crawlers are **crash-resumable**: they read already-completed route/listing URLs from existing CSV files and skip them on restart
- `ThreadPoolExecutor` is used for parallel scraping; `max_workers=12` for route discovery, `max_workers=24` for download+parse
- Listing HTML is saved to `data/raw_listing_html/<url_hash>.html` (SHA-256 of URL, first 16 hex chars)
- The ML target is `price_per_sqm`; only `single_property_listing` rows with non-null `total_price` and `area_sqm` are `training_eligible`
- Features pipe-separated in `features_pipe` column (e.g. `"Асансьор|Гараж|ВЕЦ"`)

## ML model

LightGBM (`LGBMRegressor`) via sklearn `Pipeline` + `ColumnTransformer`. `GridSearchCV` with a `PredefinedSplit` (train/val) is used for hyperparameter search. The notebook evaluates accuracy bands (within 5%, 10%, etc.) and error breakdowns by property type, city, and price band.

---

## Appraisal App (Phases 1-7 complete)

The project has grown from the Phase 1 foundation into a full multi-user appraisal
tool: auth/roles, a segment-aware AVM (LightGBM+CatBoost, R2-backed model loading),
GIS/cadastre integration for Sofia, a full comparables workflow (three value
approaches + weighted conclusion + structured adjustments), Word/Excel export, and
AI-assisted valuation via a pgvector + LangChain RAG pipeline (multi-provider:
OpenAI/Anthropic/Google). **README.md is the up-to-date feature reference** — its
Съдържание/table of contents covers every module in detail; don't assume this file's
older summaries below are exhaustive, treat README.md as authoritative on scope.

### Running the app (local)

Prerequisites: PostgreSQL 16+ with the `vector` extension (pgvector) — migration
0019 runs `CREATE EXTENSION vector`. `docker-compose.yml`'s `db` service uses
`pgvector/pgvector:pg16` for this reason (not plain `postgres:16`).

```
# 1. Start DB
docker-compose up db -d

# 2. Copy env
cp .env.example .env   # edit DATABASE_URL if needed; OPENAI_API_KEY/etc. optional (RAG only)

# 3. Install deps
pip install -r requirements.txt

# 4. Run migrations
alembic upgrade head

# 5. Import historical data (one-time, ~164K rows)
python -m scripts.import_historical_data

# 6. Optional: embedding backfill for AI-assisted valuation (real OpenAI cost)
python -m scripts.embed_listings

# 7. Start app
uvicorn app.main:app --reload
```

App is at `http://localhost:8000`. Navigate to `/scrape/` to trigger a new scrape.
On Windows, `start_app.bat` starts it on port 8891 — note it only checks whether
something is already listening on that port and opens the browser if so, it does
**not** detect a stale/pre-edit process and restart it; kill the old process
manually first if you've changed code and need the new version to actually run.

### New module: `utils/feature_engineering/feature_engineering_utils.py`

All feature engineering extracted from notebook 03, applied to every row before DB insertion:
- `engineer_features(parsed_row: dict) -> dict` — master function
- `parse_published_date(published_raw)` → `date | None` (parses Bulgarian: "31 яну, 2014")
- `map_geo_category(l1, l2, l3, city, geo2)` → category string (sofia_center / sofia_other / large_regional_city / regional_city / small_city / sea_resort / mountain_resort / other_unknown / foreign)
- `correct_agricultural_area` — converts ЗЕМЕДЕЛСКА ЗЕМЯ from decares to sqm when area ≤ 500
- `normalize_deal_type("Продава")` → `"sale"` / `"Дава под наем"` → `"rent"`

### App structure

```
app/
  main.py              # FastAPI app factory; mounts /static, includes routers,
                       #   CSRF + auth-attach + rate-limit middleware
  config.py            # pydantic-settings (reads .env)
  db/
    base.py            # create_engine, SessionLocal, Base
    session.py         # get_db() FastAPI dep + db_session() context manager for threads
    models.py          # ORM models: ScrapeRun, Listing, ListingSnapshot,
                       #   ComparablePool, ReportComparable, AppraisalReport,
                       #   AvmModel, ListingEmbedding, AiValuationRun, User,
                       #   UserConsent. listing_price_events is a real table
                       #   (alembic/versions/0007_analytics.py) but has no
                       #   ORM model -- analytics_service.py accesses it via
                       #   raw SQL only.
  routers/
    auth.py, admin.py, scrape.py, listings.py, analytics.py,
    comparables.py     # subject form, AVM/GIS panels, comparable pool, AI
                       #   suggestions/generation/history, approaches, export
    reports.py
  services/
    scrape_service.py      # shared helpers used by scripts/run_scrape.py:
                            #   _ingest_rows_to_db/_ingest_csv_to_db, sync_taxonomy_to_db,
                            #   get_scrape_status -- NOT an orchestrator itself, see
                            #   "Scrape pipeline and progress" below
    avm_retrain_service.py # maybe_retrain_avm_models -- called from scripts/run_scrape.py
    listing_service.py     # search_listings filters (incl. construction_year/floor)
    analytics_service.py   # mv_analytics_flat aggregation, market trend
    avm_service.py, gis_service.py, comparable_service.py
    llm/                    # Phase 7 RAG: providers.py (chat model factory,
                            #   3 tiers x 3 providers), embeddings.py,
                            #   listing_doc.py (text serialization),
                            #   retriever.py (hybrid SQL-filter + pgvector search),
                            #   tools.py (bound tool-calling functions),
                            #   valuation_chain.py (generation + guardrails),
                            #   embed_backfill.py (staleness-aware backfill,
                            #   called both by scripts/embed_listings.py and
                            #   automatically at the end of every scrape run)
  templates/
    base.html          # Jinja2 base with navbar + HTMX CDN + CSRF meta tag
    listings/           # search.html, _results.html, detail.html
    comparables/        # panels: _avm_panel, gis panels, _pool_panel,
                       #   _ai_suggestions_panel, _ai_generation_result,
                       #   _ai_history, _conclusion_panel, _income_analysis, ...
scripts/
  run_scrape.py           # THE real scrape pipeline -- launched as its own OS
                          #   process via subprocess.Popen from app/routers/scrape.py;
                          #   see "Scrape pipeline and progress" below
  import_historical_data.py, train_avm_model.py, embed_listings.py,
  backup_to_r2.py, prune_old_models.py, lookup_parcel.py, create_admin.py,
  chunk_legal_documents.py,  # one-off backfill: embed existing legal_standard
                             #   market_documents into legal_document_chunks
  recover_scrape_run.py   # one-off: retry a scrape run's failed downloads + re-ingest
alembic/
  versions/            # see the directory for the current head -- 0019 adds
                       #   pgvector + listing_embeddings/ai_valuation_runs;
                       #   this list is intentionally not kept in lockstep
                       #   with every new migration, don't infer "latest" from it
static/app.css
```

### DB schema key points

- `listings.ad_url` is UNIQUE — upsert key
- `first_seen_at` / `published_date` are never overwritten on upsert
- `listing_snapshots` is append-only: every scrape adds a row even if price unchanged
- `days_on_market` in snapshots = `scraped_at::date - published_date`
- All monetary/area columns use `NUMERIC` not `FLOAT`
- `geo_category` and `deal_type_normalized` ("sale"/"rent") are indexed for fast filtering; a partial composite index (`property_type_slug, geo_category, last_seen_at DESC WHERE status='active'`) backs the listings search page specifically
- `listing_embeddings` is a separate table (not a `listings` column), keyed by `(listing_id, provider, model)` — supports re-embedding with a different model without a schema change; has an HNSW index (`vector_cosine_ops`)

### Scrape pipeline and progress (corrected 2026-09-11 — see note below)

A real scrape runs as a **separate OS process**, not a thread: `app/routers/scrape.py` launches it with `subprocess.Popen([sys.executable, "-m", "scripts.run_scrape", "--run-id", run_id])`. `scripts/run_scrape.py` is the actual pipeline — taxonomy refresh → route discovery → download+parse → DB ingest → archive stale listings → analytics refresh (`compute_price_events`/`refresh_mv`) → embeddings backfill (`embed_backfill.backfill_embeddings`) → AVM retrain (`avm_retrain_service.maybe_retrain_avm_models`) — writing its own progress directly to `scrape_runs` columns (`phase`, `last_message`, `listings_ingested`, `last_heartbeat_at`, ...) via raw SQL `UPDATE`s, plus a dedicated heartbeat thread so a dead process is detectable even with no recent log line. `GET /scrape/progress/{run_id}` is a **DB-polling** SSE endpoint (`app/routers/scrape.py::progress_sse`) — it re-queries `scrape_runs` every second and streams the row as the event; it does not read from any in-process state, so it "survives uvicorn restarts" by design.

> **Historical note:** `app/services/scrape_service.py` used to also define `run_scrape_background()` — an independent, thread-based copy of this same pipeline (with its own `ProgressCapture`/in-memory `progress_store` push mechanism) that **no code anywhere ever called**. Two different "runs automatically on every scrape" features (embeddings backfill, then AVM retraining) were each added there first, by reasonable-looking inference from a docstring exactly like the one this section used to have — and each one silently never ran on a real scrape as a result, discovered only much later via a live audit. `run_scrape_background()` and its dead support code have since been deleted. If you're about to add a new "do X automatically after every scrape" step, it belongs in `scripts/run_scrape.py`, the file described above — verify by checking `app/routers/scrape.py`'s launch call still points there.
