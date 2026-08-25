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
                       #   ListingPriceEvent, ComparablePool, ReportComparable,
                       #   AppraisalReport, AvmModel, ListingEmbedding,
                       #   AiValuationRun, User, UserConsent
  routers/
    auth.py, admin.py, scrape.py, listings.py, analytics.py,
    comparables.py     # subject form, AVM/GIS panels, comparable pool, AI
                       #   suggestions/generation/history, approaches, export
    reports.py
  services/
    scrape_service.py      # ProgressCapture, run_scrape_background, _ingest_rows_to_db
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
  import_historical_data.py, train_avm_model.py, embed_listings.py,
  backup_to_r2.py, prune_old_models.py, lookup_parcel.py, create_admin.py
alembic/
  versions/            # 0001 initial schema ... 0021 (latest) perf indexes;
                       #   0019 adds pgvector + listing_embeddings/ai_valuation_runs
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

### SSE scrape progress

Background thread (`threading.Thread(daemon=True)`) runs the scrape. `ProgressCapture` wraps stdout and parses print lines from the existing utils with regex to extract route/listing counts. `GET /scrape/progress/{run_id}` streams SSE events to the browser's `EventSource`. Thread is NOT a FastAPI BackgroundTask (timeouts). At the end of a run it also runs analytics refresh + embedding backfill (both non-fatal — a failure there logs a warning but doesn't fail the scrape).
