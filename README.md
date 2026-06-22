# Оценка на жилищни имоти — България

Проект за machine learning анализ и оценка на пазарната стойност на жилищни имоти в България. Данните се набавят чрез автоматично събиране от imot.bg, обработват се с feature engineering pipeline и се подават на регресионен модел (LightGBM), предсказващ цена на квадратен метър. Резултатите се достъпват чрез FastAPI уеб приложение за оценки и сравнителен анализ.

---

## Съдържание

- [Архитектура](#архитектура)
- [Notebook pipeline](#notebook-pipeline)
- [FastAPI приложение](#fastapi-приложение)
- [Инсталация и стартиране](#инсталация-и-стартиране)
- [Структура на проекта](#структура-на-проекта)
- [ML модел — статус и развитие](#ml-модел--статус-и-развитие)

---

## Архитектура

```
imot.bg
   │
   ▼
Notebook 01          Notebook 02
Taxonomy discovery   Scrape pipeline
(sitemap → CSVs)     (routes → HTML → parsed CSV)
   │                      │
   └──────────┬───────────┘
              │
        Notebook 03
        EDA + Feature engineering
        (imot_ml_ready.parquet)
              │
        Notebook 04
        LightGBM regression
        (price_per_sqm prediction)

              │
              ▼
        FastAPI app  ◄──── scrape_run subprocess
        PostgreSQL         (Notebook 01 + 02 логика,
                           интегрирана в app)
```

Notebooks и приложението споделят `utils/` — scraping, parsing и feature engineering са реализирани като library code, не като notebook-specific логика.

---

## Notebook pipeline

Notebooks се намират в `notebooks/` и трябва да се стартират от тази директория, за да се резолвира `Path.cwd().parent` правилно към project root-а.

```bash
cd notebooks
jupyter notebook
```

### Notebook 01 — Taxonomy discovery

**`01_imot_bg_scraper_tool.ipynb`**

Fetch-ва sitemap index-а на imot.bg, обхожда всички child sitemaps и извлича валидните `/obiavi/` URL-и. Резултатът са три taxonomy CSV-та, необходими за scraping-а:

| Файл | Съдържание |
|---|---|
| `data/taxonomy/valid_deal_types.csv` | `prodazhbi` / `naemi` |
| `data/taxonomy/valid_geo_paths.csv` | ~5 500 (deal_type, geo_path) комбинации |
| `data/taxonomy/valid_property_types.csv` | 22 типа имоти |

> Ако стартирате приложението, **не е нужно** да пускате Notebook 01 ръчно — `refresh_taxonomy()` се вика автоматично при всеки scrape run.

---

### Notebook 02 — Scrape pipeline

**`02_imot_bg_query_tool.ipynb`**

Зарежда taxonomy CSV-тата, строи `ScrapeSelection` от избраните deal types × geo paths × property types и стартира паралелен crawl.

**Ключови характеристики:**
- Параллелно обхождане на маршрути с `ThreadPoolExecutor` (max 12 workers)
- Crash-resumable checkpoint файлове — при прекъсване продължава от последния завършен маршрут
- HTML файловете се записват в `data/raw_listing_html/<url_hash>.html`
- Parsed данните се записват в `data/parsed_sales_runs/<run_id>/parsed_listings.csv`

**Изходни файлове:**

```
data/parsed_sales_runs/<run_id>/
  route_results.csv         # един ред на маршрут (checkpoint)
  page_results.csv          # един ред на тествана страница
  listing_urls_raw.csv      # всички намерени URL-и
  listing_urls_unique.csv   # дедупликирани
  download_manifest.csv     # статус на всеки HTML download
  parsed_listings.csv       # парснати обяви
  parsed_listings.parquet
```

В началото на Notebook 02 задайте `LATEST_RUN` ръчно:

```python
LATEST_RUN = "parsed_sales_full_20260608_081646"
```

**imot.bg специфики:**
- Страниците са кодирани в **Windows-1251** — всички fetch-ове декодират с `content.decode("windows-1251", errors="replace")`
- Глобален семафор ограничава едновременните HTTP заявки до 8

---

### Notebook 03 — EDA и Feature Engineering

**`03_EDA_and_feature_engineering.ipynb`**

Exploratory Data Analysis и трансформации върху parsed данните. Произвежда `imot_ml_ready.parquet` — готов за ML датасет.

**Основни трансформации:**
- Нормализация на deal type (`Продава` → `sale`, `Дава под наем` → `rent`)
- Парсване на дата на публикуване от български формат (`"31 яну, 2014"`)
- Geo категоризация: `sofia_center` / `sofia_other` / `large_regional_city` / `regional_city` / `small_city` / `sea_resort` / `mountain_resort` / `other_unknown` / `foreign`
- Корекция на земеделска земя (декари → кв.м при площ ≤ 500)
- Изчисляване на `price_per_sqm_model` от `total_price / area_sqm_model`
- Флаг `training_eligible`: само `single_property_listing` с ненулева цена и площ

Целият feature engineering е екстрахиран в `utils/feature_engineering/feature_engineering_utils.py` и се прилага и в приложението при всяка ingested обява.

---

### Notebook 04 — ML регресионен анализ

**`04_residential_real_estate_regression_ML_analysis.ipynb`**

LightGBM регресия за предсказване на `price_per_sqm`.

**Методология:**
- `LGBMRegressor` в sklearn `Pipeline` с `ColumnTransformer`
- `GridSearchCV` с `PredefinedSplit` (train / validation без data leakage)
- Target: `price_per_sqm` (EUR/кв.м)

**Анализ на резултатите:**
- Accuracy bands: в рамките на ±5%, ±10%, ±15%, ±20%
- Error breakdown по тип имот, град и ценови диапазон
- Feature importance

> **ML частта ще търпи активно развитие** — вижте [ML модел — статус и развитие](#ml-модел--статус-и-развитие).

---

## FastAPI приложение

Приложението интегрира scraping pipeline-а (Notebook 01 + 02) и го прави достъпен чрез уеб интерфейс. Данните се съхраняват в PostgreSQL.

### Функционалност

- **Scrape**: стартиране, наблюдение и спиране на scrape run с live progress (SSE)
- **Listings**: търсене и преглед на обяви
- **Comparables**: инструмент за сравнителен анализ при оценки

### Scrape run — поток

```
POST /scrape/start
  1. refresh_taxonomy()          ← imot.bg sitemap → 3 taxonomy CSV-та
  2. sync taxonomy → PostgreSQL  ← за UI филтри
  3. build route URLs            ← deal_types × geo_paths × property_types
  4. collect listing URLs        ← паралелен crawl, crash-resumable
  5. download + parse HTML       ← паралелен download, crash-resumable
  6. ingest → PostgreSQL         ← upsert listings + listing_snapshots
  7. archive stale listings      ← при пълно национално покритие
```

Scrape-ът върви като **отделен OS процес** (subprocess) — независим от uvicorn. При рестарт на сървъра текущият run продължава.

### Стартиране (локално)

**Изисквания:** PostgreSQL (локален или Docker)

```bash
# 1. Стартиране на базата данни
docker-compose up db -d

# 2. Конфигурация
cp .env.example .env
# Редактирайте DATABASE_URL ако е необходимо

# 3. Зависимости
pip install -r requirements.txt

# 4. Миграции
alembic upgrade head

# 5. Еднократен импорт на исторически данни (опционален)
python -m scripts.import_historical_data

# 6. Стартиране
uvicorn app.main:app --reload
```

Приложението е достъпно на `http://localhost:8000`.

### DB схема (основни таблици)

| Таблица | Описание |
|---|---|
| `scrape_runs` | Всеки scrape run с прогрес, статус, PID |
| `listings` | Обяви — upsert key: `ad_url` |
| `listing_snapshots` | Append-only история на цените |
| `taxonomy_geo_paths` | Валидни geo пътища от sitemap |
| `taxonomy_property_types` | 22 типа имоти |
| `comparable_pool` | Пул от сравними имоти за оценки |
| `appraisal_reports` | Генерирани оценъчни доклади |

---

## Структура на проекта

```
├── notebooks/
│   ├── 01_imot_bg_scraper_tool.ipynb
│   ├── 02_imot_bg_query_tool.ipynb
│   ├── 03_EDA_and_feature_engineering.ipynb
│   └── 04_residential_real_estate_regression_ML_analysis.ipynb
│
├── utils/
│   ├── fetch_data/fetch_data_utils.py      # scraping, taxonomy refresh
│   ├── ad_parsing/ad_parsing_utils.py      # HTML parsing на обяви
│   └── feature_engineering/               # feature engineering (споделен с app)
│
├── app/                                    # FastAPI приложение
│   ├── main.py
│   ├── config.py
│   ├── db/                                # SQLAlchemy models + session
│   ├── routers/                           # scrape, listings, comparables
│   ├── services/                          # бизнес логика
│   ├── templates/                         # Jinja2 HTML
│   └── progress/                          # SSE progress store
│
├── scripts/
│   ├── run_scrape.py                      # standalone scrape subprocess
│   └── import_historical_data.py          # еднократен import от parquet
│
├── alembic/                               # DB миграции
├── static/                                # CSS
├── data/                                  # gitignored — данни
├── outputs/                               # gitignored — генерирани доклади
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── alembic.ini
└── .env.example
```

---

## ML модел — статус и развитие

Настоящата ML реализация (Notebook 04) е **базова версия** и предстои активно развитие:

**Текущо състояние:**
- LightGBM регресия с `GridSearchCV`
- Features: тип имот, локация (geo категория), площ, етаж, конструкция, година
- Трениран на исторически snapshot (~165 000 обяви)

**Планирано развитие:**
- Автоматично пре-трениране при всеки scrape run
- Включване на времеви features (дни на пазара, сезонност)
- По-гранулирана geo репрезентация (квартал-ниво)
- Explainability (SHAP) интеграция в оценъчните доклади
- A/B сравнение с допълнителни алгоритми (XGBoost, CatBoost)
- API endpoint за директно scoring на нова обява

> Моделът в момента не е интегриран в приложението — оценките се базират на пазарни сравними, не на ML prediction. Интеграцията е следваща фаза.

---

## Лиценз

[LICENSE](LICENSE)
