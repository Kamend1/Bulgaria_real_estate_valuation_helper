# Оценка на жилищни имоти — България

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-blue?logo=lightgbm&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Проект за machine learning анализ и оценка на пазарната стойност на жилищни имоти в България. Данните се набавят чрез автоматично събиране от imot.bg, обработват се с feature engineering pipeline и се подават на регресионен модел (LightGBM), предсказващ цена на квадратен метър. Резултатите се достъпват чрез FastAPI уеб приложение за оценки и сравнителен анализ.

---

## Съдържание

- [Архитектура](#архитектура)
- [Notebook pipeline](#notebook-pipeline)
- [FastAPI приложение](#fastapi-приложение)
- [Структура на проекта](#структура-на-проекта)
- [ML модел — статус и развитие](#ml-модел--статус-и-развитие)

---

## Архитектура

```text
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
| --- | --- |
| `data/taxonomy/valid_deal_types.csv` | `prodazhbi` / `naemi` |
| `data/taxonomy/valid_geo_paths.csv` | ~5 500 (deal_type, geo_path) комбинации |
| `data/taxonomy/valid_property_types.csv` | 22 типа имоти |

> Ако стартирате приложението, **не е нужно** да пускате Notebook 01 ръчно — `refresh_taxonomy()` се вика автоматично при всеки scrape run.

---

### Notebook 02 — Scrape pipeline

**`02_imot_bg_query_tool.ipynb`**

Зарежда taxonomy CSV-тата, строи `ScrapeSelection` от избраните deal types × geo paths × property types и стартира паралелен crawl.

**Ключови характеристики:**

- Паралелно обхождане на маршрути с `ThreadPoolExecutor` (max 12 workers)
- Crash-resumable checkpoint файлове — при прекъсване продължава от последния завършен маршрут
- HTML файловете се записват в `data/raw_listing_html/<url_hash>.html`
- Parsed данните се записват в `data/parsed_sales_runs/<run_id>/parsed_listings.csv`

**Изходни файлове:**

```text
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

Приложението интегрира scraping pipeline-а (Notebook 01 + 02) и предоставя пълен инструментариум за пазарен анализ и изготвяне на оценки на жилищни имоти. Данните се съхраняват в PostgreSQL.

### Начална страница (`/`)

Dashboard с обобщена статистика за базата данни в реално време:

- Брой активни обяви за продажба и наем
- Средна цена на квадратен метър (продажби / наеми)
- Разпределение по geo категория (топ 6 локации)
- Дата на последното засичане

---

### Scrape (`/scrape/`)

Управление на scrape run-ове — набавяне и актуализиране на данните.

**Поток при стартиране:**

```text
POST /scrape/start
  1. refresh_taxonomy()          ← imot.bg sitemap → 3 taxonomy CSV-та
  2. sync taxonomy → PostgreSQL  ← за UI филтри
  3. build route URLs            ← deal_types × geo_paths × property_types
  4. collect listing URLs        ← паралелен crawl, crash-resumable
  5. download + parse HTML       ← паралелен download, crash-resumable
  6. ingest → PostgreSQL         ← upsert listings + listing_snapshots
  7. archive stale listings      ← само при пълно национално покритие
```

**Функционалност:**

- Избор на deal types, geo пътища и типове имоти преди стартиране
- **Live прогрес в реално време** — SSE (Server-Sent Events) с фаза, брой маршрути, изтеглени и вписани обяви
- **Спиране** на активен run с незабавно убиване на процеса (`taskkill /F` на Windows)
- **Възобновяване** от последния checkpoint — не губи свършената работа
- **Guard** срещу паралелно стартиране на два run-а едновременно
- **История** на всички run-ове с резултати (`/scrape/history`)

Scrape-ът върви като **отделен OS процес** (subprocess) — независим от uvicorn. При рестарт на сървъра текущият run продължава, а стартиращата страница автоматично засича и показва активния run.

---

### Listings (`/listings/`)

Търсене и преглед на всички обяви в базата данни.

**Търсачка с филтри:**

- Deal type (продажба / наем)
- Тип имот (22 типа от taxonomy)
- Град (динамичен списък с брой обяви)
- Квартал (динамично зарежда се при избор на град)
- Geo категория (sofia_center, large_regional_city, sea_resort и др.)
- Ценови диапазон (мин/макс обща цена)
- Площ (мин/макс кв.м)
- Цена/кв.м (мин/макс)

**Сортиране:** последно виждан, цена ↑↓, цена/кв.м ↑↓, площ ↑↓

**Резултати:**

- Пагинация (50 на страница, HTMX partial reload)
- **Ценови тренд** за всяка обява: нова / поскъпнала / поевтиняла / непроменена (спрямо предишния snapshot)
- Select-all до 5 000 обяви за добавяне в comparable pool

**Детайлна страница** (`/listings/{id}`):

- Пълна информация за обявата (площ, етаж, конструкция, година, особености)
- **История на цените** — таблица с всички snapshots: дата, цена, цена/кв.м, площ, дни на пазара

---

### Comparables (`/comparables/`)

Инструмент за изготвяне на сравнителен пазарен анализ при оценки на имоти.

**Оценъчен доклад (draft):**

- Описание на оценявания обект: адрес, град, площ, етаж/общо, конструкция, година, описание, дата на оценката
- Поддържа множество доклади; нов draft зачиства comparable pool-а

**Comparable pool — продажни и наемни сравними:**

- Добавяне на обяви директно от търсачката (единично или select-all)
- Премахване на отделни записи или изчистване на целия pool
- **Pinning** — маркиране на до 6 сравними на тип за включване в доклада
- **Корекция в %** и аналитична бележка на всеки comparable (напр. -5% за лошо изложение)
- Автоматично изчисляване на коригирана цена/кв.м след корекцията

**Статистики на pool-а** (изчислени в реално време):

- Брой сравними, мин/ср/макс цена и площ
- Цена/кв.м: мин, средна, медиана, Q25, Q75, макс

**Excel export** (`/comparables/export/excel`):

- Стилизиран `.xlsx` файл с отделен worksheet за продажни и наемни сравними
- Pinned обяви са маркирани в зелено; оценяваният обект — в жълто
- Статистически ред с обобщение на пула
- Auto-width на колоните

---

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
| --- | --- |
| `scrape_runs` | Всеки scrape run с прогрес, статус, PID, heartbeat |
| `listings` | Обяви — upsert key: `ad_url`; статус: active / archived |
| `listing_snapshots` | Append-only ценова история на всяка обява |
| `taxonomy_geo_paths` | Валидни geo пътища от sitemap |
| `taxonomy_property_types` | 22 типа имоти с Bulgarian display names |
| `comparable_pool` | Пул от сравними: корекции, pins, бележки |
| `appraisal_reports` | Оценъчни доклади с данни за оценявания обект |

---

## Структура на проекта

```text
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

Настоящата ML реализация (Notebook 04) е **базова версия** и предстои активно развитие.

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
