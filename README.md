# Дигитален асистент на имотния оценител

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-blue?logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00?logo=&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)

Пълноценно уеб приложение за оценители на недвижими имоти в България — от набавяне на пазарни данни до готов оценителски доклад. В основата стои автоматизиран pipeline, който събира обяви от imot.bg, обработва ги и захранва **сегментен ансамбъл от ML модели** (LightGBM + CatBoost, 5 пазарни сегмента) за автоматична оценка на цена на кв.м. Върху това е изграден пълен работен процес за оценителя: пазарни сравними с корекции, доходен и остатъчен подход, **интеграция с кадастъра и устройственото планиране** (АГКК, ГИС София, НАГ София), потребители/роли и Word/Excel export на готовия доклад.

---

## Съдържание

- [Архитектура](#архитектура)
- [Notebook pipeline (произход на данните и модела)](#notebook-pipeline-произход-на-данните-и-модела)
- [FastAPI приложение](#fastapi-приложение)
- [AVM — автоматизиран модел за оценка](#avm--автоматизиран-модел-за-оценка)
- [Кадастър и устройствено планиране (GIS модул)](#кадастър-и-устройствено-планиране-gis-модул)
- [Стартиране (локално)](#стартиране-локално)
- [DB схема](#db-схема)
- [Структура на проекта](#структура-на-проекта)
- [Статус и посоки за развитие](#статус-и-посоки-за-развитие)
- [Лиценз](#лиценз)

---

## Архитектура

```text
                              imot.bg
                                 │
                    ┌────────────┴────────────┐
                    ▼                          ▼
            Notebook 01                  Notebook 02
            Taxonomy discovery           Scrape pipeline
            (sitemap → CSVs)             (routes → HTML → parsed CSV)
                    │                          │
                    └────────────┬─────────────┘
                                 │
                          PostgreSQL  ◄──── scrape_run subprocess (в приложението)
                                 │
              ┌──────────────────┼───────────────────────┐
              ▼                  ▼                        ▼
     Notebook 03/04       utils/ml (AVM)           utils/gis (кадастър)
     EDA + feature eng.   LightGBM + CatBoost       АГКК · isofmap.bg · НАГ София
     (прототип, Jupyter)  5 сегмента, MLflow         (свободни, но неофициални API)
              │                  │                        │
              └──────────────────┴───────────┬────────────┘
                                              ▼
                                       FastAPI приложение
                                architecture: routers → services → db
                     auth · scrape · listings · analytics · comparables
                          · reports · admin (AVM registry, users)
```

Notebooks и приложението споделят `utils/` — scraping, HTML parsing, feature engineering, ML feature definитions и GIS клиентите са library code, не notebook-specific логика. `utils/gis/` и `utils/ml/` са писани директно за приложението (не произлизат от notebook), но следват същия принцип на споделен код.

---

## Notebook pipeline (произход на данните и модела)

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

### Notebook 02 — Scrape pipeline

**`02_imot_bg_query_tool.ipynb`**

Зарежда taxonomy CSV-тата, строи `ScrapeSelection` от избраните deal types × geo paths × property types и стартира паралелен crawl.

**Ключови характеристики:**

- Паралелно обхождане на маршрути с `ThreadPoolExecutor`
- Crash-resumable checkpoint файлове — при прекъсване продължава от последния завършен маршрут
- HTML файловете се записват в `data/raw_listing_html/<url_hash>.html`
- Parsed данните се записват в `data/parsed_sales_runs/<run_id>/parsed_listings.csv`

**imot.bg специфики:**

- Страниците са кодирани в **Windows-1251** — всички fetch-ове декодират с `content.decode("windows-1251", errors="replace")`

### Notebook 03 — EDA и Feature Engineering

**`03_EDA_and_feature_engineering.ipynb`**

Exploratory Data Analysis и трансформации върху parsed данните. Произвежда `imot_ml_ready.parquet`.

**Основни трансформации** (екстрахирани в `utils/feature_engineering/feature_engineering_utils.py` и приложени и в приложението при всяка ingested обява):

- Нормализация на deal type (`Продава` → `sale`, `Дава под наем` → `rent`)
- Парсване на дата на публикуване от български формат (`"31 яну, 2014"`)
- Geo категоризация: `sofia_center` / `sofia_other` / `large_regional_city` / `regional_city` / `small_city` / `sea_resort` / `mountain_resort` / `other_unknown` / `foreign`
- Корекция на земеделска земя (декари → кв.м при площ ≤ 500)
- Флаг `training_eligible`: само `single_property_listing` с ненулева цена и площ

### Notebook 04 — оригинален ML прототип

**`04_residential_real_estate_regression_ML_analysis.ipynb`**

Първоначален LightGBM регресионен анализ (само жилищни имоти) — `GridSearchCV` върху sklearn `Pipeline` + `ColumnTransformer`, target `price_per_sqm`. Този notebook е **историческата отправна точка** на моделирането: методологията му (data cleaning, target transform, stratified split) е пренесена и разширена в `scripts/train_avm_model.py`, който сега тренира **5 сегмента** (не само жилищни) и е интегриран директно в приложението — вижте [AVM секцията](#avm--автоматизиран-модел-за-оценка).

---

## FastAPI приложение

Пълноценно, multi-user приложение с автентикация, роли и цялостен workflow за оценка на имот — от пазарно проучване до финализиран доклад.

### Автентикация (`/auth/`)

- Регистрация с приемане на Политика за поверителност и Общи условия (версионирани consent записи в `user_consents`)
- Login / logout, потребителски профил
- Роли `user` / `admin`; имейл, зададен в `ADMIN_EMAIL`, автоматично получава администраторски права при регистрация
- Всички защитени страници пренасочват към `/auth/login?next=...` при липса на сесия

### Начална страница (`/`)

Dashboard с обобщена статистика за базата данни в реално време: брой активни обяви (продажби/наеми), средна цена на кв.м, разпределение по geo категория, дата на последното засичане.

### Scrape (`/scrape/`)

Управление на scrape run-ове — набавяне и актуализиране на пазарните данни.

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

- Избор на deal types, geo пътища и типове имоти преди стартиране
- **Live прогрес** — SSE (Server-Sent Events) с фаза, брой маршрути, изтеглени и вписани обяви
- **Спиране** на активен run с незабавно убиване на процеса; **guard** срещу паралелно стартиране на два run-а
- Scrape-ът върви като **отделен OS процес** — независим от uvicorn; при рестарт на сървъра текущият run продължава, а страницата автоматично засича активния run
- **История** на всички run-ове (`/scrape/history`)

### Listings (`/listings/`)

Търсене и преглед на всички обяви в базата данни, с филтри (deal type, тип имот, град, квартал, geo категория, цена, площ, цена/кв.м), сортиране и пагинация (HTMX partial reload).

- **Ценови тренд** за всяка обява: нова / поскъпнала / поевтиняла / непроменена
- Select-all до 5 000 обяви за добавяне в comparable pool
- Детайлна страница (`/listings/{id}`) с пълна история на цените (всички snapshots)

### Analytics (`/analytics/`)

Агрегиран пазарен анализ върху натрупаната история от snapshots (`listing_price_events`, материализиран изглед `mv_analytics_flat`):

- Динамика на цена/кв.м във времето, филтрируема по deal type, geo категория, град, квартал, тип имот, брой засичания
- Статистика за повторно обявени/премахнати имоти

### Comparables (`/comparables/`) — сърцето на оценителския workflow

Инструмент за изготвяне на пълен пазарен анализ при оценка на имот, обединяващ три подхода за стойност плюс автоматична AVM прогноза:

**Оценяван обект** — адрес, град, площ, етаж, конструкция, година, описание, дата на оценка, вид имот (за AVM сегмента), гео-категория, квартал, **кадастрален идентификатор** (за GIS модула).

**Пазарен подход (AVM панел)** — сегментна LightGBM/CatBoost прогноза за цена на кв.м с интервал на доверие; вижте [AVM секцията](#avm--автоматизиран-модел-за-оценка).

**Кадастър и правно описание (GIS панел)** — данни от АГКК, генерирано правно описание на имота, скица, устройствена зона и свързани преписки; вижте [GIS секцията](#кадастър-и-устройствено-планиране-gis-модул).

**Comparable pool — продажни и наемни сравними:**

- Добавяне на обяви директно от търсачката (единично или select-all), премахване, изчистване на pool-а
- **Pinning** — маркиране на до 6 сравними на тип за включване в доклада
- **Корекция в %** и аналитична бележка на всеки comparable, с автоматично изчисляване на коригирана цена/кв.м
- Статистики на pool-а в реално време: брой, мин/ср/макс цена и площ, цена/кв.м (мин, средна, медиана, Q25, Q75, макс)

**Доходен подход** — наем на кв.м/месец, капитализационен процент → заключена стойност.

**Остатъчен подход** — ръчно въведена заключена стойност (за нестандартни случаи/проекти).

**Export:**

- **Excel** (`/comparables/export/excel`) — стилизиран `.xlsx`, отделен worksheet за продажни/наемни сравними, pinned обяви маркирани в зелено, оценяваният обект — в жълто, статистически ред, auto-width колони
- **Word** (`/comparables/export/docx`) — пълен оценителски доклад по шаблон (`python-docx`), включващ всички подходи и сравними таблици

### Reports (`/reports/`)

Управление на множество оценителски доклади на потребителя: списък, отваряне на активен доклад, финализиране (`draft` → `finalized`), повторно отваряне за редакция, изтриване.

### Admin (`/admin/`, само за роля `admin`)

- **Потребители** — списък с търсене, детайл, смяна на роля, деактивиране, ръчна смяна на парола, създаване на нов потребител, изтриване
- **AVM модели** (`/admin/avm`) — регистър на всички трениран модели по сегмент с метрики; **ръчно стартиране на retraining** (за всички сегменти или за конкретен), изпълнявано като фонов процес

---

## AVM — автоматизиран модел за оценка

Сегментен ансамбъл, а не един модел за всички типове имоти — цените на офис, търговски, индустриален и хотелиерски имот се движат от различни фактори спрямо жилищния пазар.

### 5 пазарни сегмента

| Сегмент | Типове имоти (taxonomy slugs) |
| --- | --- |
| Жилищни имоти | едностаен, двустаен, тристаен, четиристаен, многостаен, мезонет, ателие/таван, етаж от къща, къща, вила, стая |
| Офиси | офис |
| Търговски имоти | магазин, заведение, бизнес-имот |
| Индустриални имоти | склад, промишлено помещение |
| Хотелиерски имоти | хотел |

Всеки сегмент има собствен ред в `avm_models` — собствени хиперпараметри, feature set и статус `is_active`, независим от останалите сегменти.

### Методология

- **LightGBM** (`LGBMRegressor`) като основен модел за всеки сегмент, в sklearn `Pipeline` + `ColumnTransformer`; хиперпараметрите са тунирани индивидуално по сегмент (не споделен GridSearchCV)
- **CatBoost** като допълващ модел за 4 от 5-те сегмента (не за жилищни, където не показа полза) — крайната прогноза е **тегловна комбинация** (`blend_weight`, тунирана per-сегмент чрез cross-validation) между LightGBM и CatBoost
- **TF-IDF + SVD (15 компонента)** върху описанието на обявата, добавени като features за 4 от 5-те сегмента (не за офиси) — тествано срещу sentence-transformer ембединги (MiniLM/e5/bge-m3); TF-IDF печели навсякъде при много по-нисък изчислителен разход
- **Квантилна регресия** (`objective="quantile"`, α=0.1/0.9) за долна/горна граница на прогнозата — AVM панелът показва диапазон, не само точкова оценка
- **log1p target transform** за сегментите, при които подобрява стабилността
- Санитарна проверка при inference: прогноза извън 0.3×–3.0× спрямо медианата на кохортата (същия тип имот + гео категория) се clamp-ва

**Финални production метрики (MAE, EUR/кв.м, held-out test split):**

| Сегмент | MAE |
| --- | --- |
| Жилищни имоти | 266.6 |
| Офиси | 560.5 |
| Търговски имоти | 525.0 |
| Индустриални имоти | 236.9 |
| Хотелиерски имоти | 353.5 |

### MLflow tracking

Всички експерименти (data cleaning итерации, hyperparameter tuning, CatBoost/blend тестове, text feature ablations) са проследени в MLflow (`mlflow_tracking/mlflow.db`, SQLite backend):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
```

Пълна методология и намерения по рундове — `docs/avm_experiments/`.

### Тренировъчен pipeline

```bash
python -m scripts.train_avm_model              # всички 5 сегмента
python -m scripts.train_avm_model --segment office   # само един сегмент
```

Всеки сегмент с под `min_row_threshold` training-eligible обяви **не получава модел** — не се публикува low-confidence прогноза. Активирането на нов модел е атомарно per-сегмент (не засяга останалите сегменти).

---

## Кадастър и устройствено планиране (GIS модул)

`utils/gis/` интегрира приложението с реални, свободни (но неофициални — без публична API документация) източници на кадастрални и устройствени данни за България.

### АГКК — кадастър (национално покритие)

Свободна, отворена, автентикация не се изисква INSPIRE услуга на Агенцията по геодезия, картография и кадастър (`inspire.cadastre.bg`) — **не** същата услуга като платения WMS продукт на КАИС портала.

- **Поземлени имоти** — идентификатор, площ, административна единица, геометрия на границата
- **Сгради** — всички сгради, регистрирани върху даден имот (площта им се изчислява от геометрията, тъй като АГКК не я предоставя директно за сгради)
- **Съседни имоти** — пространствена заявка за граничещи парцели, основа на генерираното правно описание

### Автоматично правно описание на имота

Генерира текст по установения нотариален/ЗКИР стандарт — кадастрални номера и площ, изписани едновременно с цифри и словом (`68134.905.1462 /шест, осем, едно, ...точка.../`), с клауза за съседите. Когато е зададен идентификатор на конкретна сграда (4-сегментен, напр. `68134.905.1462.1`), описанието автоматично минава на "СГРАДА ... разположена в поземлен имот ..." вместо описание на голия парцел.

### Схематична скица

SVG визуализация на реалната граница на парцела (по данни от АГКК), директно в панела — без нужда от външен GIS софтуер.

### Устройствена зона (ОУП на София)

Реални параметри от Общия устройствен план на София — **Устройствена зона, Плътност на застрояване, КИНТ, Минимална озеленена площ** — извлечени директно от публичния WMS на GIS София (`isofmap.bg`), а не ръчно въвеждани. Покрива само урбанизираната територия на София в обхвата на ОУП от 2009 г.

### Свързани устройствени планове (НАГ София)

Търсене в регистъра на преписки на НАГ София (заявления и решения за ПУП/ИПРЗ/виза за проектиране) по кадастрален идентификатор, с директни връзки към действителните документи (скици-предложения, заявления, обяснителни записки).

### CLI за самостоятелни справки

```bash
python -m scripts.lookup_parcel --cadastral-id 68134.905.1462
python -m scripts.lookup_parcel --cadastral-id 68134.905.1462.1 --settlement-name "гр. София"
```

Пълна техническа документация (протоколи, reverse-engineering бележки, известни ограничения) — `docs/gis_cadastral/PLAN.md`.

> **Обхват:** покритието е гарантирано само за София. Кадастралните данни (АГКК) са национални; устройствената зона и преписките на НАГ са специфични за София — за останалите общини не е потвърден еквивалентен свободен източник.

---

## Стартиране (локално)

**Изисквания:** PostgreSQL (локален или Docker)

```bash
# 1. Стартиране на базата данни
docker-compose up db -d

# 2. Конфигурация
cp .env.example .env
# Редактирайте DATABASE_URL и SECRET_KEY; по избор ADMIN_EMAIL

# 3. Зависимости
pip install -r requirements.txt

# 4. Миграции
alembic upgrade head

# 5. Първи администраторски акаунт
python -m scripts.create_admin

# 6. Еднократен импорт на исторически данни (опционален)
python -m scripts.import_historical_data

# 7. Стартиране
uvicorn app.main:app --reload
```

Приложението е достъпно на `http://localhost:8000`.

На Windows е наличен и `start_app.bat` — проверява дали PostgreSQL слуша на порт 54891, стартира uvicorn на **порт 8891** (нарочно различен от Jupyter-ския default 8888) и отваря браузъра.

---

## DB схема

| Таблица | Описание |
| --- | --- |
| `scrape_runs` | Всеки scrape run с прогрес, статус, PID, heartbeat |
| `listings` | Обяви — upsert key: `ad_url`; статус: active / archived |
| `listing_snapshots` | Append-only ценова история на всяка обява |
| `listing_price_events` | Засечени промени в цена между последователни snapshots |
| `taxonomy_geo_paths` / `taxonomy_property_types` | Валидни geo пътища и типове имоти от sitemap |
| `comparable_pool` | Работен пул от сравними: корекции, pins, бележки |
| `report_comparables` | Финализираните до 6 сравними на тип, закачени към конкретен доклад |
| `appraisal_reports` | Оценъчни доклади — оценяван обект, кадастър, всички подходи, заключена стойност |
| `avm_models` | Регистър на трениран AVM модели — по сегмент, с метрики, хиперпараметри, `is_active` |
| `users` / `user_consents` | Потребители, роли, версионирани съгласия с политики |

Плюс материализиран изглед `mv_analytics_flat` за бързи агрегации в Analytics.

---

## Структура на проекта

```text
├── notebooks/                              # прототип/произход на pipeline-а
│   ├── 01_imot_bg_scraper_tool.ipynb
│   ├── 02_imot_bg_query_tool.ipynb
│   ├── 03_EDA_and_feature_engineering.ipynb
│   └── 04_residential_real_estate_regression_ML_analysis.ipynb
│
├── utils/
│   ├── fetch_data/fetch_data_utils.py       # scraping, taxonomy refresh
│   ├── ad_parsing/ad_parsing_utils.py       # HTML parsing на обяви
│   ├── feature_engineering/                # feature engineering (споделен с app)
│   ├── ml/                                  # AVM feature definitions, text features
│   └── gis/                                 # кадастър + устройствено планиране
│       ├── connectors/                      # AGKK, isofmap.bg, НАГ София, CKAN
│       ├── engines/                         # правно описание, сгради, spelling
│       ├── spatial_engine/                  # CRS transform, скица (SVG)
│       ├── models/                          # Pydantic схеми
│       └── cache/                           # SQLite response cache
│
├── app/                                     # FastAPI приложение
│   ├── main.py
│   ├── config.py
│   ├── db/                                  # SQLAlchemy models + session
│   ├── routers/                             # auth, admin, scrape, listings,
│   │                                        #   comparables, reports, analytics
│   ├── services/                            # бизнес логика (вкл. avm_service, gis_service)
│   └── templates/                           # Jinja2 HTML
│
├── scripts/
│   ├── create_admin.py                      # първи администраторски акаунт
│   ├── run_scrape.py                        # standalone scrape subprocess
│   ├── import_historical_data.py            # еднократен import от parquet
│   ├── train_avm_model.py                   # тренировка на AVM (всички/1 сегмент)
│   └── lookup_parcel.py                     # CLI за GIS справки
│
├── docs/
│   ├── avm_experiments/                     # AVM методология и намерения по рундове
│   └── gis_cadastral/                       # GIS reverse-engineering документация
│
├── alembic/                                 # DB миграции
├── models/                                  # gitignored — трениран AVM артефакти
├── mlflow_tracking/                         # gitignored — MLflow SQLite backend
├── backups/                                 # gitignored — pg_dump архиви
├── static/                                  # CSS
├── data/                                    # gitignored — суров/parsed данни
├── outputs/                                 # gitignored — генерирани доклади
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── alembic.ini
├── start_app.bat
└── .env.example
```

---

## Статус и посоки за развитие

**Завършено и в production:** scraping pipeline с crash-resumable checkpoints, пълен comparables workflow с три подхода за стойност, сегментен AVM (LightGBM+CatBoost+TF-IDF, квантилни интервали), автентикация с роли, Word/Excel export, GIS интеграция за София (кадастър, правно описание, скица, устройствена зона, преписки на НАГ).

**Известни ограничения:**

- Устройствена зона и преписки на НАГ покриват само София — за останалите общини не е потвърден еквивалентен свободен източник
- Правното описание все още не се включва автоматично в Word/Excel export-а (само в GIS панела на страницата)
- AVM моделите се тренират ръчно (админ панел или CLI) — няма автоматично пре-трениране при нов scrape run

**Планирано развитие:**

- Автоматично пре-трениране на AVM при натрупване на достатъчно нови данни
- Explainability (SHAP) в оценъчните доклади
- Разширяване на GIS покритието към Пловдив/Варна, ако бъде намерен еквивалентен свободен източник
- Включване на правното описание в генерирания Word доклад

---

## Лиценз

Този проект е публикуван само с демонстрационна и портфолио цел.

Кодът, методологията, scraping логиката, структурата на базата данни, моделите и оценителските workflow-и са proprietary и не могат да бъдат копирани, модифицирани, използвани, хоствани или комерсиализирани без предварително писмено разрешение.

Вижте [LICENSE](LICENSE).
