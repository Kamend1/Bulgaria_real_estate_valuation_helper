real_estate_valuation/
├── notebooks/
│   ├── 01_taxonomy_discovery.ipynb
│   ├── 02_download_pipeline_test.ipynb
│   ├── 03_parser_quality_check.ipynb
│   └── 04_market_analysis.ipynb
│
├── src/
│   └── imot_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── taxonomy.py
│       ├── selection.py
│       ├── result_pages.py
│       ├── downloader.py
│       ├── parser.py
│       ├── storage.py
│       └── exports.py
│
├── data/
│   ├── reference/
│   │   ├── valid_deal_types.csv
│   │   ├── valid_geo_paths.csv
│   │   └── valid_property_types.csv
│   │
│   ├── transport/
│   │   ├── listing_urls_transport.csv
│   │   ├── download_manifest_transport.csv
│   │   └── parsed_listings_transport.csv
│   │
│   └── raw_html/
│
├── outputs/
│   ├── reports/
│   └── benchmarks/
│
├── scripts/
│   ├── collect_listing_urls.py
│   ├── download_listings.py
│   ├── parse_listings.py
│   └── build_snapshot_db.py
│
├── tests/
│   ├── test_parser.py
│   └── fixtures/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE