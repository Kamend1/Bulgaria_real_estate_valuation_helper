from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg2://appraiser:appraiser@localhost:5432/appraisal"

    taxonomy_dir: str = "data/taxonomy"
    raw_html_dir: str = "data/raw_listing_html"
    scrape_runs_dir: str = "data/scrape_runs"
    reports_output_dir: str = "outputs/reports"
    report_template_path: str = "templates/reports/appraisal_template.docx"

    scrape_max_workers_routes: int = 8
    scrape_max_workers_listings: int = 16
    scrape_delay_seconds: float = 0.8

    app_title: str = "Оценка на недвижими имоти"


settings = Settings()
