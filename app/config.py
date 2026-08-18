from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_SECRET_KEY_DEFAULT = "change-me-in-production-use-long-random-string"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg2://appraiser:appraiser@localhost:5432/appraisal"

    taxonomy_dir: str = "data/taxonomy"
    raw_html_dir: str = "data/raw_listing_html"
    scrape_runs_dir: str = "data/scrape_runs"
    reports_output_dir: str = "outputs/reports"
    report_template_path: str = "templates/reports/appraisal_template.docx"
    avm_models_dir: str = "models/avm"
    log_dir: str = "logs"

    scrape_max_workers_routes: int = 8
    scrape_max_workers_listings: int = 16
    scrape_delay_seconds: float = 0.8

    app_title: str = "Дигитален асистент на имотния оценител"

    # No default — a missing SECRET_KEY must fail loudly at startup, not
    # silently fall back to a value that's sitting in this file's git
    # history in a public repo. (It did exactly that for a while — .env
    # never actually set it until this was caught.)
    secret_key: str
    admin_email: str = ""  # email that auto-receives admin role on first registration
    # False by default so local dev over plain http://localhost keeps working.
    # Set SESSION_HTTPS_ONLY=true once the app is actually served over HTTPS —
    # without it the session cookie has no `Secure` flag and could be sent
    # over an unencrypted connection.
    session_https_only: bool = False

    # Cloudflare R2 (S3-compatible), split by credential tier — see README's
    # "R2 достъп и ключове" section. Account/endpoint are account-wide;
    # everything else is scoped to what actually needs it.
    r2_account_id: str = ""
    r2_endpoint_url: str = ""

    # Read-only — safe to ship in ANY deployed instance's .env (including a
    # colleague's/client's). Used by app/services/avm_service.py to fetch
    # AVM models at inference time. Scope this token to Object Read only on
    # the models bucket in the Cloudflare R2 dashboard.
    r2_models_access_key_id: str = ""
    r2_models_secret_access_key: str = ""
    r2_models_bucket_name: str = ""

    # Read-write — local-machine only, never deployed anywhere. Used by
    # scripts/train_avm_model.py's --push-to-r2 and scripts/backup_to_r2.py.
    # Deliberately points at a SEPARATE bucket for backups (pg_dump/mlflow)
    # so the read-only models key above can never reach it, even if leaked
    # in full — pg_dump contains real user PII (emails, hashed passwords)
    # and appraisal report content.
    r2_maintainer_access_key_id: str = ""
    r2_maintainer_secret_access_key: str = ""
    r2_backups_bucket_name: str = ""

    @field_validator("secret_key")
    @classmethod
    def _secret_key_must_be_real(cls, v: str) -> str:
        if v == _INSECURE_SECRET_KEY_DEFAULT:
            raise ValueError(
                "SECRET_KEY is still the placeholder from .env.example — generate a real one: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        if len(v) < 32:
            raise ValueError("SECRET_KEY is too short to be a real random key (need >= 32 chars).")
        return v


settings = Settings()
