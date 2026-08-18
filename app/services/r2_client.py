"""
Boto3 S3-compatible client factories for Cloudflare R2, split by credential
tier -- see README's "R2 достъп и ключове" section for the full rationale.

Two tiers, never mixed:
- models (read-only): app/services/avm_service.py, safe in any deployed
  instance's .env.
- maintainer (read-write): scripts/train_avm_model.py's --push-to-r2 and
  scripts/backup_to_r2.py, local-machine only.
"""
from __future__ import annotations

import boto3

from app.config import settings


def _client(access_key_id: str, secret_access_key: str):
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",  # R2 ignores region but boto3 requires a value
    )


def require_models_read_config() -> None:
    missing = [
        name for name, value in [
            ("R2_ENDPOINT_URL", settings.r2_endpoint_url),
            ("R2_MODELS_ACCESS_KEY_ID", settings.r2_models_access_key_id),
            ("R2_MODELS_SECRET_ACCESS_KEY", settings.r2_models_secret_access_key),
            ("R2_MODELS_BUCKET_NAME", settings.r2_models_bucket_name),
        ] if not value
    ]
    if missing:
        raise RuntimeError(f"Missing R2 models config in .env: {', '.join(missing)}")


def get_models_read_client():
    require_models_read_config()
    return _client(settings.r2_models_access_key_id, settings.r2_models_secret_access_key)


def require_maintainer_config() -> None:
    missing = [
        name for name, value in [
            ("R2_ENDPOINT_URL", settings.r2_endpoint_url),
            ("R2_MAINTAINER_ACCESS_KEY_ID", settings.r2_maintainer_access_key_id),
            ("R2_MAINTAINER_SECRET_ACCESS_KEY", settings.r2_maintainer_secret_access_key),
        ] if not value
    ]
    if missing:
        raise RuntimeError(f"Missing R2 maintainer config in .env: {', '.join(missing)}")


def get_maintainer_client():
    require_maintainer_config()
    return _client(settings.r2_maintainer_access_key_id, settings.r2_maintainer_secret_access_key)
