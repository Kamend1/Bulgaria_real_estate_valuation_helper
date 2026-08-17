"""
Backs up the PostgreSQL database (pg_dump) and the current MLflow tracking
store (mlflow_tracking/mlflow.db) to Cloudflare R2 (S3-compatible).

Separate from DVC on purpose — see README's "Съхранение на моделите (DVC)"
section for why mlflow.db specifically isn't a good fit for DVC's
content-addressable versioning (it's a SQLite file rewritten internally
on every experiment run, so each version stored in full, not delta'd).
This script just uploads plain snapshots, no versioning semantics beyond
"one timestamped object per run."

Usage:
    python -m scripts.backup_to_r2
    python -m scripts.backup_to_r2 --skip-pgdump     # only upload mlflow.db
    python -m scripts.backup_to_r2 --skip-mlflow     # only pg_dump

Requires R2_* settings in .env (see .env.example) — role/bucket details are
never read from anywhere else, so a missing .env entry fails loudly instead
of silently uploading to the wrong place.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, ".")

import boto3
from botocore.exceptions import ClientError

from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUPS_DIR = PROJECT_ROOT / "backups"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow_tracking" / "mlflow.db"


def _require_r2_config() -> None:
    missing = [
        name for name, value in [
            ("R2_ENDPOINT_URL", settings.r2_endpoint_url),
            ("R2_ACCESS_KEY_ID", settings.r2_access_key_id),
            ("R2_SECRET_ACCESS_KEY", settings.r2_secret_access_key),
            ("R2_BUCKET_NAME", settings.r2_bucket_name),
        ] if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing R2 config in .env: {', '.join(missing)}. "
            "See .env.example for the expected keys."
        )


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",  # R2 ignores region but boto3 requires a value
    )


def _pg_dump(timestamp: str) -> Path:
    """Runs pg_dump in custom format (-Fc) — matches the manual backup
    convention already used in backups/. Strips SQLAlchemy's
    "+psycopg2" dialect suffix, which pg_dump's connection-URI parser
    doesn't understand."""
    BACKUPS_DIR.mkdir(exist_ok=True)
    out_path = BACKUPS_DIR / f"appraisal_{timestamp}.dump"

    pg_uri = settings.database_url.replace("postgresql+psycopg2://", "postgresql://", 1)

    print(f"Running pg_dump -> {out_path} ...")
    result = subprocess.run(
        ["pg_dump", "-Fc", "--no-owner", "--no-privileges", "-d", pg_uri, "-f", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump failed:\n{result.stderr}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"pg_dump complete: {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def _snapshot_mlflow_db(timestamp: str) -> Path:
    if not MLFLOW_DB_PATH.exists():
        raise FileNotFoundError(f"No MLflow tracking DB found at {MLFLOW_DB_PATH}")
    snapshot_path = BACKUPS_DIR / f"mlflow_{timestamp}.db"
    shutil.copy2(MLFLOW_DB_PATH, snapshot_path)
    size_kb = snapshot_path.stat().st_size / 1024
    print(f"MLflow DB snapshot: {snapshot_path.name} ({size_kb:.0f} KB)")
    return snapshot_path


def _upload(client, local_path: Path, key: str) -> None:
    print(f"Uploading {local_path.name} -> s3://{settings.r2_bucket_name}/{key} ...")
    try:
        client.upload_file(str(local_path), settings.r2_bucket_name, key)
    except ClientError as exc:
        raise RuntimeError(f"Upload failed for {local_path.name}: {exc}") from exc
    print("  done.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-pgdump", action="store_true")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    _require_r2_config()
    client = _r2_client()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not args.skip_pgdump:
        dump_path = _pg_dump(timestamp)
        _upload(client, dump_path, f"backups/pgdump/{dump_path.name}")

    if not args.skip_mlflow:
        mlflow_snapshot = _snapshot_mlflow_db(timestamp)
        _upload(client, mlflow_snapshot, f"backups/mlflow/{mlflow_snapshot.name}")
        mlflow_snapshot.unlink()  # local copy was only needed to hand off to upload_file

    print("Backup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
