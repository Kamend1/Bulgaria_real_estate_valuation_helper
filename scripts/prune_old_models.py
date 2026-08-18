"""
Deletes old AVM training-run directories under models/avm/<segment>/<timestamp>/,
keeping each segment's currently-active run (per avm_models.is_active) plus
its N most recent runs by timestamp. Growth here is otherwise unbounded --
every `python -m scripts.train_avm_model` run leaves a full copy of every
pipeline file behind.

Usage:
    python -m scripts.prune_old_models               # dry run (default) -- lists what would be deleted
    python -m scripts.prune_old_models --execute      # actually deletes
    python -m scripts.prune_old_models --keep 5       # override retention count (default 3)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.config import settings
from app.db.base import SessionLocal
from app.db.models import AvmModel

DEFAULT_KEEP = 3


def _active_run_dirs_by_segment(db) -> dict[str, set[str]]:
    """Run-directory names (e.g. "20260810_113906") currently referenced by
    an is_active=True avm_models row, keyed by segment."""
    active: dict[str, set[str]] = {}
    for row in db.query(AvmModel).filter_by(is_active=True).all():
        run_dir_name = Path(row.model_path).parent.name
        active.setdefault(row.segment, set()).add(run_dir_name)
    return active


def _runs_to_delete(segment_dir: Path, keep: int, active_runs: set[str]) -> list[Path]:
    runs = sorted((p for p in segment_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    keep_names = active_runs | {p.name for p in runs[:keep]}
    return [p for p in runs if p.name not in keep_names]


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run).")
    parser.add_argument(
        "--keep", type=int, default=DEFAULT_KEEP,
        help=f"Most-recent runs to keep per segment, in addition to the active one (default {DEFAULT_KEEP}).",
    )
    args = parser.parse_args()

    models_dir = Path(settings.avm_models_dir)
    if not models_dir.exists():
        print(f"No models directory at {models_dir} -- nothing to prune.")
        return 0

    db = SessionLocal()
    try:
        active_by_segment = _active_run_dirs_by_segment(db)
    finally:
        db.close()

    to_delete: list[Path] = []
    total_freed_mb = 0.0
    for segment_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        active_runs = active_by_segment.get(segment_dir.name, set())
        doomed = _runs_to_delete(segment_dir, args.keep, active_runs)
        for run_dir in doomed:
            size_mb = _dir_size_mb(run_dir)
            total_freed_mb += size_mb
            prefix = "[DRY RUN] " if not args.execute else ""
            print(f"{prefix}{run_dir}  ({size_mb:.1f} MB)")
            to_delete.append(run_dir)

    if not to_delete:
        print("Nothing to prune.")
        return 0

    verb = "Freed" if args.execute else "Would free"
    print(f"\n{verb}: {total_freed_mb:.1f} MB across {len(to_delete)} run(s).")

    if args.execute:
        for run_dir in to_delete:
            shutil.rmtree(run_dir)
    else:
        print("Re-run with --execute to actually delete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
