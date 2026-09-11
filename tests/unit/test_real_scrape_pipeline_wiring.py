"""
Regression test for a real incident (2026-09-11): the automatic AVM
retrain hook (and, before it, the automatic embeddings refresh -- caught
2026-08-28) was added to app/services/scrape_service.py::
run_scrape_background(), a function with ZERO callers anywhere in the
app. Real scrapes run exclusively through scripts/run_scrape.py, launched
via subprocess.Popen from app/routers/scrape.py -- a completely separate,
independently maintained copy of the pipeline. Both features silently
never ran for a single real scrape as a result.

run_scrape_background() and its supporting dead code (ProgressCapture,
app/progress/store.py) have since been deleted entirely specifically to
stop this class of bug from recurring a third time. This test asserts
the real pipeline actually calls the functions a "runs on every scrape"
feature needs -- a cheap, static guardrail against this exact mistake
happening again in a fourth spot.
"""
from pathlib import Path

_RUN_SCRAPE_SRC = (Path(__file__).parent.parent.parent / "scripts" / "run_scrape.py").read_text(encoding="utf-8")


def test_real_pipeline_launches_via_subprocess_from_scrape_router():
    router_src = (Path(__file__).parent.parent.parent / "app" / "routers" / "scrape.py").read_text(encoding="utf-8")
    assert "scripts.run_scrape" in router_src


def test_dead_orchestrator_and_its_support_code_are_actually_gone():
    # If run_scrape_background() (or its in-memory progress store) ever
    # comes back, any future per-scrape hook risks landing there again
    # instead of in the real pipeline below.
    service_src = (Path(__file__).parent.parent.parent / "app" / "services" / "scrape_service.py").read_text(encoding="utf-8")
    assert "def run_scrape_background" not in service_src
    assert not (Path(__file__).parent.parent.parent / "app" / "progress" / "store.py").exists()


def test_real_pipeline_calls_avm_retrain():
    assert "maybe_retrain_avm_models" in _RUN_SCRAPE_SRC


def test_real_pipeline_calls_analytics_refresh():
    assert "compute_price_events" in _RUN_SCRAPE_SRC
    assert "refresh_mv" in _RUN_SCRAPE_SRC


def test_real_pipeline_calls_embeddings_backfill():
    assert "backfill_embeddings" in _RUN_SCRAPE_SRC
