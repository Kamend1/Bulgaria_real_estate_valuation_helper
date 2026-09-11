"""
Regression test for a real incident (2026-09-11): a ~6-minute DNS outage
on the scraping machine coincided exactly with the download of the entire
rent segment of a scrape, and every one of those 37,193 failed URLs was
recorded in download_manifest.csv -- but load_completed_listing_urls()
treated ANY row (success or failure) as "already done", so a resumed run
would have skipped retrying them forever. Fixed to only count a URL as
completed when it actually has html_path populated (the field
download_one_listing/download_and_parse_one_listing only set on a real
200 response).
"""
import csv

from utils.fetch_data.fetch_data_utils import load_completed_listing_urls


def _write_manifest(path, rows):
    fieldnames = ["listing_url", "url_hash", "status_code", "html_path", "html_length", "downloaded_at", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_missing_manifest_returns_empty_set(tmp_path):
    assert load_completed_listing_urls(tmp_path / "does_not_exist.csv") == set()


def test_successful_download_counts_as_completed(tmp_path):
    path = tmp_path / "download_manifest.csv"
    _write_manifest(path, [
        {"listing_url": "https://example.com/a", "url_hash": "h1", "status_code": 200,
         "html_path": "data/raw_listing_html/h1.html", "html_length": 12345, "downloaded_at": "", "error": ""},
    ])
    assert load_completed_listing_urls(path) == {"https://example.com/a"}


def test_failed_download_does_not_count_as_completed():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "download_manifest.csv"
        _write_manifest(path, [
            {"listing_url": "https://example.com/rent-1", "url_hash": "h2", "status_code": 0,
             "html_path": "", "html_length": 0, "downloaded_at": "",
             "error": "NameResolutionError: Failed to resolve 'www.imot.bg'"},
        ])
        # The critical assertion: a failed attempt must NOT be treated as
        # done -- it has to remain eligible for retry on the next resume.
        assert load_completed_listing_urls(path) == set()


def test_mixed_manifest_only_counts_successes(tmp_path):
    path = tmp_path / "download_manifest.csv"
    _write_manifest(path, [
        {"listing_url": "https://example.com/ok", "url_hash": "h3", "status_code": 200,
         "html_path": "data/raw_listing_html/h3.html", "html_length": 999, "downloaded_at": "", "error": ""},
        {"listing_url": "https://example.com/failed", "url_hash": "h4", "status_code": 404,
         "html_path": "", "html_length": 0, "downloaded_at": "", "error": ""},
    ])
    assert load_completed_listing_urls(path) == {"https://example.com/ok"}


def test_retried_url_that_later_succeeds_counts_as_completed(tmp_path):
    # Simulates the real recovery path: the same URL appears twice in the
    # (append-only) manifest -- once from the original failed attempt, once
    # from a later successful retry. Any successful row is enough.
    path = tmp_path / "download_manifest.csv"
    _write_manifest(path, [
        {"listing_url": "https://example.com/retried", "url_hash": "h5", "status_code": 0,
         "html_path": "", "html_length": 0, "downloaded_at": "t1", "error": "connection reset"},
        {"listing_url": "https://example.com/retried", "url_hash": "h5", "status_code": 200,
         "html_path": "data/raw_listing_html/h5.html", "html_length": 4321, "downloaded_at": "t2", "error": ""},
    ])
    assert load_completed_listing_urls(path) == {"https://example.com/retried"}
