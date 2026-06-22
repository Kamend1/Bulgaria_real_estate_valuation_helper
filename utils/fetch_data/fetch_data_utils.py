import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import csv
import requests
from itertools import product
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import gzip
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import time
from typing import Any
import pandas as pd
import csv
from pathlib import Path
from datetime import datetime, timezone
from utils.ad_parsing import ad_parsing_utils as adp


BASE_URL = "https://www.imot.bg"
SALE_URL = 'https://www.imot.bg/search/prodazhbi'
RENT_URL = 'https://www.imot.bg/search/naemi'
SITEMAP_INDEX_URL = "https://www.imot.bg/sitemap/index.xml"
HEADERS = {
    "User-Agent": "Mozilla/5.0 valuation-research-bot"
}
RAW_HTML_DIR = Path("./data/raw_listing_html")


DEAL_TYPE_CONFIG = {
    "sale": {
        "label_bg": "Продажби",
        "slug": "prodazhbi",
        "base_url": "https://www.imot.bg/search/prodazhbi",
    },
    "rent": {
        "label_bg": "Наеми",
        "slug": "naemi",
        "base_url": "https://www.imot.bg/search/naemi",
    },
}

PROPERTY_TYPE_SLUGS = {
    "ednostaen",
    "dvustaen",
    "tristaen",
    "chetiristaen",
    "mnogostaen",
    "mezonet",
    "atelie-tavan",
    "staya",

    "kashta",
    "etazh-ot-kashta",
    "vila",

    "partsel",
    "zemedelska-zemya",
    "myasto",

    "magazin",
    "ofis",
    "sklad",
    "promishleno-pomeshtenie",
    "biznes-imot",
    "garazh-parkomyasto",
    "zavedenie",
    "hotel",
}

# 1 Class for Scrape Selection
class ScrapeSelection:
    def __init__(
        self,
        deal_types: list[str],
        geo_paths: list[str],
        property_types: list[str] | None = None,
    ):
        self.deal_types = deal_types
        self.geo_paths = geo_paths
        self.property_types = property_types or []

    def validate(
        self,
        valid_deal_types: set[str],
        valid_geo_paths: set[tuple[str, str]],
        valid_property_types: set[str],
    ) -> None:
        for deal_type in self.deal_types:
            if deal_type not in valid_deal_types:
                raise ValueError(f"Invalid deal_type: {deal_type}")

        for property_type in self.property_types:
            if property_type not in valid_property_types:
                raise ValueError(f"Invalid property_type: {property_type}")

        for deal_type in self.deal_types:
            for geo_path in self.geo_paths:
                if (deal_type, geo_path) not in valid_geo_paths:
                    raise ValueError(
                        f"Invalid geo_path for deal_type={deal_type}: {geo_path}"
                    )

    def iter_combinations(self):
        if not self.property_types:
            for deal_type, geo_path in product(self.deal_types, self.geo_paths):
                yield deal_type, geo_path, None
            return

        for deal_type, geo_path, property_type in product(
            self.deal_types,
            self.geo_paths,
            self.property_types,
        ):
            yield deal_type, geo_path, property_type

    def build_urls(
        self,
        valid_deal_types: set[str],
        valid_geo_paths: set[tuple[str, str]],
        valid_property_types: set[str],
    ) -> list[str]:
        self.validate(
            valid_deal_types=valid_deal_types,
            valid_geo_paths=valid_geo_paths,
            valid_property_types=valid_property_types,
        )

        return [
            self._build_imot_url(deal_type, geo_path, property_type)
            for deal_type, geo_path, property_type in self.iter_combinations()
        ]

    @staticmethod
    def _build_imot_url(
        deal_type: str,
        geo_path: str,
        property_type: str | None = None,
    ) -> str:
        geo_parts = [
            part.strip("/")
            for part in geo_path.split("/")
            if part.strip("/")
        ]

        path_parts = [
            BASE_URL,
            "obiavi",
            deal_type,
            *geo_parts,
        ]

        if property_type:
            path_parts.append(property_type.strip("/"))

        return "/".join(path_parts)


# 2. Refresh taxonomy from imot.bg sitemap (replicates Notebook 01)
def refresh_taxonomy(taxonomy_dir: str | Path) -> None:
    """
    Fetches the imot.bg sitemap, discovers all valid geo paths and property
    types, and overwrites the three taxonomy CSVs. Equivalent to running
    Notebook 01 in full.
    """
    from collections import Counter

    taxonomy_dir = Path(taxonomy_dir)
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    print("Refreshing taxonomy from imot.bg sitemap…")

    index_xml = fetch_xml(SITEMAP_INDEX_URL)
    child_sitemaps = extract_locs(index_xml)
    print(f"Child sitemaps found: {len(child_sitemaps)}")

    obiavi_urls: list[str] = []
    for sitemap_url in child_sitemaps:
        try:
            xml_text = fetch_xml(sitemap_url)
            urls = extract_locs(xml_text)
            obiavi_urls.extend(url for url in urls if "/obiavi/" in url)
            print(f"{sitemap_url}: {len(urls)} URLs")
        except Exception as exc:
            print(f"FAILED {sitemap_url}: {exc}")

    print(f"/obiavi/ URLs found: {len(obiavi_urls)}")

    parsed_routes = [
        p for url in obiavi_urls
        if (p := parse_obiavi_geo_path(url)) is not None
    ]
    print(f"Parsed routes: {len(parsed_routes)}")

    # valid_deal_types.csv — hardcoded; imot.bg only has these two
    deal_types = [
        {"deal_type": "prodazhbi", "deal_type_en": "sale",  "deal_type_bg": "Продажби"},
        {"deal_type": "naemi",     "deal_type_en": "rent",  "deal_type_bg": "Наеми"},
    ]
    with open(taxonomy_dir / "valid_deal_types.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["deal_type", "deal_type_en", "deal_type_bg"])
        writer.writeheader()
        writer.writerows(deal_types)

    # valid_property_types.csv
    property_counter = Counter(
        r["base_property_type"]
        for r in parsed_routes
        if r["base_property_type"]
    )
    with open(taxonomy_dir / "valid_property_types.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["property_type", "route_count"])
        for property_type, count in sorted(property_counter.items()):
            writer.writerow([property_type, count])

    # valid_geo_paths.csv
    geo_counter = Counter(
        (r["deal_type"], r["geo_path"])
        for r in parsed_routes
    )
    with open(taxonomy_dir / "valid_geo_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["deal_type", "geo_path", "geo_level_count", "geo_1", "geo_2", "geo_3", "route_count"])
        for (deal_type, geo_path), route_count in sorted(geo_counter.items()):
            parts = geo_path.split("/")
            padded = parts + ["", "", ""]
            writer.writerow([deal_type, geo_path, len(parts), padded[0], padded[1], padded[2], route_count])

    print("Taxonomy refresh complete.")
    print(f"  Deal types: {len(deal_types)}")
    print(f"  Property types: {len(property_counter)}")
    print(f"  Geo paths: {len(geo_counter)}")


# 3. Methods to load valid url selectors
def load_valid_deal_types(path: str) -> set[str]:
    """
    Loads valid_deal_types.csv

    Expected columns:
    deal_type, deal_type_en, deal_type_bg
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["deal_type"].strip()
            for row in reader
            if row.get("deal_type")
        }


def load_valid_geo_paths(path: str) -> set[tuple[str, str]]:
    """
    Loads valid_geo_paths.csv

    Expected columns:
    deal_type, geo_path, geo_level_count, geo_1, geo_2, geo_3, route_count

    Returns:
    {
        ("prodazhbi", "grad-sofiya"),
        ("prodazhbi", "grad-sofiya/iztok"),
        ("naemi", "grad-varna"),
        ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (
                row["deal_type"].strip(),
                row["geo_path"].strip()
            )
            for row in reader
            if row.get("deal_type") and row.get("geo_path")
        }


def load_valid_property_types(path: str) -> set[str]:
    """
    Loads valid_property_types.csv

    Expected columns:
    property_type, route_count
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["property_type"].strip()
            for row in reader
            if row.get("property_type")
        }


# 3 Generate listing urls
def extract_listing_urls(result_html: str) -> list[str]:
    soup = BeautifulSoup(result_html, "lxml")

    listing_urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/obiava-" in href:
            full_url = urljoin("https://www.imot.bg", href)
            listing_urls.add(full_url.split("#")[0])

    return sorted(listing_urls)


# 4 Methods to extract actual ads listings based on criteria
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """
    Append rows to CSV immediately.
    Creates the file with header if it does not exist.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def load_completed_routes(route_results_path: Path) -> set[str]:
    """
    Reads already completed routes so the process can resume.
    """
    if not route_results_path.exists():
        return set()

    completed = set()

    with open(route_results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            route_url = row.get("route_url")
            if route_url:
                completed.add(route_url)

    return completed


# Global concurrency cap — limits simultaneous HTTP requests to imot.bg
# regardless of how many threads are in use (Phase 4 rate limiting).
import threading as _threading
_HTTP_SEMAPHORE = _threading.Semaphore(8)


def fetch_html(
    url: str,
    delay_seconds: float = 1.0,
    max_retries: int = 3,
    timeout: int = 30,
) -> tuple[int, str]:
    """
    Fetches HTML with retry/exponential backoff.
    Global semaphore caps concurrent requests to imot.bg at 8.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        # Jitter already present: delay_seconds + random.uniform(0, 0.5)
        time.sleep(delay_seconds + random.uniform(0, 0.5))

        with _HTTP_SEMAPHORE:
            try:
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=timeout,
                    allow_redirects=True,
                )

                html = response.content.decode("windows-1251", errors="replace")

                if response.status_code == 200:
                    return response.status_code, html

                if response.status_code == 429:
                    # Rate-limited: exponential backoff, longer wait
                    wait = 30 * (2 ** (attempt - 1))   # 30s, 60s, 120s
                    last_error = f"status_429_rate_limited"
                    time.sleep(wait)
                    continue

                if response.status_code in {403, 500, 502, 503, 504}:
                    last_error = f"status_code_{response.status_code}"
                    time.sleep(attempt * 3)   # 3s, 6s, 9s
                    continue

                return response.status_code, html

            except Exception as exc:
                last_error = str(exc)
                time.sleep(attempt * 3)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def build_paginated_url(base_route_url: str, page: int) -> str:
    base_route_url = base_route_url.rstrip("/")

    if page == 1:
        return base_route_url

    return f"{base_route_url}/p-{page}"


def collect_listing_urls_until_invalid(
    base_route_url: str,
    max_pages: int = 200,
    delay_seconds: float = 1.0,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Crawls one route sequentially through pagination.

    Returns:
    - page_results: one record per tested page
    - all_listing_urls: deduplicated listing URLs across valid pages
    """

    page_results = []
    all_listing_urls = set()
    seen_page_signatures = set()

    for page in range(1, max_pages + 1):
        page_url = build_paginated_url(base_route_url, page)

        if verbose:
            print(f"Fetching page {page}: {page_url}")

        started_at = time.time()

        try:
            status_code, html = fetch_html(
                page_url,
                delay_seconds=delay_seconds,
            )
            error = None

        except Exception as exc:
            elapsed_seconds = round(time.time() - started_at, 3)

            if verbose:
                print(f"FAILED request on page {page}: {exc}")

            page_results.append({
                "base_route_url": base_route_url,
                "page": page,
                "page_url": page_url,
                "status_code": None,
                "listing_count": 0,
                "new_listing_count": 0,
                "elapsed_seconds": elapsed_seconds,
                "stop_reason": f"request_failed: {exc}",
                "error": str(exc),
            })

            break

        elapsed_seconds = round(time.time() - started_at, 3)

        listing_urls = extract_listing_urls(html)
        page_signature = tuple(listing_urls[:10])
        new_urls = set(listing_urls) - all_listing_urls

        stop_reason = ""

        if status_code != 200:
            stop_reason = f"status_code_{status_code}"

        elif not listing_urls:
            stop_reason = "no_listing_urls"

        elif page_signature in seen_page_signatures:
            stop_reason = "repeated_page"

        elif not new_urls:
            stop_reason = "no_new_listing_urls"

        page_results.append({
            "base_route_url": base_route_url,
            "page": page,
            "page_url": page_url,
            "status_code": status_code,
            "listing_count": len(listing_urls),
            "new_listing_count": len(new_urls),
            "elapsed_seconds": elapsed_seconds,
            "stop_reason": stop_reason,
            "error": error,
        })

        if stop_reason:
            if verbose:
                print(f"Stopping on page {page}: {stop_reason}")
            break

        seen_page_signatures.add(page_signature)
        all_listing_urls.update(listing_urls)

        if verbose:
            print(
                f"Page {page}: {len(listing_urls)} listings, "
                f"{len(new_urls)} new, total {len(all_listing_urls)}"
            )

    return page_results, sorted(all_listing_urls)


def collect_one_route(
    route_url: str,
    max_pages: int = 200,
    delay_seconds: float = 1.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Worker function for one route.
    """

    started_at = time.time()

    try:
        page_results, listing_urls = collect_listing_urls_until_invalid(
            base_route_url=route_url,
            max_pages=max_pages,
            delay_seconds=delay_seconds,
            verbose=verbose,
        )

        error = None

    except Exception as exc:
        page_results = []
        listing_urls = []
        error = str(exc)

    elapsed_seconds = round(time.time() - started_at, 3)

    return {
        "route_url": route_url,
        "page_results": page_results,
        "listing_urls": listing_urls,
        "listing_count": len(listing_urls),
        "pages_checked": len(page_results),
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def collect_one_route_listing_urls(
    route_url: str,
    max_pages: int = 200,
    delay_seconds: float = 0.8,
) -> tuple[dict, list[dict], list[dict]]:
    """
    Collects listing URLs for one route only.

    Returns:
    - route_result: one summary row
    - page_rows: one row per tested page
    - listing_rows: one row per discovered listing URL
    """

    started_at = utc_now_iso()

    page_rows = []
    all_listing_urls = set()
    seen_page_signatures = set()

    final_stop_reason = None
    error_message = None

    try:
        for page in range(1, max_pages + 1):
            page_url = build_paginated_url(route_url, page)

            try:
                status_code, html = fetch_html(
                    page_url,
                    delay_seconds=delay_seconds,
                )
            except Exception as exc:
                status_code = None
                html = ""
                stop_reason = f"request_failed: {exc}"
                error_message = str(exc)

                page_rows.append({
                    "route_url": route_url,
                    "page": page,
                    "page_url": page_url,
                    "status_code": status_code,
                    "listing_count": 0,
                    "new_listing_count": 0,
                    "stop_reason": stop_reason,
                    "checked_at": utc_now_iso(),
                })

                final_stop_reason = stop_reason
                break

            listing_urls = extract_listing_urls(html)

            page_signature = tuple(listing_urls[:10])
            new_urls = set(listing_urls) - all_listing_urls

            stop_reason = ""

            if status_code != 200:
                stop_reason = f"status_code_{status_code}"

            elif not listing_urls:
                stop_reason = "no_listing_urls"

            elif page_signature in seen_page_signatures:
                stop_reason = "repeated_page"

            elif not new_urls:
                stop_reason = "no_new_listing_urls"

            page_rows.append({
                "route_url": route_url,
                "page": page,
                "page_url": page_url,
                "status_code": status_code,
                "listing_count": len(listing_urls),
                "new_listing_count": len(new_urls),
                "stop_reason": stop_reason,
                "checked_at": utc_now_iso(),
            })

            if stop_reason:
                final_stop_reason = stop_reason
                break

            seen_page_signatures.add(page_signature)
            all_listing_urls.update(listing_urls)

        listing_rows = [
            {
                "route_url": route_url,
                "listing_url": listing_url,
                "discovered_at": utc_now_iso(),
            }
            for listing_url in sorted(all_listing_urls)
        ]

        route_result = {
            "route_url": route_url,
            "pages_tested": len(page_rows),
            "listing_count": len(all_listing_urls),
            "stop_reason": final_stop_reason,
            "error": error_message,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        }

        return route_result, page_rows, listing_rows

    except Exception as exc:
        route_result = {
            "route_url": route_url,
            "pages_tested": len(page_rows),
            "listing_count": len(all_listing_urls),
            "stop_reason": "worker_failed",
            "error": str(exc),
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        }

        return route_result, page_rows, []


def collect_listing_urls_for_routes_parallel_streaming(
    route_urls: list[str],
    checkpoint_dir: str | Path,
    max_workers: int = 12,
    max_pages: int = 200,
    delay_seconds: float = 0.8,
    progress_every: int = 100,
    max_in_flight_multiplier: int = 3,
) -> dict:
    """
    Crawls many route URLs in parallel.

    Critical design:
    - Does NOT keep all route/page/listing results in memory.
    - Appends route results to CSV immediately.
    - Appends page audit rows to CSV immediately.
    - Appends listing URL rows to CSV immediately.
    - Can resume after crash/freeze by skipping completed route_url rows.
    - Does NOT submit all futures at once.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    route_results_path = checkpoint_dir / "route_results.csv"
    page_results_path = checkpoint_dir / "page_results.csv"
    listing_urls_path = checkpoint_dir / "listing_urls_raw.csv"

    route_fieldnames = [
        "route_url",
        "pages_tested",
        "listing_count",
        "stop_reason",
        "error",
        "started_at",
        "finished_at",
    ]

    page_fieldnames = [
        "route_url",
        "page",
        "page_url",
        "status_code",
        "listing_count",
        "new_listing_count",
        "stop_reason",
        "checked_at",
    ]

    listing_fieldnames = [
        "route_url",
        "listing_url",
        "discovered_at",
    ]

    completed_routes = load_completed_routes(route_results_path)

    pending_routes = [
        route_url
        for route_url in route_urls
        if route_url not in completed_routes
    ]

    total_routes = len(route_urls)
    already_done = len(completed_routes)
    pending_count = len(pending_routes)

    print(f"Total routes: {total_routes}")
    print(f"Already completed: {already_done}")
    print(f"Pending routes: {pending_count}")

    if not pending_routes:
        print("Nothing to do. All routes already completed.")
        return {
            "total_routes": total_routes,
            "already_completed": already_done,
            "newly_completed": 0,
            "route_results_path": route_results_path,
            "page_results_path": page_results_path,
            "listing_urls_path": listing_urls_path,
        }

    max_in_flight = max_workers * max_in_flight_multiplier

    newly_completed = 0
    total_new_listing_rows = 0
    total_new_page_rows = 0

    route_iter = iter(pending_routes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        def submit_next() -> bool:
            try:
                route_url = next(route_iter)
            except StopIteration:
                return False

            future = executor.submit(
                collect_one_route_listing_urls,
                route_url,
                max_pages,
                delay_seconds,
            )

            futures[future] = route_url
            return True

        for _ in range(min(max_in_flight, pending_count)):
            submit_next()

        while futures:
            done, _ = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                route_url = futures.pop(future)

                try:
                    route_result, page_rows, listing_rows = future.result()
                except Exception as exc:
                    route_result = {
                        "route_url": route_url,
                        "pages_tested": 0,
                        "listing_count": 0,
                        "stop_reason": "future_failed",
                        "error": str(exc),
                        "started_at": None,
                        "finished_at": utc_now_iso(),
                    }
                    page_rows = []
                    listing_rows = []

                append_rows_csv(
                    route_results_path,
                    [route_result],
                    route_fieldnames,
                )

                append_rows_csv(
                    page_results_path,
                    page_rows,
                    page_fieldnames,
                )

                append_rows_csv(
                    listing_urls_path,
                    listing_rows,
                    listing_fieldnames,
                )

                newly_completed += 1
                total_new_page_rows += len(page_rows)
                total_new_listing_rows += len(listing_rows)

                done_total = already_done + newly_completed

                if (
                    newly_completed == 1
                    or newly_completed % progress_every == 0
                ):
                    print(
                        f"[{done_total}/{total_routes}] routes completed | "
                        f"new this run={newly_completed} | "
                        f"route listings={route_result['listing_count']} | "
                        f"pages={route_result['pages_tested']} | "
                        f"stop={route_result['stop_reason']} | "
                        f"url={route_url}"
                    )

                submit_next()

    print("Finished route discovery run.")
    print(f"New routes completed: {newly_completed}")
    print(f"New page rows written: {total_new_page_rows}")
    print(f"New listing rows written: {total_new_listing_rows}")
    print(f"Route results: {route_results_path}")
    print(f"Page results: {page_results_path}")
    print(f"Raw listing URLs: {listing_urls_path}")

    return {
        "total_routes": total_routes,
        "already_completed": already_done,
        "newly_completed": newly_completed,
        "new_page_rows": total_new_page_rows,
        "new_listing_rows": total_new_listing_rows,
        "route_results_path": route_results_path,
        "page_results_path": page_results_path,
        "listing_urls_path": listing_urls_path,
    }


# 5. Methods to explore listings
def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def fetch_listing_html(
    url: str,
    delay_seconds: float = 0.15,
) -> tuple[int, str, str | None]:
    """
    Downloads one listing page.

    Returns:
    status_code, html, error_message
    """
    time.sleep(delay_seconds)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
            allow_redirects=True,
        )

        html = response.content.decode("windows-1251", errors="replace")
        return response.status_code, html, None

    except Exception as exc:
        return 0, "", str(exc)


def save_listing_html(url: str, html: str) -> str:
    """
    Saves listing HTML to disk and returns file path.
    """
    url_hash = hash_url(url)
    html_path = RAW_HTML_DIR / f"{url_hash}.html"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(html_path)


# 6 extracting html to text
def fetch_listing_html_fast(
    url: str,
    min_delay: float = 0.1,
    max_delay: float = 0.2,
) -> tuple[int, str, str | None]:
    time.sleep(random.uniform(min_delay, max_delay))

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
            allow_redirects=True,
        )

        html = response.content.decode("windows-1251", errors="replace")

        return response.status_code, html, None

    except Exception as exc:
        return 0, "", str(exc)


def download_one_listing(url: str) -> dict:
    url_hash = hash_url(url)
    html_path = RAW_HTML_DIR / f"{url_hash}.html"

    status_code, html, error = fetch_listing_html_fast(url)

    html_length = len(html) if html else 0

    if status_code == 200 and html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
    else:
        html_path = None

    return {
        "listing_url": url,
        "url_hash": url_hash,
        "status_code": status_code,
        "html_path": str(html_path) if html_path else None,
        "html_length": html_length,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def download_listing_batch_parallel(
    listing_urls: list[str],
    max_workers: int = 5,
    limit: int | None = None,
) -> list[dict]:
    urls_to_download = listing_urls[:limit] if limit else listing_urls

    manifest = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one_listing, url): url
            for url in urls_to_download
        }

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            manifest.append(result)

            print(
                f"[{i}/{len(urls_to_download)}] "
                f"status={result['status_code']} "
                f"size={result['html_length']} "
                f"url={result['listing_url']}"
            )

    return manifest


# 7 perform general scraping to obtain geo and property type categories
def fetch_xml(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    content = response.content

    if url.endswith(".gz"):
        content = gzip.decompress(content)

    return content.decode("utf-8", errors="replace")


def extract_locs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)

    locs = []
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            locs.append(elem.text.strip())

    return locs


def parse_obiavi_geo_path(url: str) -> dict | None:
    """
    Parse an imot.bg /obiavi/ URL and extract:
    - deal_type: prodazhbi / naemi
    - geo_path: e.g. grad-sofiya or grad-sofiya/iztok
    - base_property_type: e.g. dvustaen, ofis, kashta, or None
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    # Expected:
    # /obiavi/{deal_type}/{geo...}/{optional_property_type}
    if len(parts) < 3:
        return None

    if parts[0] != "obiavi":
        return None

    deal_type = parts[1]

    if deal_type not in {"prodazhbi", "naemi"}:
        return None

    remaining = parts[2:]

    if not remaining:
        return None

    last_part = remaining[-1]

    if last_part in PROPERTY_TYPE_SLUGS:
        base_property_type = last_part
        geo_parts = remaining[:-1]
    else:
        base_property_type = None
        geo_parts = remaining

    if not geo_parts:
        return None

    return {
        "url": url,
        "deal_type": deal_type,
        "geo_path": "/".join(geo_parts),
        "geo_level_count": len(geo_parts),
        "geo_parts": geo_parts,
        "base_property_type": base_property_type,
    }


def append_one_row_csv(path: Path, row: dict) -> None:
    """
    Append one row to CSV immediately.
    Creates the file with header if it does not exist.
    """
    if not row:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def load_completed_listing_urls(manifest_path: Path) -> set[str]:
    """
    Reads already downloaded listing URLs so the process can resume.
    """
    if not manifest_path.exists():
        return set()

    completed = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            listing_url = row.get("listing_url")
            if listing_url:
                completed.add(listing_url)

    return completed


def download_one_listing(url: str) -> dict:
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    url_hash = hash_url(url)
    html_path = RAW_HTML_DIR / f"{url_hash}.html"

    status_code, html, error = fetch_listing_html_fast(url)

    html_length = len(html) if html else 0

    if status_code == 200 and html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
    else:
        html_path = None

    return {
        "listing_url": url,
        "url_hash": url_hash,
        "status_code": status_code,
        "html_path": str(html_path) if html_path else None,
        "html_length": html_length,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def download_and_parse_one_listing(url: str) -> tuple[dict, dict | None]:
    """
    Downloads one listing HTML file and immediately parses it.

    Returns:
    - download_row: manifest row for download status
    - parsed_row: parsed listing row, or None if download failed
    """

    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    url_hash = hash_url(url)
    html_path = RAW_HTML_DIR / f"{url_hash}.html"

    status_code, html, error = fetch_listing_html_fast(url)

    html_length = len(html) if html else 0

    if status_code == 200 and html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        html_path_str = str(html_path)
    else:
        html_path_str = None

    download_row = {
        "listing_url": url,
        "url_hash": url_hash,
        "status_code": status_code,
        "html_path": html_path_str,
        "html_length": html_length,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }

    if status_code != 200 or not html_path_str:
        return download_row, None

    try:
        parsed_row = adp.parse_imot_listing(
            html_path=html_path_str,
            ad_url=url,
            include_raw_text_preview=False,
        )

        if parsed_row is None:
            parsed_row = {}

        parsed_row["parse_error"] = None

    except Exception as exc:
        parsed_row = {
            "listing_url": url,
            "url_hash": url_hash,
            "html_path": html_path_str,
            "parse_error": str(exc),
        }

    return download_row, parsed_row


def download_and_parse_listing_batch_streaming(
    listing_urls: list[str],
    output_dir: str | Path,
    max_workers: int = 24,
    limit: int | None = None,
    chunk_size: int = 3000,
    progress_every: int = 300,
) -> dict:
    """
    Safer replacement for download_listing_batch_parallel.

    Critical design:
    - Does NOT keep all manifest rows in memory.
    - Does NOT keep all parsed rows in memory.
    - Appends download manifest rows in buffered batches.
    - Appends parsed rows in buffered batches.
    - Can resume after crash/freeze.
    - Does NOT submit all futures at once.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "download_manifest.csv"
    parsed_path = output_dir / "parsed_listings.csv"

    urls_to_process = listing_urls[:limit] if limit else listing_urls

    completed_urls = load_completed_listing_urls(manifest_path)

    pending_urls = [
        url
        for url in urls_to_process
        if url not in completed_urls
    ]

    total_urls = len(urls_to_process)
    already_done = len(completed_urls)
    pending_count = len(pending_urls)

    print(f"Total listing URLs: {total_urls}")
    print(f"Already completed: {already_done}")
    print(f"Pending URLs: {pending_count}")

    if not pending_urls:
        print("Nothing to do. All listing URLs already processed.")
        return {
            "total_urls": total_urls,
            "already_completed": already_done,
            "newly_completed": 0,
            "manifest_path": manifest_path,
            "parsed_path": parsed_path,
        }

    newly_completed = 0
    parsed_count = 0
    failed_download_count = 0
    failed_parse_count = 0

    flush_every = 300

    manifest_buffer = []
    parsed_buffer = []

    manifest_fieldnames = [
        "listing_url",
        "url_hash",
        "status_code",
        "html_path",
        "html_length",
        "downloaded_at",
        "error",
    ]

    parsed_fieldnames = None

    for batch_start in range(0, pending_count, chunk_size):
        batch = pending_urls[batch_start:batch_start + chunk_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_and_parse_one_listing, url): url
                for url in batch
            }

            for future in as_completed(futures):
                url = futures[future]

                try:
                    download_row, parsed_row = future.result()

                except Exception as exc:
                    download_row = {
                        "listing_url": url,
                        "url_hash": hash_url(url),
                        "status_code": 0,
                        "html_path": None,
                        "html_length": 0,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    }

                    parsed_row = None

                manifest_buffer.append(download_row)

                if parsed_row is not None:
                    parsed_buffer.append(parsed_row)
                    parsed_count += 1

                    if parsed_row.get("parse_error"):
                        failed_parse_count += 1

                    if parsed_fieldnames is None:
                        parsed_fieldnames = list(parsed_row.keys())

                if download_row.get("status_code") != 200:
                    failed_download_count += 1

                newly_completed += 1
                done_total = already_done + newly_completed

                if len(manifest_buffer) >= flush_every:
                    append_rows_csv(
                        manifest_path,
                        manifest_buffer,
                        manifest_fieldnames,
                    )
                    manifest_buffer.clear()

                if len(parsed_buffer) >= flush_every:
                    append_rows_csv(
                        parsed_path,
                        parsed_buffer,
                        parsed_fieldnames,
                    )
                    parsed_buffer.clear()

                if newly_completed == 1 or newly_completed % progress_every == 0:
                    print(
                        f"[{done_total}/{total_urls}] listings processed | "
                        f"new this run={newly_completed} | "
                        f"status={download_row.get('status_code')} | "
                        f"parsed={parsed_row is not None} | "
                        f"url={url}"
                    )

        # Safety flush after each chunk
        if manifest_buffer:
            append_rows_csv(
                manifest_path,
                manifest_buffer,
                manifest_fieldnames,
            )
            manifest_buffer.clear()

        if parsed_buffer:
            append_rows_csv(
                parsed_path,
                parsed_buffer,
                parsed_fieldnames,
            )
            parsed_buffer.clear()

    print("Finished listing download + parse run.")
    print(f"New listings processed: {newly_completed}")
    print(f"Parsed rows written: {parsed_count}")
    print(f"Failed downloads: {failed_download_count}")
    print(f"Failed parses: {failed_parse_count}")
    print(f"Download manifest: {manifest_path}")
    print(f"Parsed listings: {parsed_path}")

    return {
        "total_urls": total_urls,
        "already_completed": already_done,
        "newly_completed": newly_completed,
        "parsed_count": parsed_count,
        "failed_download_count": failed_download_count,
        "failed_parse_count": failed_parse_count,
        "manifest_path": manifest_path,
        "parsed_path": parsed_path,
    }