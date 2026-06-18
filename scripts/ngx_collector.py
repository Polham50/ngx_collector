"""
NGX-FND Data Collector
======================
Collects corporate financial disclosure documents from the Nigerian Exchange (NGX)
for the LLM Financial Narrative Analysis research project.

Sources:
  - Company IR pages (annual reports, earnings releases)
  - Proshare Nigeria (proshareng.com)
  - NGX Group portal

Usage:
  python ngx_collector.py --mode discover   # Discover available PDFs
  python ngx_collector.py --mode download   # Download discovered PDFs
  python ngx_collector.py --mode extract    # Extract text from PDFs
  python ngx_collector.py --mode full       # Run all steps
  python ngx_collector.py --ticker MTNN     # Run for a single company
"""

import os
import json
import time
import hashlib
import logging
import argparse
import requests
import urllib3
import fitz  # PyMuPDF

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
RAW_PDF_DIR    = DATA_DIR / "raw_pdfs"
TEXT_DIR       = DATA_DIR / "extracted_text"
METADATA_DIR   = DATA_DIR / "metadata"
LOG_DIR        = BASE_DIR / "logs"
COMPANIES_FILE = DATA_DIR / "companies.json"

for d in [RAW_PDF_DIR, TEXT_DIR, METADATA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
TARGET_YEARS  = list(range(2020, 2025))   # 2020–2024
REQUEST_DELAY = 2.0                        # seconds between requests
MAX_PDF_SIZE  = 50 * 1024 * 1024           # 50 MB cap

# Keywords used to identify relevant disclosure documents
REPORT_KEYWORDS = [
    "annual report", "annual-report", "annualreport",
    "full year results", "full-year", "fy202", "fy 202",
    "half year results", "half-year", "h1 202", "h2 202",
    "earnings release", "investor presentation",
    "q1 results", "q2 results", "q3 results", "q4 results",
    "financial statements", "audited accounts",
    "directors report", "chairman statement",
]

# Narrative sections to extract (regex-friendly titles)
NARRATIVE_SECTIONS = [
    "chairman", "chief executive", "managing director",
    "operating review", "business review", "strategic review",
    "management discussion", "outlook", "prospects",
    "going concern", "principal risks", "future plans",
    "dividend", "corporate governance",
]

# ── Proshare scraper ───────────────────────────────────────────────────────────

def search_proshare(company_name: str, ticker: str) -> list[dict]:
    """Search Proshare Nigeria for company filings."""
    results = []
    search_url = f"https://www.proshareng.com/news/search/?q={ticker}+annual+report"
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True).lower()
            if any(kw in text for kw in REPORT_KEYWORDS) and any(str(y) in text for y in TARGET_YEARS):
                full_url = urljoin("https://www.proshareng.com", href)
                results.append({
                    "source":   "proshare",
                    "ticker":   ticker,
                    "url":      full_url,
                    "title":    link.get_text(strip=True),
                    "doc_type": classify_document(text),
                })
        log.info(f"[Proshare] {ticker}: found {len(results)} links")
    except Exception as e:
        log.warning(f"[Proshare] {ticker} search failed: {e}")
    return results


def search_ngx_portal(ticker: str) -> list[dict]:
    """Search NGX Group company filings page."""
    results = []
    url = f"https://ngxgroup.com/exchange/data/company-filings/?ticker={ticker}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True).lower()
            if href.endswith(".pdf") and any(str(y) in href + text for y in TARGET_YEARS):
                full_url = urljoin("https://ngxgroup.com", href)
                results.append({
                    "source":   "ngx_portal",
                    "ticker":   ticker,
                    "url":      full_url,
                    "title":    link.get_text(strip=True),
                    "doc_type": classify_document(text + href.lower()),
                })
        log.info(f"[NGX Portal] {ticker}: found {len(results)} links")
    except Exception as e:
        log.warning(f"[NGX Portal] {ticker} search failed: {e}")
    return results


def search_company_ir(company: dict) -> list[dict]:
    """Scrape company's own IR page for PDF links."""
    results = []
    ir_url = company.get("ir_url", "")
    if not ir_url or "sec.gov.ng" in ir_url:
        return results

    try:
        resp = requests.get(ir_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = (link.get_text(strip=True) + " " + href).lower()
            if href.endswith(".pdf") and (
                any(kw in text for kw in REPORT_KEYWORDS) or
                any(str(y) in text for y in TARGET_YEARS)
            ):
                full_url = urljoin(ir_url, href)
                results.append({
                    "source":   "company_ir",
                    "ticker":   company["ticker"],
                    "url":      full_url,
                    "title":    link.get_text(strip=True) or Path(href).stem,
                    "doc_type": classify_document(text),
                })
        log.info(f"[IR Page] {company['ticker']}: found {len(results)} PDF links")
    except Exception as e:
        log.warning(f"[IR Page] {company['ticker']} failed: {e}")
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def classify_document(text: str) -> str:
    """Classify document type from URL/title text."""
    text = text.lower()
    if any(x in text for x in ["annual report", "annual-report", "fy", "full year"]):
        return "annual_report"
    if any(x in text for x in ["half year", "h1", "h2", "interim"]):
        return "interim_report"
    if any(x in text for x in ["q1", "q2", "q3", "q4", "quarterly"]):
        return "quarterly_results"
    if any(x in text for x in ["earnings", "press release", "unaudited"]):
        return "earnings_release"
    if any(x in text for x in ["agm", "annual general"]):
        return "agm_statement"
    return "other"


def infer_year(text: str, url: str) -> str | None:
    """Try to extract the report year from title/URL."""
    combined = (text + " " + url).lower()
    for y in sorted(TARGET_YEARS, reverse=True):
        if str(y) in combined:
            return str(y)
    return None


def url_to_filename(ticker: str, url: str, title: str) -> str:
    """Generate a safe, unique filename for a PDF."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    safe_title = "".join(c if c.isalnum() else "_" for c in title)[:40]
    return f"{ticker}_{safe_title}_{url_hash}.pdf"


def download_pdf(url: str, dest_path: Path) -> bool:
    """Download a PDF to dest_path. Returns True on success."""
    if dest_path.exists():
        log.info(f"  [skip] already exists: {dest_path.name}")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            log.warning(f"  [skip] not a PDF ({content_type}): {url}")
            return False

        size = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > MAX_PDF_SIZE:
                    log.warning(f"  [skip] PDF too large (>{MAX_PDF_SIZE//1024//1024}MB): {url}")
                    dest_path.unlink(missing_ok=True)
                    return False
                f.write(chunk)
        log.info(f"  [ok] downloaded {dest_path.name} ({size//1024} KB)")
        return True
    except Exception as e:
        log.warning(f"  [fail] {url}: {e}")
        dest_path.unlink(missing_ok=True)
        return False


# ── PDF Text Extraction ────────────────────────────────────────────────────────

def extract_narrative_sections(pdf_path: Path) -> dict:
    """
    Extract narrative/qualitative sections from a financial PDF.
    Returns a dict with section names as keys and extracted text as values.
    """
    result = {
        "full_text":           "",
        "chairman_statement":  "",
        "ceo_review":          "",
        "operating_review":    "",
        "outlook":             "",
        "governance":          "",
        "page_count":          0,
        "extraction_status":   "ok",
    }

    try:
        with fitz.open(pdf_path) as pdf:
            result["page_count"] = len(pdf)
            all_pages_text = []
            for page in pdf:
                text = page.get_text("text")
                if text:
                    all_pages_text.append(text)

            full_text = "\n".join(all_pages_text)
            result["full_text"] = full_text

            # Segment into narrative sections
            result["chairman_statement"] = extract_section(
                full_text,
                start_markers=["chairman", "chairman's statement", "chairman's letter"],
                end_markers=["chief executive", "managing director", "board of directors",
                             "corporate governance", "directors' report"]
            )
            result["ceo_review"] = extract_section(
                full_text,
                start_markers=["chief executive", "ceo's review", "managing director",
                                "chief executive's review", "ceo review"],
                end_markers=["financial review", "operating review", "business review",
                             "corporate governance", "our strategy"]
            )
            result["operating_review"] = extract_section(
                full_text,
                start_markers=["operating review", "business review", "financial review",
                                "management discussion", "performance review"],
                end_markers=["risk", "governance", "sustainability", "directors' report"]
            )
            result["outlook"] = extract_section(
                full_text,
                start_markers=["outlook", "prospects", "looking ahead", "future plans",
                                "going forward", "2024 outlook", "2025 outlook"],
                end_markers=["risk", "governance", "directors", "financial statements",
                             "auditor", "notes to"]
            )

    except Exception as e:
        result["extraction_status"] = f"error: {e}"
        log.error(f"Extraction failed for {pdf_path.name}: {e}")

    return result


import re
def extract_section(text: str, start_markers: list, end_markers: list,
                    max_chars: int = 25000) -> str:
    """
    Extract a section from text given start and end marker phrases.
    Requires markers to start on a new line to avoid mid-sentence truncation.
    """
    text_lower = text.lower()
    start_pos = -1

    for marker in start_markers:
        # Allow marker at the very beginning or preceded by newline
        # Ensure it's a header line: relatively short, ends with newline, and does NOT contain a period after the marker
        pattern = re.compile(r'(?:^|\n)\s*' + re.escape(marker.lower()) + r'[^\n\.]{0,60}(?:\n|$)')
        match = pattern.search(text_lower)
        if match:
            # We want to capture starting from the actual text, not the newline preceding it.
            # match.start() could be the newline character. Let's find the start of the marker itself.
            start_pos = text_lower.find(marker.lower(), match.start())
            break

    if start_pos == -1:
        return ""

    end_pos = len(text)
    search_from = start_pos + 50  # skip the header itself

    for marker in end_markers:
        # Same rule for end markers: must be a standalone header line without a period
        pattern = re.compile(r'\n\s*' + re.escape(marker.lower()) + r'[^\n\.]{0,60}(?:\n|$)')
        match = pattern.search(text_lower, search_from)
        if match and match.start() < end_pos:
            end_pos = match.start()

    section = text[start_pos:min(end_pos, start_pos + max_chars)].strip()
    return section


# ── Metadata Tracker ───────────────────────────────────────────────────────────

class MetadataTracker:
    """Tracks collection progress and document metadata."""

    def __init__(self):
        self.path = METADATA_DIR / "collection_registry.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"discovered": [], "downloaded": [], "extracted": [], "failed": []}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_discovered(self, record: dict):
        # Deduplicate by URL
        urls = {r["url"] for r in self.data["discovered"]}
        if record["url"] not in urls:
            record["discovered_at"] = datetime.now().isoformat()
            self.data["discovered"].append(record)

    def mark_downloaded(self, url: str, local_path: str):
        for r in self.data["discovered"]:
            if r["url"] == url:
                r["local_path"] = local_path
                r["downloaded_at"] = datetime.now().isoformat()
        if url not in self.data["downloaded"]:
            self.data["downloaded"].append(url)

    def mark_extracted(self, url: str, text_path: str, stats: dict):
        for r in self.data["discovered"]:
            if r["url"] == url:
                r["text_path"] = text_path
                r["extraction_stats"] = stats
                r["extracted_at"] = datetime.now().isoformat()
        if url not in self.data["extracted"]:
            self.data["extracted"].append(url)

    def mark_failed(self, url: str, reason: str):
        self.data["failed"].append({"url": url, "reason": reason,
                                    "failed_at": datetime.now().isoformat()})

    def summary(self) -> dict:
        return {
            "total_discovered": len(self.data["discovered"]),
            "total_downloaded": len(self.data["downloaded"]),
            "total_extracted":  len(self.data["extracted"]),
            "total_failed":     len(self.data["failed"]),
        }


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def load_companies(ticker_filter: str | None = None) -> list[dict]:
    with open(COMPANIES_FILE) as f:
        companies = json.load(f)["companies"]
    if ticker_filter:
        companies = [c for c in companies if c["ticker"] == ticker_filter.upper()]
    return companies


def run_discovery(companies: list[dict], tracker: MetadataTracker):
    """Phase 1: Discover all available filing URLs."""
    log.info(f"=== DISCOVERY PHASE: {len(companies)} companies ===")

    for company in companies:
        ticker = company["ticker"]
        log.info(f"Discovering: {ticker} — {company['name']}")

        all_links = []
        all_links += search_company_ir(company)
        time.sleep(REQUEST_DELAY)

        all_links += search_proshare(company["name"], ticker)
        time.sleep(REQUEST_DELAY)

        all_links += search_ngx_portal(ticker)
        time.sleep(REQUEST_DELAY)

        for link in all_links:
            link["company_name"] = company["name"]
            link["sector"]       = company["sector"]
            link["year"]         = infer_year(link.get("title", ""), link["url"])
            tracker.add_discovered(link)

        log.info(f"  -> {ticker}: {len(all_links)} documents found")

    tracker.save()
    log.info(f"Discovery complete. {tracker.summary()}")


def run_download(tracker: MetadataTracker, ticker_filter: str | None = None):
    """Phase 2: Download all discovered PDFs."""
    log.info("=== DOWNLOAD PHASE ===")
    to_download = [
        r for r in tracker.data["discovered"]
        if r["url"] not in tracker.data["downloaded"]
        and r["url"] not in [f["url"] for f in tracker.data["failed"]]
        and (ticker_filter is None or r.get("ticker") == ticker_filter.upper())
        and r.get("doc_type") != "other"
    ]
    log.info(f"Queued {len(to_download)} documents for download")

    for record in to_download:
        ticker   = record["ticker"]
        filename = url_to_filename(ticker, record["url"], record.get("title", "doc"))
        dest     = RAW_PDF_DIR / ticker / filename
        dest.parent.mkdir(exist_ok=True)

        success = download_pdf(record["url"], dest)
        time.sleep(REQUEST_DELAY)

        if success:
            tracker.mark_downloaded(record["url"], str(dest))
        else:
            tracker.mark_failed(record["url"], "download_failed")

    tracker.save()
    log.info(f"Download complete. {tracker.summary()}")


def run_extraction(tracker: MetadataTracker, ticker_filter: str | None = None):
    """Phase 3: Extract narrative text from downloaded PDFs."""
    log.info("=== EXTRACTION PHASE ===")
    to_extract = [
        r for r in tracker.data["discovered"]
        if r["url"] in tracker.data["downloaded"]
        and r["url"] not in tracker.data["extracted"]
        and "local_path" in r
        and (ticker_filter is None or r.get("ticker") == ticker_filter.upper())
    ]
    log.info(f"Queued {len(to_extract)} PDFs for text extraction")

    corpus_records = []

    for record in to_extract:
        pdf_path  = Path(record["local_path"])
        if not pdf_path.exists():
            log.warning(f"  PDF not found: {pdf_path}")
            continue

        log.info(f"  Extracting: {pdf_path.name}")
        sections = extract_narrative_sections(pdf_path)

        # Save extracted text
        ticker    = record["ticker"]
        text_dir  = TEXT_DIR / ticker
        text_dir.mkdir(exist_ok=True)
        text_file = text_dir / (pdf_path.stem + ".json")

        output = {
            **record,
            "sections": sections,
            "char_counts": {
                k: len(v) for k, v in sections.items()
                if isinstance(v, str)
            }
        }

        with open(text_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        stats = {
            "page_count":          sections["page_count"],
            "full_text_chars":     len(sections["full_text"]),
            "chairman_chars":      len(sections["chairman_statement"]),
            "ceo_chars":           len(sections["ceo_review"]),
            "operating_chars":     len(sections["operating_review"]),
            "outlook_chars":       len(sections["outlook"]),
            "extraction_status":   sections["extraction_status"],
        }
        tracker.mark_extracted(record["url"], str(text_file), stats)

        corpus_records.append({
            "ticker":     ticker,
            "company":    record.get("company_name", ""),
            "sector":     record.get("sector", ""),
            "year":       record.get("year", ""),
            "doc_type":   record.get("doc_type", ""),
            "source":     record.get("source", ""),
            "pdf_path":   str(pdf_path),
            "text_path":  str(text_file),
            **stats
        })

    tracker.save()

    # Save master corpus index
    all_corpus = []
    for r in tracker.data["discovered"]:
        if "extraction_stats" in r and "text_path" in r:
            all_corpus.append({
                "ticker":     r.get("ticker", ""),
                "company":    r.get("company_name", ""),
                "sector":     r.get("sector", ""),
                "year":       r.get("year", ""),
                "doc_type":   r.get("doc_type", ""),
                "source":     r.get("source", ""),
                "pdf_path":   r.get("local_path", ""),
                "text_path":  r.get("text_path", ""),
                **r["extraction_stats"]
            })

    if all_corpus:
        corpus_df = pd.DataFrame(all_corpus)
        corpus_df.to_csv(METADATA_DIR / "corpus_index.csv", index=False)
        log.info(f"  Corpus index saved: {len(corpus_df)} documents")

    log.info(f"Extraction complete. {tracker.summary()}")


def print_summary(tracker: MetadataTracker):
    """Print a readable summary of collection status."""
    s = tracker.summary()
    print("\n" + "="*50)
    print("NGX-FND Collection Summary")
    print("="*50)
    print(f"  Documents discovered : {s['total_discovered']}")
    print(f"  PDFs downloaded      : {s['total_downloaded']}")
    print(f"  Texts extracted      : {s['total_extracted']}")
    print(f"  Failed               : {s['total_failed']}")

    if tracker.data["discovered"]:
        df = pd.DataFrame(tracker.data["discovered"])
        if "sector" in df.columns:
            print("\nBy sector:")
            print(df.groupby("sector").size().to_string())
        if "doc_type" in df.columns:
            print("\nBy document type:")
            print(df.groupby("doc_type").size().to_string())
        if "year" in df.columns:
            print("\nBy year:")
            print(df.groupby("year").size().to_string())
    print("="*50 + "\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NGX Financial Disclosure Collector")
    parser.add_argument("--mode",   choices=["discover","download","extract","full","summary"],
                        default="summary")
    parser.add_argument("--ticker", help="Run for a single ticker only (e.g. MTNN)")
    args = parser.parse_args()

    tracker   = MetadataTracker()
    companies = load_companies(ticker_filter=args.ticker)

    if args.mode in ("discover", "full"):
        run_discovery(companies, tracker)

    if args.mode in ("download", "full"):
        run_download(tracker, ticker_filter=args.ticker)

    if args.mode in ("extract", "full"):
        run_extraction(tracker, ticker_filter=args.ticker)

    print_summary(tracker)


if __name__ == "__main__":
    main()
