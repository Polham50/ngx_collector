"""
NGX Price Data Collector
========================
Collects historical stock price data for NGX-listed companies
to support event study / return correlation analysis (RQ3).

Sources (in priority order):
  1. Stanbic IBTC / Meristem public data endpoints
  2. NGX Group market data page (web scrape)
  3. Proshare market data
  4. Yahoo Finance (via yfinance) as fallback — NGX tickers use .LG suffix

Usage:
  python ngx_price_collector.py --mode fetch        # Fetch all price data
  python ngx_price_collector.py --mode event_windows # Compute event windows
  python ngx_price_collector.py --ticker MTNN        # Single company
"""

import os
import json
import time
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
PRICE_DIR    = DATA_DIR / "prices"
META_DIR     = DATA_DIR / "metadata"
LOG_DIR      = BASE_DIR / "logs"
COMPANIES_F  = DATA_DIR / "companies.json"

PRICE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html, */*",
}
REQUEST_DELAY = 2.0
START_DATE    = "2020-01-01"
END_DATE      = "2024-12-31"


# ── Yahoo Finance fallback ─────────────────────────────────────────────────────

def fetch_via_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch NGX prices from Yahoo Finance.
    NGX tickers on Yahoo use the format: TICKER.LG (Lagos Stock Exchange).
    E.g., MTNN.LG, ZENITHBANK.LG
    """
    try:
        import yfinance as yf
        yf_ticker = f"{ticker}.LG"
        log.info(f"  [yfinance] fetching {yf_ticker}")
        df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            log.warning(f"  [yfinance] no data for {yf_ticker}")
            return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Date": "date", "Close": "close",
                                 "Open": "open", "High": "high",
                                 "Low": "low", "Volume": "volume"})
        df["ticker"] = ticker
        df["source"] = "yfinance"
        log.info(f"  [yfinance] {ticker}: {len(df)} rows")
        return df[["date", "ticker", "open", "high", "low", "close", "volume", "source"]]
    except ImportError:
        log.warning("  yfinance not installed. Run: pip install yfinance --break-system-packages")
        return None
    except Exception as e:
        log.warning(f"  [yfinance] {ticker} failed: {e}")
        return None


def fetch_via_proshare(ticker: str) -> pd.DataFrame | None:
    """Scrape price history from Proshare market data pages."""
    results = []
    url = f"https://www.proshareng.com/stocks/{ticker}/pricehistory"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table")
        if not table:
            return None

        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 4:
                try:
                    results.append({
                        "date":   pd.to_datetime(cols[0], dayfirst=True),
                        "ticker": ticker,
                        "open":   float(cols[1].replace(",", "")),
                        "high":   float(cols[2].replace(",", "")),
                        "low":    float(cols[3].replace(",", "")),
                        "close":  float(cols[4].replace(",", "")) if len(cols) > 4 else None,
                        "volume": float(cols[5].replace(",", "")) if len(cols) > 5 else None,
                        "source": "proshare",
                    })
                except (ValueError, IndexError):
                    continue

        if results:
            df = pd.DataFrame(results)
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
            log.info(f"  [proshare] {ticker}: {len(df)} rows")
            return df
    except Exception as e:
        log.warning(f"  [proshare] {ticker} failed: {e}")
    return None


def fetch_ngx_market_data(ticker: str) -> pd.DataFrame | None:
    """Fetch price data from NGX Group market data portal."""
    url = f"https://ngxgroup.com/exchange/data/equities-price-list/?ticker={ticker}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table", {"class": lambda x: x and "price" in x.lower()})
        if not table:
            table = soup.find("table")
        if not table:
            return None

        rows = table.find_all("tr")
        results = []
        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 3:
                try:
                    results.append({
                        "date":   pd.to_datetime(cols[0], dayfirst=True),
                        "ticker": ticker,
                        "close":  float(cols[-1].replace(",", "")),
                        "source": "ngx_portal",
                    })
                except (ValueError, IndexError):
                    continue

        if results:
            df = pd.DataFrame(results)
            log.info(f"  [ngx_portal] {ticker}: {len(df)} rows")
            return df
    except Exception as e:
        log.warning(f"  [ngx_portal] {ticker} failed: {e}")
    return None


# ── Main fetch logic ───────────────────────────────────────────────────────────

def fetch_price_data(ticker: str) -> pd.DataFrame | None:
    """
    Try each data source in priority order. Return first successful result.
    """
    # Try yfinance first (most reliable for historical data)
    df = fetch_via_yfinance(ticker, START_DATE, END_DATE)
    if df is not None and not df.empty:
        return df
    time.sleep(REQUEST_DELAY)

    # Try Proshare
    df = fetch_via_proshare(ticker)
    if df is not None and not df.empty:
        return df
    time.sleep(REQUEST_DELAY)

    # Try NGX portal
    df = fetch_ngx_market_data(ticker)
    if df is not None and not df.empty:
        return df

    log.warning(f"  No price data found for {ticker} from any source")
    return None


def save_price_data(ticker: str, df: pd.DataFrame):
    path = PRICE_DIR / f"{ticker}_prices.csv"
    df.to_csv(path, index=False)
    log.info(f"  Saved: {path}")


def load_price_data(ticker: str) -> pd.DataFrame | None:
    path = PRICE_DIR / f"{ticker}_prices.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    return None


# ── Event Window Calculator ────────────────────────────────────────────────────

def compute_abnormal_returns(
    ticker: str,
    event_date: str,
    windows: list[int] = [1, 3, 5],
    estimation_window: int = 120
) -> dict:
    """
    Compute Cumulative Abnormal Returns (CAR) around a disclosure event date.

    Method: Market-adjusted model
      AR(t) = R(t) - R_market(t)
      CAR = sum of AR over event window

    Returns dict of CARs for each window size.
    """
    df = load_price_data(ticker)
    if df is None or df.empty:
        return {"error": "no_price_data"}

    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()

    event_dt = pd.to_datetime(event_date)

    # Find event index
    df["date"] = pd.to_datetime(df["date"])
    event_idx_arr = df.index[df["date"] >= event_dt].tolist()
    if not event_idx_arr:
        return {"error": "event_date_out_of_range"}
    event_idx = event_idx_arr[0]

    # Estimation window: [-estimation_window-1, -2] trading days before event
    est_start = max(0, event_idx - estimation_window - 1)
    est_end   = max(0, event_idx - 2)
    est_returns = df.iloc[est_start:est_end]["return"].dropna()

    # Expected return = mean return over estimation window (simple market model)
    expected_return = est_returns.mean() if len(est_returns) > 10 else 0.0

    results = {"ticker": ticker, "event_date": event_date,
               "expected_daily_return": round(expected_return, 6)}

    for w in windows:
        post_window = df.iloc[event_idx: event_idx + w]
        if post_window.empty:
            results[f"CAR_{w}d"] = None
            continue
        actual_returns   = post_window["return"].fillna(0)
        abnormal_returns = actual_returns - expected_return
        car = abnormal_returns.sum()
        results[f"CAR_{w}d"] = round(car, 6)
        results[f"n_days_{w}d"] = len(post_window)

    return results


def build_event_study_dataset(corpus_index_path: Path) -> pd.DataFrame:
    """
    Join corpus metadata with price event windows to build the
    final dataset for RQ3 regression analysis.
    """
    if not corpus_index_path.exists():
        log.error(f"Corpus index not found: {corpus_index_path}")
        return pd.DataFrame()

    corpus = pd.read_csv(corpus_index_path)
    corpus = corpus[corpus["year"].notna()]
    results = []

    for _, row in corpus.iterrows():
        ticker = row["ticker"]
        year   = str(int(row["year"]))
        # Approximate disclosure date: use fiscal year end + 90 days
        # (Nigerian companies typically file 60–90 days after year end)
        try:
            approx_event_date = pd.to_datetime(f"{year}-12-31") + timedelta(days=90)
            event_str = approx_event_date.strftime("%Y-%m-%d")
        except Exception:
            continue

        cars = compute_abnormal_returns(ticker, event_str)
        results.append({
            **row.to_dict(),
            **cars
        })
        time.sleep(0.1)

    df_out = pd.DataFrame(results)
    out_path = META_DIR / "event_study_dataset.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"Event study dataset saved: {out_path} ({len(df_out)} rows)")
    return df_out


# ── Summary ────────────────────────────────────────────────────────────────────

def print_price_summary():
    files = list(PRICE_DIR.glob("*.csv"))
    print(f"\n{'='*50}")
    print(f"Price Data Summary — {len(files)} tickers collected")
    print(f"{'='*50}")
    total_rows = 0
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            rows = len(df)
            total_rows += rows
            date_range = ""
            if "date" in df.columns and rows > 0:
                date_range = f"  [{df['date'].min()} → {df['date'].max()}]"
            print(f"  {f.stem:<30} {rows:>5} rows{date_range}")
        except Exception:
            print(f"  {f.stem:<30} [error reading file]")
    print(f"\n  Total rows: {total_rows}")
    print(f"{'='*50}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NGX Price Data Collector")
    parser.add_argument("--mode",   choices=["fetch", "event_windows", "summary", "full"],
                        default="summary")
    parser.add_argument("--ticker", help="Single ticker (e.g. MTNN)")
    args = parser.parse_args()

    with open(COMPANIES_F) as f:
        companies = json.load(f)["companies"]

    if args.ticker:
        companies = [c for c in companies if c["ticker"] == args.ticker.upper()]

    if args.mode in ("fetch", "full"):
        log.info(f"=== PRICE FETCH: {len(companies)} companies ===")
        for company in companies:
            ticker = company["ticker"]
            existing = load_price_data(ticker)
            if existing is not None and len(existing) > 100:
                log.info(f"  [skip] {ticker}: already have {len(existing)} rows")
                continue
            log.info(f"Fetching: {ticker}")
            df = fetch_price_data(ticker)
            if df is not None:
                save_price_data(ticker, df)
            time.sleep(REQUEST_DELAY)

    if args.mode in ("event_windows", "full"):
        corpus_path = META_DIR / "corpus_index.csv"
        log.info("=== EVENT WINDOW COMPUTATION ===")
        build_event_study_dataset(corpus_path)

    if args.mode in ("summary", "full"):
        print_price_summary()


if __name__ == "__main__":
    main()
