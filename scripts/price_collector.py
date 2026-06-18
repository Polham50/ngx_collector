"""
NGX-FND Price Data Collector
==============================
Fetches historical stock price data for NGX-listed companies
to compute post-disclosure abnormal returns.

Sources (in priority order):
  1. yfinance — NGX tickers use suffix .LG (Lagos) e.g. MTNN.LG
  2. Manual CSV fallback (for tickers not available on yfinance)

Outputs:
  data/prices/raw_prices.csv       — daily OHLCV for all tickers
  data/prices/returns.csv          — daily log returns
  data/prices/disclosure_events.csv — disclosure dates per company

Usage:
  python price_collector.py --fetch       # Download all prices
  python price_collector.py --events      # Build disclosure event table
  python price_collector.py --full        # Both steps
"""

import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
PRICES_DIR    = DATA_DIR / "prices"
METADATA_DIR  = DATA_DIR / "metadata"
LOG_DIR       = BASE_DIR / "logs"
COMPANIES_FILE= DATA_DIR / "companies.json"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
START_DATE    = "2019-01-01"   # one year before corpus start for Fama-French window
END_DATE      = "2024-12-31"
REQUEST_DELAY = 1.5

# NGX tickers mapped to yfinance suffixes
# NGX stocks trade on Lagos exchange — suffix .LG on some platforms
# Many NGX stocks are also available without suffix or via OTC
TICKER_MAP = {
    "ZENITHBANK": "ZENITHBANK.LG",
    "GTCO":       "GTCO.LG",
    "ACCESS":     "ACCESS.LG",
    "UBA":        "UBA.LG",
    "FBNH":       "FBNH.LG",
    "STANBIC":    "STANBIC.LG",
    "FIDELITYBK": "FIDELITYBK.LG",
    "FCMB":       "FCMB.LG",
    "STERLING":   "STERLING.LG",
    "WEMA":       "WEMA.LG",
    "SEPLAT":     "SEPLAT.LG",
    "OANDO":      "OANDO.LG",
    "TOTAL":      "TOTAL.LG",
    "ARDOVA":     "ARDOVA.LG",
    "CONOIL":     "CONOIL.LG",
    "DANGCEM":    "DANGCEM.LG",
    "BUACEMENT":  "BUACEMENT.LG",
    "WAPCO":      "WAPCO.LG",
    "MTNN":       "MTNN.LG",
    "AIRTELAFRI": "AIRTELAFRI.LG",
    "NESTLE":     "NESTLE.LG",
    "UNILEVER":   "UNILEVER.LG",
    "DANGSUGAR":  "DANGSUGAR.LG",
    "FLOURMILL":  "FLOURMILL.LG",
    "GUINNESS":   "GUINNESS.LG",
    "NB":         "NB.LG",
    "CADBURY":    "CADBURY.LG",
    "PRESCO":     "PRESCO.LG",
    "OKOMUOIL":   "OKOMUOIL.LG",
    "NASCON":     "NASCON.LG",
    "TRANSCORP":  "TRANSCORP.LG",
    "BUAFOODS":   "BUAFOODS.LG",
    "GEREGU":     "GEREGU.LG",
    "ARADEL":     "ARADEL.LG",
    "BERGER":     "BERGER.LG",
    "VITAFOAM":   "VITAFOAM.LG",
    "INTBREW":    "INTBREW.LG",
}

# NGX All-Share Index proxy (use as market benchmark)
MARKET_INDEX = "^NGSEINDX"   # NGX All-Share Index on Yahoo Finance


# ── Price Fetcher ──────────────────────────────────────────────────────────────

def fetch_ticker_prices(ticker: str, yf_ticker: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV for a single ticker."""
    try:
        t = yf.Ticker(yf_ticker)
        df = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            log.warning(f"  [{ticker}] No data returned from yfinance ({yf_ticker})")
            return None
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df["ticker"] = ticker
        log.info(f"  [{ticker}] {len(df)} days fetched ({df.index.min().date()} – {df.index.max().date()})")
        return df
    except Exception as e:
        log.warning(f"  [{ticker}] yfinance error: {e}")
        return None


def fetch_market_index() -> pd.DataFrame | None:
    """Fetch NGX All-Share Index as market benchmark."""
    try:
        t = yf.Ticker(MARKET_INDEX)
        df = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            # Fallback: try NGX ETF or MSCI Nigeria
            log.warning("NGX All-Share not found — trying fallback MSCI Nigeria ETF")
            t = yf.Ticker("NGE")   # VanEck Nigeria ETF
            df = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            return None
        df = df[["Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.columns = ["market_close"]
        log.info(f"  Market index: {len(df)} days fetched")
        return df
    except Exception as e:
        log.warning(f"  Market index fetch failed: {e}")
        return None


def fetch_all_prices() -> pd.DataFrame:
    """Fetch prices for all tickers and combine into a single DataFrame."""
    all_frames = []

    log.info(f"=== FETCHING PRICES: {len(TICKER_MAP)} tickers ===")
    log.info(f"Period: {START_DATE} – {END_DATE}")

    for ticker, yf_ticker in TICKER_MAP.items():
        df = fetch_ticker_prices(ticker, yf_ticker)
        if df is not None:
            all_frames.append(df)
        time.sleep(REQUEST_DELAY)

    if not all_frames:
        log.error("No price data fetched. Check internet connection and ticker symbols.")
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    out_path = PRICES_DIR / "raw_prices.csv"
    combined.to_csv(out_path)
    log.info(f"\nRaw prices saved: {out_path} ({len(combined)} rows, {combined['ticker'].nunique()} tickers)")
    return combined


# ── Returns Computation ────────────────────────────────────────────────────────

def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns and simple returns for all tickers.
    Also fetches market index and computes excess returns.
    """
    log.info("Computing returns...")

    # Pivot to wide format: date × ticker
    close_wide = prices_df.pivot_table(index="date", columns="ticker", values="Close")
    close_wide = close_wide.sort_index()

    # Log returns
    log_returns = np.log(close_wide / close_wide.shift(1))
    # Simple returns
    simple_returns = close_wide.pct_change()

    # Market index
    market_df = fetch_market_index()
    if market_df is not None:
        market_returns = np.log(market_df["market_close"] / market_df["market_close"].shift(1))
        market_returns.name = "market_return"
        log_returns   = log_returns.join(market_returns, how="left")
        simple_returns = simple_returns.join(market_returns, how="left")

    # Save
    log_returns.to_csv(PRICES_DIR / "log_returns.csv")
    simple_returns.to_csv(PRICES_DIR / "simple_returns.csv")

    log.info(f"Returns saved: {len(log_returns)} trading days × {len(log_returns.columns)} series")
    return log_returns


# ── Disclosure Event Table ─────────────────────────────────────────────────────

def build_disclosure_events() -> pd.DataFrame:
    """
    Build a table of disclosure dates per company-year.
    Uses approximate fiscal year-end + typical filing lag for Nigerian companies:
      - Annual reports: typically filed 3–4 months after fiscal year end
      - FY Dec companies: reports typically released March–April
      - Interim results: typically August–September
    Returns a DataFrame with company, year, doc_type, estimated_date.
    """
    log.info("Building disclosure event table...")

    with open(COMPANIES_FILE) as f:
        companies = json.load(f)["companies"]

    # Nigerian fiscal year patterns
    # Most NGX companies have Dec 31 year-end
    # Banks often release full-year results in March
    # Consumer goods: April-May
    # Oil & Gas: March-April

    FILING_LAG = {  # months after FY end
        "Banking":       3,   # March
        "Oil & Gas":     4,   # April
        "Consumer Goods":4,   # April-May
        "Telecoms":      3,   # March
        "Industrial":    4,   # April
    }

    INTERIM_MONTH = {  # month interim results typically released
        "Banking":       8,   # August
        "Oil & Gas":     9,   # September
        "Consumer Goods":9,
        "Telecoms":      8,
        "Industrial":    9,
    }

    rows = []
    target_years = list(range(2020, 2025))

    for company in companies:
        ticker = company["ticker"]
        sector = company["sector"]
        lag    = FILING_LAG.get(sector, 4)
        int_m  = INTERIM_MONTH.get(sector, 9)

        for year in target_years:
            # Annual report disclosure (approx)
            annual_month = lag
            annual_date  = pd.Timestamp(year=year, month=annual_month, day=15)

            rows.append({
                "ticker":          ticker,
                "company":         company["name"],
                "sector":          sector,
                "fiscal_year":     year,
                "doc_type":        "annual_report",
                "estimated_date":  annual_date,
                "date_source":     "estimated",
                "actual_date":     None,   # to be filled from corpus metadata
            })

            # Interim (H1) disclosure
            interim_date = pd.Timestamp(year=year, month=int_m, day=15)
            rows.append({
                "ticker":          ticker,
                "company":         company["name"],
                "sector":          sector,
                "fiscal_year":     year,
                "doc_type":        "interim_report",
                "estimated_date":  interim_date,
                "date_source":     "estimated",
                "actual_date":     None,
            })

    events_df = pd.DataFrame(rows)

    # Try to enrich with actual dates from corpus metadata if available
    corpus_path = METADATA_DIR / "corpus_index.csv"
    if corpus_path.exists():
        corpus_df = pd.read_csv(corpus_path)
        if "disclosure_date" in corpus_df.columns:
            date_map = corpus_df.set_index(["ticker","year","doc_type"])["disclosure_date"].to_dict()
            for idx, row in events_df.iterrows():
                key = (row["ticker"], str(row["fiscal_year"]), row["doc_type"])
                if key in date_map:
                    events_df.at[idx, "actual_date"]  = date_map[key]
                    events_df.at[idx, "date_source"] = "actual"

    out_path = PRICES_DIR / "disclosure_events.csv"
    events_df.to_csv(out_path, index=False)
    log.info(f"Disclosure events saved: {out_path} ({len(events_df)} events)")
    return events_df


def main():
    parser = argparse.ArgumentParser(description="NGX Price Data Collector")
    parser.add_argument("--fetch",  action="store_true", help="Fetch raw prices")
    parser.add_argument("--events", action="store_true", help="Build disclosure event table")
    parser.add_argument("--full",   action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.fetch or args.full:
        prices = fetch_all_prices()
        if not prices.empty:
            compute_returns(prices)

    if args.events or args.full:
        build_disclosure_events()

    if not any([args.fetch, args.events, args.full]):
        parser.print_help()


if __name__ == "__main__":
    main()
