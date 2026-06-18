"""
NGX-FND Abnormal Returns Calculator
=====================================
Implements event study methodology to compute abnormal returns
around corporate disclosure dates on the Nigerian Exchange.

Methodology:
  1. Market Model: AR = R_i,t - (alpha_i + beta_i * R_m,t)
     - Estimation window: [-250, -30] trading days before event
     - Event window: [-1, +1], [-1, +3], [0, +3], [0, +5]

  2. Buy-and-Hold Abnormal Returns (BHAR):
     BHAR = prod(1 + R_i,t) - prod(1 + R_m,t)  over event window

  3. Cumulative Abnormal Returns (CAR):
     CAR = sum(AR_i,t)  over event window

  4. Fama-French 3-Factor adjustment (where factor data available)

Outputs:
  data/prices/abnormal_returns.csv    -- CAR/BHAR per event
  data/prices/event_windows.csv       -- daily ARs per event

Usage:
  python abnormal_returns.py
  python abnormal_returns.py --window 5    # use [-1,+5] event window
"""

import json
import warnings
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
PRICES_DIR   = BASE_DIR / "data" / "prices"
METADATA_DIR = BASE_DIR / "data" / "metadata"
LOG_DIR      = BASE_DIR / "logs"
COMPANIES_FILE = BASE_DIR / "data" / "companies.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"abnormal_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Canonical sector lookup ────────────────────────────────────────────────────

def load_sector_map() -> dict:
    """Return {ticker: sector} from companies.json (canonical source of truth)."""
    if not COMPANIES_FILE.exists():
        return {}
    with open(COMPANIES_FILE) as f:
        companies = json.load(f)["companies"]
    return {c["ticker"]: c["sector"] for c in companies}

# ── Config ─────────────────────────────────────────────────────────────────────
ESTIMATION_START = -250   # trading days before event
ESTIMATION_END   = -30    # trading days before event (buffer to avoid anticipation)
MIN_ESTIMATION_OBS = 100  # minimum observations for reliable beta estimation

EVENT_WINDOWS = {
    "CAR[-1,+1]": (-1,  1),
    "CAR[0,+1]":  ( 0,  1),
    "CAR[0,+3]":  ( 0,  3),
    "CAR[0,+5]":  ( 0,  5),
    "CAR[-1,+3]": (-1,  3),
    "CAR[-1,+5]": (-1,  5),
}


# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_returns() -> pd.DataFrame | None:
    path = PRICES_DIR / "simple_returns.csv"
    if not path.exists():
        log.error(f"Returns not found: {path}. Run price_collector.py --fetch first.")
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def load_events() -> pd.DataFrame | None:
    path = PRICES_DIR / "disclosure_events.csv"
    if not path.exists():
        log.error(f"Events not found: {path}. Run price_collector.py --events first.")
        return None
    df = pd.read_csv(path, parse_dates=["estimated_date"])
    df["event_date"] = pd.to_datetime(
        df["actual_date"].fillna(df["estimated_date"])
    )
    return df


def load_llm_predictions() -> pd.DataFrame | None:
    """Load the best available LLM predictions to join with abnormal returns."""
    # Look for most recent combined_5shot run
    results_dir = BASE_DIR / "data" / "llm_results"
    if not results_dir.exists():
        return None
    runs = sorted([d for d in results_dir.iterdir() if d.is_dir()
                   and "combined_5shot" in d.name], reverse=True)
    if not runs:
        runs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        return None
    pred_path = runs[0] / "predictions.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path, dtype=str)
    df = df[df["error"].isna() | (df["error"] == "")]
    log.info(f"Loaded predictions from: {runs[0].name} ({len(df)} rows)")
    return df


# ── Market Model ───────────────────────────────────────────────────────────────

def estimate_market_model(stock_returns: pd.Series,
                           market_returns: pd.Series,
                           event_idx: int,
                           est_start: int = ESTIMATION_START,
                           est_end:   int = ESTIMATION_END) -> dict | None:
    """
    Estimate market model parameters (alpha, beta) using OLS
    over the estimation window.

    Returns dict with alpha, beta, r_squared, std_error, n_obs
    or None if insufficient data.
    """
    # Align on common dates
    aligned = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()

    if event_idx >= len(aligned):
        return None

    est_slice = aligned.iloc[max(0, event_idx + est_start): event_idx + est_end]

    if len(est_slice) < MIN_ESTIMATION_OBS:
        return None

    y = est_slice["stock"].values
    X = sm.add_constant(est_slice["market"].values)

    try:
        model   = sm.OLS(y, X).fit()
        alpha   = model.params[0]
        beta    = model.params[1]
        r2      = model.rsquared
        std_err = model.resid.std()
        return {
            "alpha":     alpha,
            "beta":      beta,
            "r_squared": r2,
            "std_error": std_err,
            "n_obs":     len(est_slice),
        }
    except Exception as e:
        log.debug(f"OLS failed: {e}")
        return None


def compute_event_abnormal_returns(stock_returns: pd.Series,
                                   market_returns: pd.Series,
                                   event_date: pd.Timestamp,
                                   params: dict,
                                   max_window: int = 10) -> pd.DataFrame:
    """
    Compute daily abnormal returns around an event date.
    AR_t = R_i,t - (alpha + beta * R_m,t)
    """
    # Get trading days around event
    aligned = pd.DataFrame({
        "stock":  stock_returns,
        "market": market_returns
    }).dropna()

    # Find event date index (use nearest trading day)
    dates = aligned.index
    event_loc = dates.searchsorted(event_date)
    if event_loc >= len(dates):
        return pd.DataFrame()

    start_loc = max(0, event_loc - max_window)
    end_loc   = min(len(dates), event_loc + max_window + 1)

    window_data = aligned.iloc[start_loc:end_loc].copy()
    window_data["day_relative"] = range(start_loc - event_loc, end_loc - event_loc)

    # Compute AR
    window_data["expected_return"] = (
        params["alpha"] + params["beta"] * window_data["market"]
    )
    window_data["AR"] = window_data["stock"] - window_data["expected_return"]

    return window_data[["day_relative", "stock", "market", "expected_return", "AR"]]


def compute_bhar(stock_returns: pd.Series,
                 market_returns: pd.Series,
                 event_date: pd.Timestamp,
                 window_days: int = 5) -> float | None:
    """
    Compute Buy-and-Hold Abnormal Return over event window.
    BHAR = prod(1 + R_i) - prod(1 + R_m)
    """
    aligned = pd.DataFrame({
        "stock":  stock_returns,
        "market": market_returns
    }).dropna()

    dates    = aligned.index
    ev_loc   = dates.searchsorted(event_date)
    if ev_loc >= len(dates):
        return None

    end_loc  = min(len(dates), ev_loc + window_days + 1)
    window   = aligned.iloc[ev_loc:end_loc]

    if len(window) < 2:
        return None

    stock_bhr  = (1 + window["stock"]).prod() - 1
    market_bhr = (1 + window["market"]).prod() - 1
    return round(stock_bhr - market_bhr, 6)


# ── CAR Aggregation ────────────────────────────────────────────────────────────

def compute_all_cars(ar_df: pd.DataFrame, event_windows: dict) -> dict:
    """
    Compute Cumulative Abnormal Returns for all defined event windows.
    """
    cars = {}
    for window_label, (w_start, w_end) in event_windows.items():
        window_data = ar_df[
            (ar_df["day_relative"] >= w_start) &
            (ar_df["day_relative"] <= w_end)
        ]
        if len(window_data) > 0:
            cars[window_label] = round(window_data["AR"].sum(), 6)
        else:
            cars[window_label] = np.nan
    return cars


# ── Sentiment Score Mapping ────────────────────────────────────────────────────

SENTIMENT_SCORE_MAP = {
    # Cardinal mapping
    "positive": 1,
    "negative": -1,
    "neutral":  0,
}

INTENSITY_MULTIPLIER = {
    "mild":     0.5,
    "moderate": 1.0,
    "strong":   1.5,
}

GUIDANCE_TYPE_SCORE = {
    "positive":    1,
    "negative":   -1,
    "neutral":     0,
    "conditional": 0.5,
    "none":        0,
}


def build_sentiment_scores(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LLM predictions to company-year level sentiment scores.
    Uses best model (5-shot combined) predictions.

    Returns DataFrame with ticker, year, and multiple sentiment score columns.
    """
    df = predictions_df.copy()

    # Filter to best available prompt (5-shot combined preferred)
    if "prompt_key" in df:
        for preferred in ["combined_5shot", "combined_3shot", "combined_0shot",
                          "sentiment_5shot", "sentiment_3shot"]:
            subset = df[df["prompt_key"] == preferred]
            if len(subset) > 10:
                df = subset
                log.info(f"Using prompt: {preferred}")
                break

    # Use highest-accuracy model if available (set after running metrics.py)
    # Default to GPT-4o -> Claude -> Gemini priority
    for preferred_model in ["gpt4o", "claude", "gemini"]:
        subset = df[df["model"] == preferred_model]
        if len(subset) > 10:
            df = subset
            log.info(f"Using model: {preferred_model}")
            break

    # Map to numeric scores
    df["sentiment_score"] = df["pred_sentiment"].map(SENTIMENT_SCORE_MAP).fillna(0)
    df["intensity_mult"]  = df["pred_intensity"].map(INTENSITY_MULTIPLIER).fillna(1.0)
    df["weighted_score"]  = df["sentiment_score"] * df["intensity_mult"]
    df["guidance_score"]  = df["pred_guid_type"].map(GUIDANCE_TYPE_SCORE).fillna(0)

    # Aggregate to ticker × year
    agg_df = df.groupby(["ticker", "year"]).agg(
        sentiment_score_mean   = ("sentiment_score",  "mean"),
        sentiment_score_sum    = ("sentiment_score",  "sum"),
        weighted_score_mean    = ("weighted_score",   "mean"),
        guidance_score_mean    = ("guidance_score",   "mean"),
        guidance_rate          = ("pred_guidance",    lambda x: (x.str.lower() == "true").mean()),
        n_passages             = ("passage_id",       "count"),
        pct_positive           = ("pred_sentiment",   lambda x: (x == "positive").mean()),
        pct_negative           = ("pred_sentiment",   lambda x: (x == "negative").mean()),
        pct_neutral            = ("pred_sentiment",   lambda x: (x == "neutral").mean()),
    ).reset_index()

    # Net sentiment index: (positive - negative) / total
    agg_df["net_sentiment"] = (agg_df["pct_positive"] - agg_df["pct_negative"])

    # Guidance-adjusted sentiment
    agg_df["guidance_adj_sentiment"] = (
        agg_df["weighted_score_mean"] * 0.6 +
        agg_df["guidance_score_mean"] * 0.4
    )

    return agg_df


# ── Main Event Study ───────────────────────────────────────────────────────────

def run_event_study() -> pd.DataFrame:
    """
    Full event study pipeline:
    1. For each company × event date:
       a. Estimate market model on estimation window
       b. Compute AR for event window days
       c. Aggregate to CAR and BHAR
    2. Join with LLM sentiment scores
    3. Save results
    """
    returns_df  = load_returns()
    events_df   = load_events()
    predictions = load_llm_predictions()

    if returns_df is None or events_df is None:
        log.error("Missing price or event data. Run price_collector.py first.")
        return pd.DataFrame()

    if "market_return" not in returns_df.columns:
        log.warning("Market index not found in returns. Using equal-weight market proxy.")
        stock_cols = [c for c in returns_df.columns if c != "market_return"]
        returns_df["market_return"] = returns_df[stock_cols].mean(axis=1)

    market_returns = returns_df["market_return"].dropna()

    results     = []
    ar_windows  = []

    tickers_with_data = set(returns_df.columns) - {"market_return"}
    events_to_process = events_df[events_df["ticker"].isin(tickers_with_data)]

    log.info(f"Processing {len(events_to_process)} events for {events_to_process['ticker'].nunique()} tickers")

    for _, event in events_to_process.iterrows():
        ticker      = event["ticker"]
        event_date  = event["event_date"]
        fiscal_year = event["fiscal_year"]
        doc_type    = event["doc_type"]

        if ticker not in returns_df.columns:
            continue

        stock_returns = returns_df[ticker].dropna()
        aligned = stock_returns.index.intersection(market_returns.index)
        if len(aligned) < MIN_ESTIMATION_OBS + 10:
            continue

        stock_r  = stock_returns.reindex(aligned)
        market_r = market_returns.reindex(aligned)

        # Find event location
        dates    = pd.Series(aligned)
        ev_loc   = aligned.searchsorted(event_date)
        if ev_loc >= len(aligned):
            continue

        # Estimate market model
        params = estimate_market_model(stock_r, market_r, ev_loc)
        if params is None:
            log.debug(f"  Skipping {ticker} {fiscal_year}: insufficient estimation window")
            continue

        # Compute AR series for event window
        ar_df = compute_event_abnormal_returns(stock_r, market_r, event_date, params)
        if ar_df.empty:
            continue

        # Add event metadata
        ar_df["ticker"]      = ticker
        ar_df["fiscal_year"] = fiscal_year
        ar_df["doc_type"]    = doc_type
        ar_df["event_date"]  = event_date
        ar_windows.append(ar_df)

        # Compute CARs
        cars  = compute_all_cars(ar_df, EVENT_WINDOWS)
        bhar5 = compute_bhar(stock_r, market_r, event_date, window_days=5)

        row = {
            "ticker":          ticker,
            "company":         event.get("company", ""),
            "sector":          event.get("sector", ""),
            "fiscal_year":     fiscal_year,
            "doc_type":        doc_type,
            "event_date":      event_date,
            "date_source":     event.get("date_source", "estimated"),
            "alpha":           round(params["alpha"],     6),
            "beta":            round(params["beta"],      4),
            "r_squared":       round(params["r_squared"], 4),
            "std_error":       round(params["std_error"], 6),
            "est_n_obs":       params["n_obs"],
            "BHAR[0,+5]":      bhar5,
            **cars,
        }
        results.append(row)

    if not results:
        log.warning("No events processed. Check that price data and events overlap.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # ── Fix sector to canonical values from companies.json ──────────────────
    sector_map = load_sector_map()
    if sector_map:
        results_df["sector"] = results_df["ticker"].map(sector_map)
        log.info(f"Sector fix applied from companies.json "
                 f"({results_df['sector'].notna().sum()}/{len(results_df)} mapped)")

    # Join with LLM sentiment scores
    if predictions is not None:
        sentiment_df = build_sentiment_scores(predictions)
        sentiment_df["fiscal_year"] = sentiment_df["year"].astype(str)
        results_df["fiscal_year"]   = results_df["fiscal_year"].astype(str)
        results_df = results_df.merge(
            sentiment_df, on=["ticker", "fiscal_year"], how="left"
        )
        log.info(f"Sentiment scores joined: {results_df['sentiment_score_mean'].notna().sum()} events matched")

    # Save
    out_path = PRICES_DIR / "abnormal_returns.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"\nAbnormal returns saved: {out_path} ({len(results_df)} events)")

    if ar_windows:
        all_ar = pd.concat(ar_windows)
        all_ar.to_csv(PRICES_DIR / "event_windows.csv", index=False)
        log.info(f"Event window daily ARs saved: {PRICES_DIR / 'event_windows.csv'}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Abnormal Returns Calculator")
    parser.add_argument("--window", type=int, default=5,
                        help="Primary event window in days post-disclosure (default: 5)")
    args = parser.parse_args()
    run_event_study()


if __name__ == "__main__":
    main()
