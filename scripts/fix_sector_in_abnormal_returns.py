"""
Fix Sector in abnormal_returns.csv
====================================
One-time repair: reads the existing abnormal_returns.csv and overwrites the
'sector' column with the canonical sector from companies.json (keyed on ticker).

Also fixes any sentinel `pred_sentiment` values that may have slipped through
from the LLM predictions join.

Usage:
    python fix_sector_in_abnormal_returns.py
"""

import json
from pathlib import Path
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent.parent
PRICES_DIR   = BASE_DIR / "data" / "prices"
COMPANIES_FILE = BASE_DIR / "data" / "companies.json"
AR_FILE      = PRICES_DIR / "abnormal_returns.csv"

def main():
    # Load canonical sector map
    with open(COMPANIES_FILE) as f:
        companies = json.load(f)["companies"]
    sector_map = {c["ticker"]: c["sector"] for c in companies}

    if not AR_FILE.exists():
        print(f"[ERROR] {AR_FILE} not found. Run abnormal_returns.py first.")
        return

    df = pd.read_csv(AR_FILE)
    original_count = len(df)
    print(f"Loaded {original_count} rows from {AR_FILE.name}")

    # Report before
    print("\nBefore fix — sector distribution:")
    print(df.groupby(["ticker", "sector"]).size().to_string())

    # Apply canonical sector
    df["sector"] = df["ticker"].map(sector_map)

    unmapped = df[df["sector"].isna()]["ticker"].unique()
    if len(unmapped):
        print(f"\n[WARNING] No sector found for tickers: {unmapped.tolist()}")
        print("  These rows will have NaN sector. Add them to companies.json if needed.")

    # Report after
    print("\nAfter fix — sector distribution:")
    print(df.groupby(["ticker", "sector"]).size().to_string())

    # Fix any annotation label encoding issues in pred_sentiment
    # (short-form labels like 'p', 'n', 'u' sometimes appear from old annotation exports)
    label_map = {"p": "positive", "n": "negative", "u": "neutral"}
    if "pred_sentiment" in df.columns:
        df["pred_sentiment"] = df["pred_sentiment"].replace(label_map)

    df.to_csv(AR_FILE, index=False)
    print(f"\n[OK] Saved fixed file: {AR_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    main()
