"""
Fix legacy short-form annotation labels in annotation_queue.csv.

Converts:
  sentiment_label: 'p' -> 'positive', 'n' -> 'negative', 'u' -> 'neutral'
  guidance_type:   'p' -> 'positive', 'n' -> 'negative', 'u' -> 'neutral', 'c' -> 'conditional'
  sentiment_intensity: '1' -> 'mild', '2' -> 'moderate', '3' -> 'strong'

Usage:
    python fix_annotation_labels.py
"""
from pathlib import Path
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
QUEUE_FILE = BASE_DIR / "data" / "metadata" / "annotation_queue.csv"


def main():
    if not QUEUE_FILE.exists():
        print(f"[ERROR] {QUEUE_FILE} not found.")
        return

    df = pd.read_csv(QUEUE_FILE, dtype=str)
    print(f"Loaded {len(df)} rows from {QUEUE_FILE.name}")

    changes = 0

    # Fix sentiment_label
    sentiment_map = {"p": "positive", "n": "negative", "u": "neutral"}
    if "sentiment_label" in df.columns:
        before = df["sentiment_label"].value_counts().to_dict()
        df["sentiment_label"] = df["sentiment_label"].replace(sentiment_map)
        after = df["sentiment_label"].value_counts().to_dict()
        if before != after:
            print(f"\n  sentiment_label before: {before}")
            print(f"  sentiment_label after:  {after}")
            changes += 1
        else:
            print("  sentiment_label: no short-form labels found")

    # Fix guidance_type
    guid_map = {"p": "positive", "n": "negative", "u": "neutral", "c": "conditional"}
    if "guidance_type" in df.columns:
        before = df["guidance_type"].value_counts().to_dict()
        df["guidance_type"] = df["guidance_type"].replace(guid_map)
        after = df["guidance_type"].value_counts().to_dict()
        if before != after:
            print(f"\n  guidance_type before: {before}")
            print(f"  guidance_type after:  {after}")
            changes += 1
        else:
            print("  guidance_type: no short-form labels found")

    # Fix sentiment_intensity
    intensity_map = {"1": "mild", "2": "moderate", "3": "strong"}
    if "sentiment_intensity" in df.columns:
        before = df["sentiment_intensity"].value_counts().to_dict()
        df["sentiment_intensity"] = df["sentiment_intensity"].replace(intensity_map)
        after = df["sentiment_intensity"].value_counts().to_dict()
        if before != after:
            print(f"\n  sentiment_intensity before: {before}")
            print(f"  sentiment_intensity after:  {after}")
            changes += 1
        else:
            print("  sentiment_intensity: no numeric labels found")

    if changes > 0:
        df.to_csv(QUEUE_FILE, index=False)
        print(f"\n[OK] Saved fixed file ({changes} column(s) updated)")
    else:
        print("\n[OK] No legacy labels found - file is already correct")


if __name__ == "__main__":
    main()
