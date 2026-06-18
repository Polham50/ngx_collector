"""
NGX-FND Annotation Tool
========================
CLI tool for human annotation of financial narrative passages.
Annotators label each passage for:
  1. Sentiment polarity (positive / negative / neutral)
  2. Sentiment intensity (1=mild, 2=moderate, 3=strong)
  3. Forward guidance presence (yes / no)
  4. Guidance type (positive / negative / neutral / conditional)
  5. Guidance span (the key sentence)

Usage:
  python annotator.py --annotator "Timothy"              # Start/resume annotation session
  python annotator.py --annotator "Timothy" --sector Banking  # Annotate specific sector
  python annotator.py --stats                            # View annotation progress
  python annotator.py --export                           # Export gold standard CSV
"""

import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
QUEUE_FILE   = METADATA_DIR / "annotation_queue.csv"
GOLD_FILE    = METADATA_DIR / "gold_standard.csv"

# ── Label Definitions ──────────────────────────────────────────────────────────
SENTIMENT_LABELS = {
    "p": "positive",
    "n": "negative",
    "u": "neutral",
}
INTENSITY_LABELS = {
    "1": "mild",
    "2": "moderate",
    "3": "strong",
}
GUIDANCE_LABELS = {
    "y": True,
    "n": False,
}
GUIDANCE_TYPE_LABELS = {
    "p": "positive",
    "n": "negative",
    "u": "neutral",
    "c": "conditional",
    "s": "skip",
}

ANNOTATION_GUIDE = """
================================================================
           NGX-FND ANNOTATION GUIDELINES (Quick Ref)         
================================================================
 SENTIMENT — What is the OVERALL tone of this passage?        
   positive  -> growth, improvement, optimism, strong results  
   negative  -> decline, loss, challenges, downturn            
   neutral   -> factual, balanced, no clear directional tone   
                                                              
 INTENSITY — How strong is the sentiment?                     
   1 = mild      (slight lean, hedged language)               
   2 = moderate  (clear but measured)                         
   3 = strong    (emphatic, unambiguous)                      
                                                              
 GUIDANCE — Does the passage contain FORWARD-LOOKING          
            statements about future performance?              
   yes -> explicit forecast, target, plan, expectation        
   no  -> purely historical/descriptive                        
                                                              
 GUIDANCE TYPE (if yes):                                      
   positive    -> expects improvement / growth                 
   negative    -> expects decline / challenges ahead           
   neutral     -> no directional signal                        
   conditional -> depends on external factors (FX, policy...) 
================================================================
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_queue() -> pd.DataFrame:
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(f"Annotation queue not found: {QUEUE_FILE}\n"
                                "Run text_cleaner.py first.")
    return pd.read_csv(QUEUE_FILE, dtype=str)


def save_queue(df: pd.DataFrame):
    df.to_csv(QUEUE_FILE, index=False)


def clear_screen():
    print("\033[2J\033[H", end="")


def wrap_text(text: str, width: int = 80) -> str:
    paragraphs = text.split("\n\n")
    wrapped = []
    for para in paragraphs:
        wrapped.append(textwrap.fill(para.strip(), width=width))
    return "\n\n".join(wrapped)


def prompt(question: str, valid_keys: dict, allow_skip: bool = True) -> str | None:
    """Prompt user for input with validation."""
    options = " / ".join(f"[{k}]{v}" for k, v in valid_keys.items())
    if allow_skip:
        options += " / [s]skip / [q]quit / [?]help"
    while True:
        raw = input(f"\n{question}\n  {options}: ").strip().lower()
        if raw == "q":
            return "QUIT"
        if raw == "s" and allow_skip:
            return "SKIP"
        if raw == "?" and allow_skip:
            print(ANNOTATION_GUIDE)
            continue
        if raw in valid_keys:
            return valid_keys[raw]
        print(f"  Invalid input. Enter one of: {list(valid_keys.keys())}")


# ── Annotation Session ─────────────────────────────────────────────────────────

def run_annotation_session(annotator: str, sector_filter: str | None = None,
                            batch_size: int = 20):
    """Run an interactive annotation session."""
    df = load_queue()

    # Filter to pending passages for this annotator
    pending = df[
        (df["annotation_status"] == "pending") |
        (df["annotator"].isna()) |
        (df["annotator"] == "")
    ].copy()

    if sector_filter:
        pending = pending[pending["sector"].str.lower() == sector_filter.lower()]

    # Prioritize good quality, outlook and chairman sections first
    priority_order = {"good": 0, "acceptable": 1, "poor": 2}
    section_order  = {"outlook": 0, "chairman_statement": 1, "ceo_review": 2,
                      "operating_review": 3}
    pending["_q_order"] = pending["quality"].map(priority_order).fillna(9)
    pending["_s_order"] = pending["section"].map(section_order).fillna(9)
    pending = pending.sort_values(["_q_order", "_s_order"]).head(batch_size)

    if pending.empty:
        print(f"\nNo pending passages found{' for sector: ' + sector_filter if sector_filter else ''}.")
        return

    print(ANNOTATION_GUIDE)
    print(f"\n[!] Starting annotation session for: {annotator}")
    print(f"   Passages in this batch: {len(pending)}")
    print(f"   Press [s] to skip, [q] to quit, [?] for help\n")
    input("   Press ENTER to begin...")

    annotated_count = 0

    for idx, row in pending.iterrows():
        clear_screen()
        progress = f"[{annotated_count + 1}/{len(pending)}]"

        print(f"\n{'='*70}")
        print(f"{progress}  {row.get('ticker','')} | {row.get('company','')} | "
              f"{row.get('year','')} | {row.get('doc_type','')} | "
              f"Section: {row.get('section','')}")
        print(f"{'='*70}\n")
        print(wrap_text(str(row.get("text", ""))))
        print(f"\n{'─'*70}")
        print(f"Word count: {row.get('word_count','')} | Quality: {row.get('quality','')}")

        # Q1: Sentiment
        sentiment = prompt("1. Overall SENTIMENT of this passage?",
                           {"p": "positive", "n": "negative", "u": "neutral"})
        if sentiment == "QUIT":
            break
        if sentiment == "SKIP":
            df.at[idx, "annotation_status"] = "skipped"
            df.at[idx, "annotator"] = annotator
            save_queue(df)
            continue

        # Q2: Intensity
        intensity = prompt("2. INTENSITY of sentiment?",
                           {"1": "mild", "2": "moderate", "3": "strong"})
        if intensity == "QUIT":
            break
        if intensity == "SKIP":
            intensity = None

        # Q3: Forward guidance
        has_guidance_str = prompt("3. Does this passage contain FORWARD GUIDANCE?",
                                  {"y": "yes", "n": "no"})
        if has_guidance_str == "QUIT":
            break
        has_guidance = has_guidance_str == "yes"

        guidance_type = None
        guidance_span = None

        if has_guidance:
            # Q4: Guidance type
            guidance_type = prompt("4. GUIDANCE TYPE?",
                                   {"p": "positive", "n": "negative",
                                    "u": "neutral",  "c": "conditional"})
            if guidance_type == "QUIT":
                break

            # Q5: Key guidance sentence
            print("\n5. Paste the KEY SENTENCE containing guidance (or press ENTER to skip):")
            guidance_span = input("   > ").strip() or None

        # Optional notes
        print("\n6. Any annotation notes? (press ENTER to skip)")
        notes = input("   > ").strip() or None

        # Save annotation
        df.at[idx, "sentiment_label"]     = sentiment
        df.at[idx, "sentiment_intensity"] = intensity
        df.at[idx, "has_guidance"]        = has_guidance
        df.at[idx, "guidance_type"]       = guidance_type
        df.at[idx, "guidance_span"]       = guidance_span
        df.at[idx, "annotation_notes"]    = notes
        df.at[idx, "annotator"]           = annotator
        df.at[idx, "annotation_status"]   = "done"
        df.at[idx, "annotated_at"]        = datetime.now().isoformat()

        save_queue(df)
        annotated_count += 1
        print(f"\n  [OK] Saved! ({annotated_count} annotated this session)")

    print(f"\n{'='*70}")
    print(f"Session complete. {annotated_count} passages annotated by {annotator}.")
    print_annotation_stats()


# ── Statistics ────────────────────────────────────────────────────────────────

def print_annotation_stats():
    df = load_queue()
    total     = len(df)
    done      = (df["annotation_status"] == "done").sum()
    pending   = (df["annotation_status"] == "pending").sum()
    skipped   = (df["annotation_status"] == "skipped").sum()

    print(f"\n{'='*60}")
    print("Annotation Progress")
    print(f"{'='*60}")
    print(f"  Total passages   : {total}")
    print(f"  [OK] Annotated      : {done} ({round(done/total*100,1)}%)")
    print(f"  [...] Pending        : {pending}")
    print(f"  [>] Skipped        : {skipped}")

    done_df = df[df["annotation_status"] == "done"]
    if not done_df.empty:
        print("\n  Sentiment distribution:")
        if "sentiment_label" in done_df:
            print(done_df["sentiment_label"].value_counts().to_string())

        print("\n  Guidance rate:")
        if "has_guidance" in done_df:
            print(done_df["has_guidance"].value_counts().to_string())

        if "annotator" in done_df:
            print("\n  By annotator:")
            print(done_df["annotator"].value_counts().to_string())

        if "sector" in done_df:
            print("\n  By sector:")
            print(done_df["sector"].value_counts().to_string())

    print(f"{'='*60}\n")


def export_gold_standard():
    """Export completed annotations as gold standard CSV."""
    df = load_queue()
    gold = df[df["annotation_status"] == "done"].copy()

    if gold.empty:
        print("No completed annotations to export yet.")
        return

    # Drop internal columns
    export_cols = [
        "passage_id", "ticker", "company", "sector", "year",
        "doc_type", "section", "text", "word_count",
        "sentiment_label", "sentiment_intensity",
        "has_guidance", "guidance_type", "guidance_span",
        "annotator", "annotation_notes"
    ]
    export_cols = [c for c in export_cols if c in gold.columns]
    gold[export_cols].to_csv(GOLD_FILE, index=False)
    print(f"\n[OK] Gold standard exported: {len(gold)} passages → {GOLD_FILE}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NGX-FND Annotation Tool")
    parser.add_argument("--annotator", help="Your name (for tracking)")
    parser.add_argument("--sector",    help="Filter to a specific sector")
    parser.add_argument("--batch",     type=int, default=20,
                        help="Number of passages per session (default: 20)")
    parser.add_argument("--stats",     action="store_true", help="Show annotation stats")
    parser.add_argument("--export",    action="store_true", help="Export gold standard")
    args = parser.parse_args()

    if args.stats:
        print_annotation_stats()
        return

    if args.export:
        export_gold_standard()
        return

    if not args.annotator:
        parser.print_help()
        return

    run_annotation_session(
        annotator=args.annotator,
        sector_filter=args.sector,
        batch_size=args.batch
    )


if __name__ == "__main__":
    main()
