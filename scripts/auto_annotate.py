import os
import time
import json
import logging
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# ── Paths & Setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
QUEUE_FILE = METADATA_DIR / "annotation_queue.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    log.error("GEMINI_API_KEY not found in .env file or environment variables.")
    log.error("Please add it to a .env file in the root directory before running.")
    exit(1)

genai.configure(api_key=API_KEY)

# ── LLM Configuration ──────────────────────────────────────────────────────────
# We use Gemini 1.5 Flash for fast, cheap, and accurate JSON extraction
model = genai.GenerativeModel('gemini-1.5-flash')

PROMPT_TEMPLATE = """
You are an expert financial analyst extracting sentiment and forward guidance from African corporate disclosures.
Analyze the following passage from a company's financial report and extract the structured data requested.

Rules:
1. sentiment_label: Must be exactly "positive", "negative", or "neutral".
2. sentiment_intensity: Must be exactly "mild", "moderate", or "strong".
3. has_guidance: true if there are forward-looking statements or expectations about the future, false otherwise.
4. guidance_type: Must be "positive", "negative", "neutral", "conditional", or "" if no guidance.
5. guidance_span: The EXACT sentence from the passage containing the forward guidance. If no guidance, leave empty.

Passage:
{passage}
"""

def analyze_passage(text: str) -> dict:
    """Call Gemini to analyze a single passage."""
    try:
        response = model.generate_content(
            PROMPT_TEMPLATE.format(passage=text),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "sentiment_label": {"type": "STRING", "enum": ["positive", "negative", "neutral"]},
                        "sentiment_intensity": {"type": "STRING", "enum": ["mild", "moderate", "strong"]},
                        "has_guidance": {"type": "BOOLEAN"},
                        "guidance_type": {"type": "STRING", "enum": ["positive", "negative", "neutral", "conditional", ""]},
                        "guidance_span": {"type": "STRING"}
                    },
                    "required": ["sentiment_label", "sentiment_intensity", "has_guidance", "guidance_type", "guidance_span"]
                }
            )
        )
        return json.loads(response.text)
    except Exception as e:
        log.error(f"LLM API Error: {e}")
        return None

# ── Main Script ────────────────────────────────────────────────────────────────
def main(batch_size: int = 50):
    log.info(f"Loading queue from {QUEUE_FILE}")
    if not QUEUE_FILE.exists():
        log.error("Queue file not found!")
        return

    df = pd.read_csv(QUEUE_FILE, dtype=str)
    df = df.fillna("")

    # Only process annotatable passages that are currently pending
    annotatable_mask = df["quality"].isin(["good", "acceptable"])
    pending_mask = (df["annotation_status"] == "pending") | (df["annotation_status"] == "")
    
    candidates = df[annotatable_mask & pending_mask]
    
    if candidates.empty:
        log.info("No pending passages found for auto-annotation!")
        return

    # Limit to batch size to avoid long runs or API limits
    batch = candidates.head(batch_size)
    log.info(f"Starting AI annotation batch of {len(batch)} passages...")

    success_count = 0
    for idx, row in batch.iterrows():
        passage_id = row.get("passage_id", f"Row {idx}")
        text = row.get("text", "")
        
        log.info(f"Processing [{idx}]: {passage_id}")
        
        # We only annotate if the text is long enough to be meaningful
        if len(text.strip()) < 50:
            log.warning("Passage too short, skipping.")
            continue
            
        result = analyze_passage(text)
        if result:
            df.at[idx, "sentiment_label"] = result.get("sentiment_label", "neutral")
            df.at[idx, "sentiment_intensity"] = result.get("sentiment_intensity", "moderate")
            df.at[idx, "has_guidance"] = str(result.get("has_guidance", False))
            df.at[idx, "guidance_type"] = result.get("guidance_type", "")
            df.at[idx, "guidance_span"] = result.get("guidance_span", "")
            
            df.at[idx, "annotator"] = "Gemini-1.5-Flash"
            df.at[idx, "annotation_status"] = "ai_annotated"
            
            success_count += 1
        else:
            log.warning(f"Failed to get LLM response for {passage_id}")
            
        # Avoid hitting rate limits (adjust based on tier)
        time.sleep(1.0)

    # Save results
    if success_count > 0:
        df.to_csv(QUEUE_FILE, index=False)
        log.info(f"Successfully annotated {success_count} passages and updated CSV.")
    else:
        log.info("No passages were successfully annotated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-Annotate NGX Passages using Gemini")
    parser.add_argument("--batch", type=int, default=50, help="Number of passages to process")
    args = parser.parse_args()
    
    main(batch_size=args.batch)
