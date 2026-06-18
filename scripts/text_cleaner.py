"""
NGX-FND Text Cleaner & Preprocessor
=====================================
Cleans and normalizes extracted PDF text for LLM evaluation.
Handles Nigerian financial document quirks: OCR artifacts, 
page headers/footers, table noise, and formatting inconsistencies.

Usage:
  python text_cleaner.py --input ../data/extracted_text --output ../data/cleaned_text
  python text_cleaner.py --ticker MTNN
"""

import os
import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
TEXT_DIR     = BASE_DIR / "data" / "extracted_text"
CLEANED_DIR  = BASE_DIR / "data" / "cleaned_text"
METADATA_DIR = BASE_DIR / "data" / "metadata"
LOG_DIR      = BASE_DIR / "logs"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"cleaner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Cleaning Rules ─────────────────────────────────────────────────────────────

# Page header/footer patterns common in Nigerian annual reports
HEADER_FOOTER_PATTERNS = [
    r"^\s*\d+\s*$",                              # lone page numbers
    r"^\s*page\s+\d+\s+of\s+\d+\s*$",           # "Page 3 of 120"
    r"^\s*annual report\s+20\d{2}\s*$",          # "Annual Report 2023"
    r"^\s*[A-Z\s]{3,50}\s+plc\s*$",             # "ZENITH BANK PLC"
    r"^\s*www\.[a-z\-\.]+\.(com|ng|net)\s*$",   # website URLs
    r"^\s*\|\s*\d+\s*\|\s*$",                   # table page markers
    r"confidential.*?only\s*$",
    r"^\s*for the (year|period) ended.*?$",
    r"^.*?(?:/|\|)\s*(?:annual report|business review|strategic report|financial statements).*$", # Airtel/others footer
    r"^\s*(?:strategic report|business review|annual report)\s*$", # standalone footer fragments
]

# Boilerplate legal/regulatory phrases to strip
BOILERPLATE_PHRASES = [
    r"this annual report (is|has been) prepared",
    r"forward.looking statements.*?actual results",
    r"registered (office|address):\s*[^\n]{0,100}",
    r"rc\s*no\.?\s*\d+",                         # RC numbers
    r"frc/\d{4}/\w+/\d+",                        # FRC registration codes
    r"incorporated in nigeria",
    r"securities and exchange commission",
    r"nigerian exchange group",
    r"this report (was|is) printed on",
]

# OCR artifact patterns
OCR_ARTIFACTS = [
    r"[^\x00-\x7F\u00C0-\u024F\u2018\u2019\u201C\u201D\u2013\u2014\u20A6]{3,}",  # non-latin runs
    r"(\w)\1{4,}",          # repeated chars: "aaaaaa"
    r"[|]{2,}",             # multiple pipes from table borders
    r"_{3,}",               # long underscores
    r"\.{4,}",              # leader dots from TOC
    r"\s{3,}",              # excessive whitespace (replaced with double space)
]

# Normalize Nigerian currency symbols
CURRENCY_MAP = {
    "₦": "NGN ",
    "N'": "NGN ",
    "N ": "NGN ",
}

# Section quality thresholds
MIN_SECTION_CHARS = 200    # discard sections shorter than this
MAX_SECTION_CHARS = 40000  # truncate sections longer than this


# ── Core Cleaning Functions ────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode, fix smart quotes, standardize dashes."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # smart single quotes
    text = text.replace("\u201C", '"').replace("\u201D", '"')   # smart double quotes
    text = text.replace("\u2013", "-").replace("\u2014", " - ") # en/em dashes
    text = text.replace("\u00A0", " ")                          # non-breaking space
    text = text.replace("\u20A6", "NGN")                        # Naira sign
    return text


def remove_header_footers(text: str) -> str:
    """Remove page headers and footers line by line."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        is_noise = False
        for pattern in HEADER_FOOTER_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_noise = True
                break
        if not is_noise:
            cleaned.append(line)
    return "\n".join(cleaned)


def remove_boilerplate(text: str) -> str:
    """Strip common boilerplate legal/regulatory sentences."""
    for pattern in BOILERPLATE_PHRASES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def fix_ocr_artifacts(text: str) -> str:
    """Fix common OCR errors in scanned Nigerian documents."""
    # Fix hyphenation across lines: "strat-\negy" → "strategy"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse excessive whitespace
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove pipe artifacts from tables
    text = re.sub(r"\|{2,}", " ", text)
    # Remove TOC leader dots
    text = re.sub(r"\.{4,}", " ", text)
    # Remove long underscores (form fields, dividers)
    text = re.sub(r"_{3,}", " ", text)
    # Strip Private Use Area characters (PDF font icons like   )
    text = re.sub(r'[\uE000-\uF8FF]', ' ', text)
    return text


def fix_number_formatting(text: str) -> str:
    """Normalize number formats common in Nigerian filings."""
    # "N'000" or "₦'000" → "(NGN thousands)"
    text = re.sub(r"[N₦]'000", "(NGN thousands)", text, flags=re.IGNORECASE)
    text = re.sub(r"[N₦]'(million|billion)", r"(NGN \1)", text, flags=re.IGNORECASE)
    # Standardize "Naira" references
    text = re.sub(r"\bnaira\b", "NGN", text, flags=re.IGNORECASE)
    return text


def segment_paragraphs(text: str) -> str:
    """Ensure clean paragraph breaks for LLM input."""
    # Split by blank lines to get natural paragraphs
    blocks = re.split(r'\n\s*\n', text)
    cleaned_blocks = []
    
    for block in blocks:
        # Merge wrapped lines within the block
        merged_block = re.sub(r'(?<!\n)\n(?!\n)', ' ', block).strip()
        merged_block = re.sub(r'\s+', ' ', merged_block)
        if not merged_block:
            continue
            
        # If the block starts with a lowercase letter, it's a continuation of the previous block
        # (happens often with PDF page breaks or aggressive blank lines)
        if cleaned_blocks and merged_block[0].islower():
            cleaned_blocks[-1] += " " + merged_block
        else:
            cleaned_blocks.append(merged_block)
            
    return "\n\n".join(cleaned_blocks)


def clean_section(text: str, section_name: str = "") -> dict:
    """
    Full cleaning pipeline for a single narrative section.
    Returns cleaned text and quality metadata.
    """
    if not text or not text.strip():
        return {"text": "", "quality": "empty", "char_count": 0}

    original_len = len(text)

    text = normalize_unicode(text)
    text = remove_header_footers(text)
    text = remove_boilerplate(text)
    text = fix_ocr_artifacts(text)
    text = fix_number_formatting(text)
    text = segment_paragraphs(text)
    text = text.strip()

    char_count = len(text)

    # Quality assessment
    if char_count < MIN_SECTION_CHARS:
        quality = "too_short"
    elif char_count > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS] + "\n[TRUNCATED]"
        quality = "truncated"
    else:
        quality = assess_quality(text)

    return {
        "text":          text,
        "quality":       quality,
        "char_count":    char_count,
        "original_len":  original_len,
        "compression":   round((1 - char_count / original_len) * 100, 1) if original_len else 0,
    }


def assess_quality(text: str) -> str:
    """
    Assess text quality for annotation suitability.
    Returns: 'good' | 'acceptable' | 'poor'
    """
    if not text:
        return "empty"

    words = text.split()
    word_count = len(words)

    # Too few meaningful words
    if word_count < 50:
        return "poor"

    # Check for excessive OCR noise (non-alpha chars)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.6:
        return "poor"

    # Check sentence structure (should have full stops)
    sentence_count = len(re.findall(r"[.!?]", text))
    if sentence_count < 3:
        return "poor"

    # Check for forward-looking language (key for our task)
    forward_keywords = [
        "outlook", "expect", "anticipate", "forecast", "project",
        "guidance", "target", "plan", "strategy", "objective",
        "will", "intend", "believe", "2024", "2025", "going forward",
        "next year", "upcoming", "future"
    ]
    has_forward = any(kw in text.lower() for kw in forward_keywords)

    # Check for sentiment language
    sentiment_keywords = [
        "growth", "decline", "increase", "decrease", "improved",
        "challenges", "opportunity", "risk", "strong", "weak",
        "performance", "revenue", "profit", "loss", "recovery"
    ]
    has_sentiment = any(kw in text.lower() for kw in sentiment_keywords)

    if has_forward and has_sentiment:
        return "good"
    elif has_forward or has_sentiment:
        return "acceptable"
    else:
        return "poor"


# ── Passage Extraction for Annotation ─────────────────────────────────────────

def extract_annotation_passages(cleaned_sections: dict, doc_meta: dict) -> list[dict]:
    """
    Extract bite-sized passages (200–800 words) for human annotation.
    Prioritizes sections with forward-looking and sentiment language.
    """
    passages = []
    priority_sections = ["outlook", "chairman_statement", "ceo_review", "operating_review"]

    for section_name in priority_sections:
        section = cleaned_sections.get(section_name, {})
        text = section.get("text", "")
        quality = section.get("quality", "")

        if quality not in ("good", "acceptable") or not text:
            continue

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 100]

        for i, para in enumerate(paragraphs):
            word_count = len(para.split())
            if word_count < 30:
                continue

            # Combine short adjacent paragraphs
            if word_count < 80 and i + 1 < len(paragraphs):
                para = para + "\n\n" + paragraphs[i + 1]
                word_count = len(para.split())

            if word_count > 400:
                # Split long paragraphs at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                chunk, chunks = [], []
                for sent in sentences:
                    chunk.append(sent)
                    if len(" ".join(chunk).split()) >= 150:
                        chunks.append(" ".join(chunk))
                        chunk = []
                if chunk:
                    chunks.append(" ".join(chunk))
                sub_passages = chunks
            else:
                sub_passages = [para]

            for j, passage_text in enumerate(sub_passages):
                # Check for table noise: if > 25% of tokens contain digits, drop it
                words = passage_text.split()
                num_count = sum(1 for w in words if re.search(r'\d', w))
                if len(words) < 30 or (num_count / len(words)) > 0.25:
                    continue

                passages.append({
                    "passage_id":   f"{doc_meta['ticker']}_{doc_meta.get('year','UNK')}_{section_name}_{i}_{j}",
                    "ticker":       doc_meta["ticker"],
                    "company":      doc_meta.get("company_name", ""),
                    "sector":       doc_meta.get("sector", ""),
                    "year":         doc_meta.get("year", ""),
                    "doc_type":     doc_meta.get("doc_type", ""),
                    "section":      section_name,
                    "text":         passage_text,
                    "word_count":   len(passage_text.split()),
                    "quality":      assess_quality(passage_text),
                    # Annotation fields (to be filled)
                    "sentiment_label":     None,  # positive / negative / neutral
                    "sentiment_intensity": None,  # 1-3 (mild/moderate/strong)
                    "has_guidance":        None,  # True / False
                    "guidance_type":       None,  # positive / negative / neutral / conditional
                    "guidance_span":       None,  # text span containing guidance
                    "annotator":           None,
                    "annotation_notes":    None,
                    "annotation_status":   "pending",
                })

    return passages


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def process_file(json_path: Path) -> dict | None:
    """Clean a single extracted JSON file and return cleaned output."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error(f"Failed to load {json_path}: {e}")
        return None

    sections_raw = data.get("sections", {})
    cleaned_sections = {}

    for section_name in ["full_text", "chairman_statement", "ceo_review",
                          "operating_review", "outlook"]:
        raw_text = sections_raw.get(section_name, "")
        cleaned_sections[section_name] = clean_section(raw_text, section_name)

    doc_meta = {k: v for k, v in data.items() if k != "sections"}

    passages = extract_annotation_passages(cleaned_sections, doc_meta)

    return {
        "meta":             doc_meta,
        "cleaned_sections": cleaned_sections,
        "passages":         passages,
        "passage_count":    len(passages),
        "good_passages":    sum(1 for p in passages if p["quality"] == "good"),
        "processed_at":     datetime.now().isoformat(),
    }


def run_cleaner(ticker_filter: str | None = None):
    """Run cleaning pipeline over all extracted text files."""
    log.info("=== CLEANING PHASE ===")

    source_files = list(TEXT_DIR.rglob("*.json"))
    if ticker_filter:
        source_files = [f for f in source_files if f.parent.name == ticker_filter.upper()]

    log.info(f"Processing {len(source_files)} extracted files")

    all_passages = []
    summary_rows = []

    for json_path in source_files:
        log.info(f"  Cleaning: {json_path.name}")
        result = process_file(json_path)
        if not result:
            continue

        ticker = result["meta"].get("ticker", "UNKNOWN")
        out_dir = CLEANED_DIR / ticker
        out_dir.mkdir(exist_ok=True)

        out_path = out_dir / (json_path.stem + "_cleaned.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_passages.extend(result["passages"])
        summary_rows.append({
            "ticker":         ticker,
            "file":           json_path.name,
            "passage_count":  result["passage_count"],
            "good_passages":  result["good_passages"],
            "year":           result["meta"].get("year", ""),
            "sector":         result["meta"].get("sector", ""),
            "doc_type":       result["meta"].get("doc_type", ""),
        })

    # Save master annotation queue
    if all_passages:
        passages_df = pd.DataFrame(all_passages)
        out_csv = METADATA_DIR / "annotation_queue.csv"
        
        # Preserve existing annotations if they exist
        if out_csv.exists():
            try:
                old_df = pd.read_csv(out_csv, dtype=str)
                if "annotation_status" in old_df.columns:
                    annotated = old_df[old_df["annotation_status"] != "pending"].copy()
                    if not annotated.empty:
                        # Merge by passage_id
                        cols_to_restore = ["sentiment_label", "sentiment_intensity", "has_guidance", 
                                           "guidance_type", "guidance_span", "annotator", 
                                           "annotation_notes", "annotation_status", "annotated_at"]
                        # Remove duplicates to avoid reindexing errors
                        annotated = annotated[~annotated["passage_id"].duplicated(keep='last')]
                        
                        passages_df.set_index("passage_id", inplace=True)
                        
                        # Also handle duplicates in the new passages DataFrame
                        passages_df = passages_df[~passages_df.index.duplicated(keep='first')]
                        
                        annotated.set_index("passage_id", inplace=True)
                        
                        for col in cols_to_restore:
                            if col in annotated.columns and col in passages_df.columns:
                                passages_df[col].update(annotated[col])
                                
                        passages_df.reset_index(inplace=True)
                        log.info(f"Restored annotations for {len(annotated)} passages")
            except Exception as e:
                log.error(f"Failed to restore annotations: {e}")

        passages_df.to_csv(out_csv, index=False, escapechar='\\')
        log.info(f"Annotation queue saved: {len(passages_df)} passages -> {out_csv}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(METADATA_DIR / "cleaning_summary.csv", index=False, escapechar='\\')

        print("\n" + "="*50)
        print("Cleaning Summary")
        print("="*50)
        print(f"  Files processed     : {len(summary_rows)}")
        print(f"  Total passages      : {len(all_passages)}")
        print(f"  Good quality        : {sum(p['quality']=='good' for p in all_passages)}")
        print(f"  Acceptable quality  : {sum(p['quality']=='acceptable' for p in all_passages)}")
        print(f"  Poor quality        : {sum(p['quality']=='poor' for p in all_passages)}")
        print("\nBy sector:")
        print(summary_df.groupby("sector")[["passage_count","good_passages"]].sum().to_string())
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Text Cleaner")
    parser.add_argument("--ticker", help="Process single ticker only")
    args = parser.parse_args()
    run_cleaner(ticker_filter=args.ticker)


if __name__ == "__main__":
    main()
