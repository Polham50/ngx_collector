"""
NGX-FND LLM Evaluator
======================
Evaluates GPT-4o, Claude 3.5/3.7, and Gemini 1.5 Pro on Nigerian Exchange
financial narrative passages for:
  - Task A: Sentiment classification (positive / negative / neutral)
  - Task B: Forward guidance detection
  - Task C: Combined (single-pass)

Supports zero-shot, 3-shot, and 5-shot prompting.
Outputs per-model predictions + aggregate results for the paper.

Usage:
  # Run all models, combined task, 5-shot
  python llm_evaluator.py --task combined --shots 5

  # Run specific model only
  python llm_evaluator.py --task sentiment --shots 3 --model claude

  # Run on a small sample first (test mode)
  python llm_evaluator.py --task combined --shots 5 --sample 20

  # Resume interrupted run
  python llm_evaluator.py --task combined --shots 5 --resume

  # Compare against FinBERT baseline
  python llm_evaluator.py --task sentiment --shots 0 --baseline

API keys are read from environment variables:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY
"""

import os
import re
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
RESULTS_DIR  = BASE_DIR / "data" / "llm_results"
LOG_DIR      = BASE_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Import prompts ─────────────────────────────────────────────────────────────
from prompts import SYSTEM_PROMPT, PROMPT_REGISTRY, get_prompt

# ── Model Config ───────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "gpt4o": {
        "provider":   "openai",
        "model_id":   "gpt-4o",
        "label":      "GPT-4o",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "claude": {
        "provider":   "anthropic",
        "model_id":   "claude-sonnet-4-20250514",
        "label":      "Claude Sonnet 4",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "gemini": {
        "provider":   "google",
        "model_id":   "gemini-2.5-pro",
        "label":      "Gemini 2.5 Pro",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "llama3": {
        "provider":   "groq",
        "model_id":   "llama-3.3-70b-versatile",
        "label":      "Llama 3.3 70B (Groq)",
        "max_tokens": 512,
        "temperature": 0.0,
    },
}

# Retry config
MAX_RETRIES   = 3
RETRY_DELAY   = 5   # seconds
RATE_LIMIT_DELAY = {
    "openai":    1.2,
    "anthropic": 1.0,
    "google":    1.5,
    "groq":      2.0,
}


# ── API Callers ────────────────────────────────────────────────────────────────

def call_openai(prompt: str, config: dict) -> str:
    """Call OpenAI GPT-4o."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=config["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, config: dict) -> str:
    """Call Claude via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=config["model_id"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_google(prompt: str, config: dict) -> str:
    """Call Gemini via Google Generative AI."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        model_name=config["model_id"],
        generation_config=genai.GenerationConfig(
            max_output_tokens=config["max_tokens"],
            temperature=config["temperature"],
        ),
    )
    response = model.generate_content(SYSTEM_PROMPT + "\n\n" + prompt)
    return response.text


def call_groq(prompt: str, config: dict) -> str:
    """Call Groq using the OpenAI client."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model=config["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


CALLERS = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
    "groq":      call_groq,
}


def call_model(model_key: str, prompt: str) -> dict:
    """
    Call a model with retries. Returns a result dict with:
    raw_response, parsed_json, latency_ms, error, retry_count.
    """
    config   = MODEL_CONFIG[model_key]
    provider = config["provider"]
    caller   = CALLERS[provider]

    result = {
        "model":        model_key,
        "model_label":  config["label"],
        "raw_response": None,
        "parsed":       None,
        "latency_ms":   None,
        "error":        None,
        "retry_count":  0,
    }

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            raw = caller(prompt, config)
            result["latency_ms"]   = round((time.time() - t0) * 1000)
            result["raw_response"] = raw
            result["parsed"]       = parse_json_response(raw)
            result["retry_count"]  = attempt
            break
        except Exception as e:
            result["error"]       = str(e)
            result["retry_count"] = attempt + 1
            log.warning(f"  [{config['label']}] attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    time.sleep(RATE_LIMIT_DELAY.get(provider, 1.0))
    return result


def parse_json_response(raw: str) -> dict | None:
    """Parse JSON from model response, handling common formatting issues."""
    if not raw:
        return None
    # Strip markdown fences if present
    clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
    # Strip any leading/trailing text outside braces
    brace_start = clean.find("{")
    brace_end   = clean.rfind("}") + 1
    if brace_start != -1 and brace_end > brace_start:
        clean = clean[brace_start:brace_end]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        log.warning(f"  JSON parse failed. Raw: {raw[:200]}")
        return None


# ── FinBERT Baseline ───────────────────────────────────────────────────────────

def run_finbert_baseline(passages: list[str]) -> list[dict]:
    """
    Run FinBERT as a baseline sentiment model.
    Uses transformers pipeline — falls back gracefully if not available.
    """
    try:
        from transformers import pipeline
        log.info("Loading FinBERT pipeline...")
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        results = []
        for passage in tqdm(passages, desc="FinBERT"):
            try:
                out = finbert(passage[:2000])[0]
                label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
                results.append({
                    "model":       "finbert",
                    "model_label": "FinBERT",
                    "parsed": {
                        "sentiment": label_map.get(out["label"].lower(), "neutral"),
                        "intensity": "moderate",   # FinBERT doesn't give intensity
                        "finbert_score": round(out["score"], 4),
                    },
                    "error": None,
                })
            except Exception as e:
                results.append({"model": "finbert", "model_label": "FinBERT",
                                "parsed": None, "error": str(e)})
        return results
    except ImportError:
        log.warning("transformers not installed — skipping FinBERT baseline")
        return []


# ── Evaluation Runner ──────────────────────────────────────────────────────────

class EvaluationRun:
    """Manages a single evaluation run across models and passages."""

    def __init__(self, task: str, shots: int, models: list[str],
                 sample: int | None = None, resume: bool = False):
        self.task    = task
        self.shots   = shots
        self.models  = models
        self.sample  = sample
        self.resume  = resume
        self.prompt_key = self._resolve_prompt_key()
        self.run_id  = f"{task}_{shots}shot_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.out_dir = RESULTS_DIR / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.out_dir / "predictions.csv"

    def _resolve_prompt_key(self) -> str:
        key = f"{self.task}_{self.shots}shot"
        if key not in PROMPT_REGISTRY:
            # Fall back to closest available
            available = [k for k in PROMPT_REGISTRY if k.startswith(self.task)]
            if not available:
                raise ValueError(f"No prompts found for task '{self.task}'")
            # Pick lowest-shot available
            key = sorted(available, key=lambda x: PROMPT_REGISTRY[x]["shots"])[0]
            log.warning(f"Prompt '{self.task}_{self.shots}shot' not found, using '{key}'")
        return key

    def load_passages(self) -> pd.DataFrame:
        gold_path = METADATA_DIR / "gold_standard.csv"
        queue_path = METADATA_DIR / "annotation_queue.csv"

        if gold_path.exists():
            df = pd.read_csv(gold_path, dtype=str)
            log.info(f"Loaded gold standard: {len(df)} passages")
        elif queue_path.exists():
            df = pd.read_csv(queue_path, dtype=str)
            # Use all passages with good/acceptable quality even if not fully annotated
            if "quality" in df:
                df = df[df["quality"].isin(["good", "acceptable"])]
            log.info(f"Gold standard not found — using annotation queue: {len(df)} passages")
        else:
            raise FileNotFoundError(
                "No passages found. Run text_cleaner.py first to generate annotation_queue.csv"
            )

        if self.sample:
            df = df.sample(min(self.sample, len(df)), random_state=42)
            log.info(f"Sampled {len(df)} passages")

        return df.reset_index(drop=True)

    def get_already_evaluated(self) -> set[str]:
        """Return passage IDs already evaluated in a previous interrupted run."""
        if not self.resume or not self.predictions_path.exists():
            return set()
        existing = pd.read_csv(self.predictions_path, dtype=str)
        done = set(existing["passage_id"].unique())
        log.info(f"Resuming: {len(done)} passages already evaluated")
        return done

    def run(self, run_baseline: bool = False) -> pd.DataFrame:
        passages_df = self.load_passages()
        done_ids    = self.get_already_evaluated()

        all_rows = []
        if self.resume and self.predictions_path.exists():
            all_rows = pd.read_csv(self.predictions_path, dtype=str).to_dict("records")

        pending = passages_df[~passages_df["passage_id"].isin(done_ids)]
        log.info(f"Evaluating {len(pending)} passages × {len(self.models)} models")
        log.info(f"Prompt: {self.prompt_key}")

        # FinBERT baseline (sentiment tasks only)
        if run_baseline and self.task in ("sentiment", "combined"):
            log.info("Running FinBERT baseline...")
            fb_results = run_finbert_baseline(pending["text"].tolist())
            for i, (_, row) in enumerate(pending.iterrows()):
                if i < len(fb_results):
                    fb = fb_results[i]
                    all_rows.append(self._build_row(row, fb, self.prompt_key))

        # LLM evaluation
        for model_key in self.models:
            if model_key not in MODEL_CONFIG:
                log.warning(f"Unknown model: {model_key} — skipping")
                continue

            api_key_env = {
                "openai":    "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google":    "GOOGLE_API_KEY",
                "groq":      "GROQ_API_KEY",
            }[MODEL_CONFIG[model_key]["provider"]]

            if not os.environ.get(api_key_env):
                log.warning(f"[{model_key}] {api_key_env} not set — skipping")
                continue

            log.info(f"\n{'='*50}")
            log.info(f"Model: {MODEL_CONFIG[model_key]['label']}")
            log.info(f"{'='*50}")

            for _, row in tqdm(pending.iterrows(), total=len(pending),
                               desc=MODEL_CONFIG[model_key]["label"]):
                passage  = str(row.get("text", ""))
                prompt   = get_prompt(self.prompt_key, passage)
                result   = call_model(model_key, prompt)
                all_rows.append(self._build_row(row, result, self.prompt_key))

            # Checkpoint after each model
            pd.DataFrame(all_rows).to_csv(self.predictions_path, index=False)
            log.info(f"  Checkpoint saved: {self.predictions_path}")

        predictions_df = pd.DataFrame(all_rows)
        predictions_df.to_csv(self.predictions_path, index=False)
        log.info(f"\nPredictions saved: {self.predictions_path}")
        return predictions_df

    def _build_row(self, passage_row: pd.Series, model_result: dict,
                   prompt_key: str) -> dict:
        """Flatten passage metadata + model prediction into one row."""
        parsed = model_result.get("parsed") or {}
        row = {
            # Passage metadata
            "passage_id":     passage_row.get("passage_id", ""),
            "ticker":         passage_row.get("ticker", ""),
            "company":        passage_row.get("company", ""),
            "sector":         passage_row.get("sector", ""),
            "year":           passage_row.get("year", ""),
            "doc_type":       passage_row.get("doc_type", ""),
            "section":        passage_row.get("section", ""),
            "word_count":     passage_row.get("word_count", ""),
            # Gold labels (from human annotation)
            "gold_sentiment": passage_row.get("sentiment_label", ""),
            "gold_intensity": passage_row.get("sentiment_intensity", ""),
            "gold_guidance":  passage_row.get("has_guidance", ""),
            "gold_guid_type": passage_row.get("guidance_type", ""),
            # Model info
            "model":          model_result.get("model", ""),
            "model_label":    model_result.get("model_label", ""),
            "prompt_key":     prompt_key,
            "shots":          PROMPT_REGISTRY[prompt_key]["shots"],
            # Predictions
            "pred_sentiment": parsed.get("sentiment", ""),
            "pred_intensity": parsed.get("intensity", ""),
            "pred_guidance":  str(parsed.get("has_guidance", "")),
            "pred_guid_type": parsed.get("guidance_type", ""),
            "pred_key_phrase":parsed.get("sentiment_key_phrase") or parsed.get("key_phrase", ""),
            "pred_rationale": parsed.get("overall_rationale") or parsed.get("rationale", ""),
            "guid_confidence":parsed.get("guidance_confidence") or parsed.get("confidence", ""),
            "guid_spans":     json.dumps(parsed.get("guidance_spans", [])),
            # Run metadata
            "latency_ms":     model_result.get("latency_ms", ""),
            "error":          model_result.get("error", ""),
            "retry_count":    model_result.get("retry_count", 0),
            "raw_response":   model_result.get("raw_response", ""),
        }
        return row


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NGX-FND LLM Evaluator")
    parser.add_argument("--task",     choices=["sentiment","guidance","combined"],
                        default="combined")
    parser.add_argument("--shots",    type=int, choices=[0, 3, 5], default=5)
    parser.add_argument("--model",    choices=list(MODEL_CONFIG.keys()) + ["all"],
                        default="all")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Evaluate on N random passages (for testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume interrupted evaluation run")
    parser.add_argument("--baseline", action="store_true",
                        help="Include FinBERT baseline (requires transformers)")
    args = parser.parse_args()

    models = list(MODEL_CONFIG.keys()) if args.model == "all" else [args.model]

    run = EvaluationRun(
        task=args.task, shots=args.shots,
        models=models, sample=args.sample, resume=args.resume,
    )

    log.info(f"\n{'='*60}")
    log.info(f"NGX-FND LLM Evaluation Run")
    log.info(f"Task: {args.task} | Shots: {args.shots} | Models: {models}")
    log.info(f"Run ID: {run.run_id}")
    log.info(f"{'='*60}")

    predictions = run.run(run_baseline=args.baseline)
    log.info(f"\n[OK] Evaluation complete. {len(predictions)} prediction rows saved.")
    log.info(f"   Results: {run.out_dir}")
    log.info(f"   Run: python metrics.py --run_id {run.run_id}")


if __name__ == "__main__":
    main()
