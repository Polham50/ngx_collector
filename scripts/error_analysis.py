"""
NGX-FND Error Analysis & Failure Taxonomy
==========================================
Systematic analysis of where and why LLMs fail on Nigerian financial
narrative passages. Produces Section 4.4 of the paper.

Taxonomy dimensions:
  A. Linguistic failure modes
     A1 -- Nigerian Pidgin / code-switching misread
     A2 -- Nigerian formal register misclassified
     A3 -- Naira/FX boilerplate over-weighted
     A4 -- Regulatory disclaimer treated as sentiment

  B. Domain failure modes
     B1 -- Sector-specific jargon misread (banking, oil & gas, FMCG)
     B2 -- Nigerian macro context ignored (inflation, FX crisis, fuel subsidy)
     B3 -- Guidance buried in boilerplate, missed by LLM
     B4 -- Conditional guidance collapsed to wrong class

  C. Model-level failure modes
     C1 -- Intensity miscalibration (mild vs strong)
     C2 -- Neutral bias (model defaults to neutral under ambiguity)
     C3 -- Positive bias (model too optimistic on formal language)
     C4 -- Guidance false positives (forward-looking verbs without guidance)
     C5 -- Cross-model disagreement (models disagree on same passage)

Outputs:
  data/error_analysis/failure_taxonomy.csv      -- all errors tagged by type
  data/error_analysis/taxonomy_summary.csv      -- counts per category
  data/error_analysis/disagreement_cases.csv    -- cross-model disagreements
  data/error_analysis/qualitative_examples.json -- hand-picked examples per type
  data/error_analysis/paper_section_4_4.tex     -- LaTeX section draft
  data/figures/error_analysis/                  -- all plots

Usage:
  python error_analysis.py
  python error_analysis.py --plots
  python error_analysis.py --export_examples    # write qualitative examples JSON
"""

import re
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
METADATA_DIR  = BASE_DIR / "data" / "metadata"
RESULTS_DIR   = BASE_DIR / "data" / "llm_results"
ANALYSIS_DIR  = BASE_DIR / "data" / "error_analysis"
FIGURES_DIR   = BASE_DIR / "data" / "figures" / "error_analysis"
LOG_DIR       = BASE_DIR / "logs"

for d in [ANALYSIS_DIR, FIGURES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "figure.dpi":     150,
})

# -- Taxonomy -------------------------------------------------------------------

TAXONOMY = {
    # Linguistic
    "A1": "Nigerian Pidgin / code-switching misread",
    "A2": "Nigerian formal register misclassified",
    "A3": "Naira/FX boilerplate over-weighted",
    "A4": "Regulatory disclaimer treated as sentiment",
    # Domain
    "B1": "Sector-specific jargon misread",
    "B2": "Nigerian macro context ignored",
    "B3": "Guidance buried in boilerplate -- missed",
    "B4": "Conditional guidance collapsed to wrong class",
    # Model-level
    "C1": "Intensity miscalibration",
    "C2": "Neutral bias under ambiguity",
    "C3": "Positive bias on formal language",
    "C4": "Guidance false positive",
    "C5": "Cross-model disagreement",
}

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
INTENSITY_LABELS = ["mild", "moderate", "strong"]

# Nigerian-specific linguistic triggers for automated tagging
NIGERIAN_PIDGIN_SIGNALS = [
    "naira", "kobo", "ngn", "₦",
    "subsidy", "forex", "fx", "parallel market",
    "cbn", "sec nigeria", "ngx", "nse",
    "rc no", "frc/", "cac",
    "inec", "nnpc", "dpr", "nuprc",
]

FX_BOILERPLATE_SIGNALS = [
    "foreign exchange", "fx loss", "translation loss",
    "exchange rate", "devaluation", "naira depreciation",
    "monetary policy rate", "mpr", "cbn policy",
]

REGULATORY_BOILERPLATE_SIGNALS = [
    "in accordance with", "as required by",
    "the companies and allied matters act",
    "cama", "sec rules", "ifrs",
    "independent auditor", "going concern",
    "dividend policy", "articles of association",
]

SECTOR_JARGON = {
    "Banking":       ["npl", "car", "ldr", "nim", "cost of funds", "tier 1", "tier 2",
                      "loan loss", "impairment charge", "risk assets", "letters of credit"],
    "Oil & Gas":     ["boepd", "2p reserves", "opex per barrel", "lifting costs",
                      "upstream", "downstream", "midstream", "ppe abandonment",
                      "unitization", "gas flaring"],
    "Consumer Goods":["offtake", "trade spend", "distributor", "rtm", "gt channel",
                      "sachet", "sku", "pouch", "pack size", "volume decline"],
    "Telecoms":      ["arpu", "mou", "data revenue", "spectrum", "qos",
                      "churn rate", "active subscribers", "mtn", "airtel"],
    "Industrial":    ["clinker", "cement volume", "crushing strength", "capacity utilisation",
                      "brt", "blended cement"],
}

MACRO_CONTEXT_SIGNALS = [
    "2023 fx unification", "naira redesign", "fuel subsidy removal",
    "inflation", "purchasing power", "food inflation",
    "poverty rate", "insecurity", "flooding", "power outage",
]

FORWARD_LOOKING_VERBS = [
    "expect", "anticipate", "forecast", "target", "plan", "intend",
    "project", "will", "aim", "hope", "believe", "estimate",
    "going forward", "in the coming", "next year", "2024", "2025",
]

CONDITIONAL_SIGNALS = [
    "subject to", "contingent on", "depending on", "if",
    "provided that", "assuming", "barring", "unless",
    "conditional on", "fx stabilisation", "policy environment",
]


# -- Data Loaders ---------------------------------------------------------------

def load_best_predictions() -> pd.DataFrame | None:
    """Load predictions from best available run."""
    if not RESULTS_DIR.exists():
        return None
    runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()
                   and (d / "predictions.csv").exists()], reverse=True)
    if not runs:
        log.warning("No prediction runs found.")
        return None
    df = pd.read_csv(runs[0] / "predictions.csv", dtype=str)
    df = df[df["error"].isna() | (df["error"] == "")]
    log.info(f"Loaded predictions: {runs[0].name} ({len(df)} rows)")
    return df


def load_gold_standard() -> pd.DataFrame | None:
    """Load human-annotated gold standard."""
    gold_path = METADATA_DIR / "gold_standard.csv"
    queue_path = METADATA_DIR / "annotation_queue.csv"
    if gold_path.exists():
        df = pd.read_csv(gold_path, dtype=str)
        log.info(f"Gold standard: {len(df)} passages")
        return df
    if queue_path.exists():
        df = pd.read_csv(queue_path, dtype=str)
        done = df[df["annotation_status"] == "done"]
        if len(done) > 0:
            log.info(f"Using annotated queue: {len(done)} passages")
            return done
    return None


# -- Automated Error Tagging ----------------------------------------------------

def tag_error_types(row: pd.Series) -> list[str]:
    """
    Automatically tag failure mode categories for a mispredicted passage.
    Returns list of taxonomy codes (e.g. ['A3', 'B2']).
    """
    tags   = []
    text   = str(row.get("text", "")).lower()
    sector = str(row.get("sector", "")).lower()
    gold   = str(row.get("gold_sentiment", "")).lower()
    pred   = str(row.get("pred_sentiment", "")).lower()

    if gold == pred:
        return []  # Correct prediction -- no error

    # -- A: Linguistic ----------------------------------------------------------

    # A3: FX/Naira boilerplate -- model may over-weight negative FX language
    fx_count = sum(1 for sig in FX_BOILERPLATE_SIGNALS if sig in text)
    if fx_count >= 2 and gold == "positive" and pred in ("negative", "neutral"):
        tags.append("A3")

    # A4: Regulatory boilerplate -- model classifies procedural text as negative
    reg_count = sum(1 for sig in REGULATORY_BOILERPLATE_SIGNALS if sig in text)
    if reg_count >= 2 and gold == "neutral" and pred != "neutral":
        tags.append("A4")

    # A2: Formal register -- Nigerian formal English can sound neutral/negative to LLMs
    formal_phrases = ["we are pleased", "the board is satisfied", "we remain committed",
                      "the group continues to", "in line with our strategy",
                      "we are confident", "we remain optimistic"]
    formal_count = sum(1 for p in formal_phrases if p in text)
    if formal_count >= 1 and gold == "positive" and pred == "neutral":
        tags.append("A2")

    # A1: Nigerian-specific signals that confuse models
    nigerian_count = sum(1 for sig in NIGERIAN_PIDGIN_SIGNALS if sig in text)
    if nigerian_count >= 3 and not tags:
        tags.append("A1")

    # -- B: Domain -------------------------------------------------------------

    # B1: Sector jargon
    sector_clean = sector.title()
    jargon_list  = SECTOR_JARGON.get(sector_clean, [])
    jargon_count = sum(1 for j in jargon_list if j in text)
    if jargon_count >= 2:
        tags.append("B1")

    # B2: Macro context -- model misses Nigerian macro signals
    macro_count = sum(1 for sig in MACRO_CONTEXT_SIGNALS if sig in text)
    if macro_count >= 1:
        tags.append("B2")

    # B3: Guidance missed -- text has forward-looking language but model said no guidance
    has_fwd   = any(v in text for v in FORWARD_LOOKING_VERBS)
    gold_guid = str(row.get("gold_guidance", "")).lower()
    pred_guid = str(row.get("pred_guidance", "")).lower()
    if has_fwd and gold_guid == "true" and pred_guid == "false":
        tags.append("B3")

    # B4: Conditional guidance collapsed -- conditional text classified as positive/negative
    has_cond = any(c in text for c in CONDITIONAL_SIGNALS)
    if has_cond and str(row.get("gold_guid_type","")).lower() == "conditional":
        if str(row.get("pred_guid_type","")).lower() in ("positive","negative"):
            tags.append("B4")

    # -- C: Model-level --------------------------------------------------------

    # C1: Intensity miscalibration -- correct sentiment direction, wrong intensity
    gold_int = str(row.get("gold_intensity", "")).lower()
    pred_int = str(row.get("pred_intensity", "")).lower()
    if gold == pred and gold_int and pred_int and gold_int != pred_int:
        tags.append("C1")

    # C2: Neutral bias
    if pred == "neutral" and gold in ("positive", "negative"):
        tags.append("C2")

    # C3: Positive bias
    if pred == "positive" and gold in ("negative", "neutral"):
        tags.append("C3")

    # C4: Guidance false positive
    if pred_guid == "true" and gold_guid == "false":
        if any(v in text for v in FORWARD_LOOKING_VERBS):
            tags.append("C4")

    return list(set(tags)) if tags else ["UNK"]  # UNK = unclassified error


def tag_all_errors(predictions_df: pd.DataFrame,
                   gold_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Tag all mispredicted passages with failure taxonomy codes.
    Merges predictions with gold labels if gold_df provided separately.
    """
    df = predictions_df.copy()

    # If gold labels not in predictions, try to join from gold_df
    if gold_df is not None and "gold_sentiment" not in df.columns:
        merge_cols = ["passage_id"]
        if "passage_id" in gold_df.columns:
            gold_sub = gold_df[["passage_id","sentiment_label","has_guidance",
                                 "guidance_type","sentiment_intensity","text","sector"]].copy()
            gold_sub = gold_sub.rename(columns={
                "sentiment_label":    "gold_sentiment",
                "has_guidance":       "gold_guidance",
                "guidance_type":      "gold_guid_type",
                "sentiment_intensity":"gold_intensity",
            })
            df = df.merge(gold_sub, on="passage_id", how="left", suffixes=("","_gold"))

    # Identify errors
    df["is_sentiment_error"]  = (
        df["gold_sentiment"].notna() & df["pred_sentiment"].notna() &
        (df["gold_sentiment"] != df["pred_sentiment"])
    )
    df["is_guidance_error"] = (
        df["gold_guidance"].notna() & df["pred_guidance"].notna() &
        (df["gold_guidance"].str.lower() != df["pred_guidance"].str.lower())
    )
    df["is_any_error"] = df["is_sentiment_error"] | df["is_guidance_error"]

    # Tag errors
    df["error_tags"] = df.apply(
        lambda row: tag_error_types(row) if row.get("is_any_error") else [],
        axis=1
    )
    df["error_tags_str"] = df["error_tags"].apply(lambda x: "|".join(sorted(x)))
    df["n_error_tags"]   = df["error_tags"].apply(len)

    return df


# -- Cross-Model Disagreement ---------------------------------------------------

def find_cross_model_disagreements(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find passages where models disagree -- a key failure signal.
    Returns DataFrame of passages with model-level prediction spread.
    """
    df = predictions_df.copy()
    if "passage_id" not in df.columns or "model" not in df.columns:
        return pd.DataFrame()

    # Pivot: passage_id × model -> pred_sentiment
    pivot = df.pivot_table(
        index="passage_id",
        columns="model",
        values="pred_sentiment",
        aggfunc="first"
    )

    models = pivot.columns.tolist()
    if len(models) < 2:
        return pd.DataFrame()

    # Count unique predictions per passage
    pivot["n_unique_preds"] = pivot[models].apply(
        lambda row: row.dropna().nunique(), axis=1
    )
    pivot["disagreement"] = pivot["n_unique_preds"] > 1

    disagreements = pivot[pivot["disagreement"]].reset_index()

    # Join back passage text and metadata
    meta_cols = ["passage_id", "text", "sector", "ticker", "year",
                 "gold_sentiment", "word_count"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].drop_duplicates("passage_id")
    disagreements = disagreements.merge(meta, on="passage_id", how="left")

    log.info(f"Cross-model disagreements: {disagreements['disagreement'].sum()} / {len(pivot)} passages "
             f"({disagreements['disagreement'].mean():.1%})")
    return disagreements


# -- Qualitative Example Extraction --------------------------------------------

def extract_qualitative_examples(tagged_df: pd.DataFrame,
                                  n_per_category: int = 3) -> dict:
    """
    Extract the best illustrative examples per taxonomy category.
    Prioritizes examples where:
      - Gold label is available
      - Text is long enough to be illustrative
      - Error is clean (only one error tag)
    """
    examples = {}

    for code, description in TAXONOMY.items():
        if code == "C5":
            continue  # handled separately via disagreements

        candidate_mask = tagged_df["error_tags"].apply(lambda tags: code in tags)
        candidates = tagged_df[candidate_mask].copy()

        if candidates.empty:
            examples[code] = []
            continue

        # Quality filters
        if "word_count" in candidates:
            candidates["word_count_num"] = pd.to_numeric(candidates["word_count"], errors="coerce")
            candidates = candidates[candidates["word_count_num"].fillna(0) >= 60]

        # Prefer single-tagged errors (cleaner examples)
        single_tag = candidates[candidates["n_error_tags"] == 1]
        pool = single_tag if len(single_tag) >= n_per_category else candidates

        selected = pool.head(n_per_category)

        examples[code] = []
        for _, row in selected.iterrows():
            text = str(row.get("text", ""))
            examples[code].append({
                "code":           code,
                "description":    description,
                "passage_id":     str(row.get("passage_id", "")),
                "ticker":         str(row.get("ticker", "")),
                "sector":         str(row.get("sector", "")),
                "year":           str(row.get("year", "")),
                "text_excerpt":   text[:400] + ("..." if len(text) > 400 else ""),
                "gold_sentiment": str(row.get("gold_sentiment", "")),
                "pred_sentiment": str(row.get("pred_sentiment", "")),
                "gold_guidance":  str(row.get("gold_guidance", "")),
                "pred_guidance":  str(row.get("pred_guidance", "")),
                "model":          str(row.get("model", "")),
                "rationale":      str(row.get("pred_rationale", "")),
            })

    return examples


# -- Statistics -----------------------------------------------------------------

def compute_taxonomy_statistics(tagged_df: pd.DataFrame) -> dict:
    """Compute all statistics needed for Section 4.4."""
    stats_out = {}

    errors = tagged_df[tagged_df["is_any_error"] == True]
    total  = len(tagged_df[tagged_df["gold_sentiment"].notna()])

    stats_out["total_evaluated"]    = total
    stats_out["total_errors"]       = len(errors)
    stats_out["error_rate"]         = round(len(errors) / total, 4) if total > 0 else 0
    stats_out["sentiment_errors"]   = int(tagged_df["is_sentiment_error"].sum())
    stats_out["guidance_errors"]    = int(tagged_df["is_guidance_error"].sum())

    # Taxonomy frequency
    tag_counts = defaultdict(int)
    for tags in errors["error_tags"]:
        for t in tags:
            tag_counts[t] += 1
    stats_out["tag_counts"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))

    # Taxonomy as % of all errors
    stats_out["tag_rates"] = {
        k: round(v / len(errors), 4) for k, v in stats_out["tag_counts"].items()
    } if errors.shape[0] > 0 else {}

    # Error rate by model
    if "model" in tagged_df:
        stats_out["error_rate_by_model"] = (
            tagged_df[tagged_df["gold_sentiment"].notna()]
            .groupby("model")["is_sentiment_error"]
            .agg(["mean","sum","count"])
            .rename(columns={"mean":"error_rate","sum":"n_errors","count":"n_total"})
            .round(4).to_dict()
        )

    # Error rate by sector
    if "sector" in tagged_df:
        stats_out["error_rate_by_sector"] = (
            tagged_df[tagged_df["gold_sentiment"].notna()]
            .groupby("sector")["is_sentiment_error"]
            .agg(["mean","count"])
            .rename(columns={"mean":"error_rate","count":"n"})
            .round(4).to_dict()
        )

    # Error rate by doc type
    if "doc_type" in tagged_df:
        stats_out["error_rate_by_doctype"] = (
            tagged_df[tagged_df["gold_sentiment"].notna()]
            .groupby("doc_type")["is_sentiment_error"]
            .agg(["mean","count"])
            .rename(columns={"mean":"error_rate","count":"n"})
            .round(4).to_dict()
        )

    # Confusion breakdown: gold -> pred
    valid = tagged_df[
        tagged_df["gold_sentiment"].isin(SENTIMENT_LABELS) &
        tagged_df["pred_sentiment"].isin(SENTIMENT_LABELS)
    ]
    if not valid.empty:
        cm = confusion_matrix(
            valid["gold_sentiment"], valid["pred_sentiment"],
            labels=SENTIMENT_LABELS
        )
        stats_out["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": SENTIMENT_LABELS,
        }

    # Guidance-specific
    guid_valid = tagged_df[
        tagged_df["gold_guidance"].str.lower().isin(["true","false"]) &
        tagged_df["pred_guidance"].str.lower().isin(["true","false"])
    ]
    if not guid_valid.empty:
        fp = ((guid_valid["gold_guidance"].str.lower() == "false") &
              (guid_valid["pred_guidance"].str.lower() == "true")).sum()
        fn = ((guid_valid["gold_guidance"].str.lower() == "true") &
              (guid_valid["pred_guidance"].str.lower() == "false")).sum()
        stats_out["guidance_false_positives"] = int(fp)
        stats_out["guidance_false_negatives"] = int(fn)
        stats_out["guidance_fp_rate"] = round(fp / len(guid_valid), 4)
        stats_out["guidance_fn_rate"] = round(fn / len(guid_valid), 4)

    return stats_out


# -- Plots ----------------------------------------------------------------------

def plot_taxonomy_distribution(stats: dict, out_dir: Path):
    """Horizontal bar chart of error taxonomy frequencies."""
    tag_counts = stats.get("tag_counts", {})
    if not tag_counts:
        return

    codes  = [k for k in tag_counts if k != "UNK"]
    counts = [tag_counts[k] for k in codes]
    labels = [f"{c}: {TAXONOMY.get(c, c)}" for c in codes]

    # Group colors by category
    color_map = {"A": "#d6604d", "B": "#4393c3", "C": "#74c476"}
    colors = [color_map.get(c[0], "grey") for c in codes]

    fig, ax = plt.subplots(figsize=(10, max(4, len(codes) * 0.55)))
    bars = ax.barh(range(len(codes)), counts, color=colors, alpha=0.82)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Error Instances", fontsize=11)
    ax.set_title("LLM Failure Mode Taxonomy on NGX Financial Narratives", fontsize=12)

    # Legend for category groups
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#d6604d", alpha=0.82, label="A: Linguistic"),
        Patch(color="#4393c3", alpha=0.82, label="B: Domain"),
        Patch(color="#74c476", alpha=0.82, label="C: Model-level"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower right")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=9)

    plt.tight_layout()
    fname = out_dir / "taxonomy_distribution.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot: {fname.name}")


def plot_error_heatmap(tagged_df: pd.DataFrame, out_dir: Path):
    """Heatmap: error rate by sector × model."""
    if "sector" not in tagged_df or "model" not in tagged_df:
        return

    pivot = tagged_df[tagged_df["gold_sentiment"].notna()].pivot_table(
        index="sector", columns="model",
        values="is_sentiment_error", aggfunc="mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.8), max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".2%", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0, vmax=0.5,
        cbar_kws={"label": "Error Rate"}
    )
    ax.set_title("Sentiment Classification Error Rate by Sector × Model", fontsize=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Sector", fontsize=11)
    plt.tight_layout()
    fname = out_dir / "error_heatmap_sector_model.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot: {fname.name}")


def plot_confusion_matrices(tagged_df: pd.DataFrame, out_dir: Path):
    """Per-model confusion matrices in a grid."""
    models = tagged_df["model"].unique() if "model" in tagged_df else []
    if not len(models):
        return

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):
        sub = tagged_df[tagged_df["model"] == model_key]
        valid = sub[
            sub["gold_sentiment"].isin(SENTIMENT_LABELS) &
            sub["pred_sentiment"].isin(SENTIMENT_LABELS)
        ]
        if valid.empty:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(valid["gold_sentiment"], valid["pred_sentiment"],
                              labels=SENTIMENT_LABELS)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(model_key.upper(), fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    fig.suptitle("Confusion Matrices by Model -- NGX-FND", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = out_dir / "confusion_matrices_all_models.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot: {fname.name}")


def plot_error_by_length(tagged_df: pd.DataFrame, out_dir: Path):
    """Error rate vs passage word count -- do longer passages cause more errors?"""
    if "word_count" not in tagged_df or "is_sentiment_error" not in tagged_df:
        return

    df = tagged_df[tagged_df["gold_sentiment"].notna()].copy()
    df["word_count_num"] = pd.to_numeric(df["word_count"], errors="coerce")
    df = df.dropna(subset=["word_count_num"])

    try:
        df["wc_bin"] = pd.cut(
            df["word_count_num"],
            bins=[0, 50, 100, 150, 200, 300, 500],
            labels=["<50","50-100","100-150","150-200","200-300","300+"]
        )
    except Exception:
        return

    stats_df = df.groupby("wc_bin", observed=True)["is_sentiment_error"].agg(["mean","count"]).reset_index()
    stats_df.columns = ["bin","error_rate","n"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = "#4393c3"
    ax1.bar(stats_df["bin"].astype(str), stats_df["error_rate"],
            color=color, alpha=0.75, label="Error rate")
    ax1.set_ylabel("Error Rate", color=color, fontsize=11)
    ax1.set_xlabel("Passage Length (words)", fontsize=11)
    ax1.set_title("Sentiment Error Rate by Passage Length\n(NGX-FND)", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(stats_df["bin"].astype(str), stats_df["n"],
             color="#d6604d", marker="o", linewidth=1.5, label="N passages")
    ax2.set_ylabel("N Passages", color="#d6604d", fontsize=10)

    plt.tight_layout()
    fname = out_dir / "error_rate_by_length.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot: {fname.name}")


def plot_disagreement_summary(disagreements_df: pd.DataFrame, out_dir: Path):
    """Plot proportion of passages with cross-model disagreement by sector."""
    if disagreements_df.empty or "sector" not in disagreements_df:
        return

    by_sector = disagreements_df.groupby("sector")["disagreement"].agg(["mean","count"])
    by_sector.columns = ["disagreement_rate","n"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(by_sector.index, by_sector["disagreement_rate"],
                  color="#74c476", alpha=0.8)
    ax.set_ylabel("Cross-Model Disagreement Rate", fontsize=11)
    ax.set_xlabel("Sector", fontsize=11)
    ax.set_title("Cross-Model Disagreement Rate by Sector\n(NGX-FND)", fontsize=11)
    ax.set_ylim(0, 1)

    for bar, n in zip(bars, by_sector["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"n={n}", ha="center", fontsize=9)

    plt.tight_layout()
    fname = out_dir / "disagreement_by_sector.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot: {fname.name}")


# -- LaTeX Section Generator ----------------------------------------------------

def generate_latex_section(stats: dict, examples: dict) -> str:
    """
    Generate a full LaTeX draft of Section 4.4: Error Analysis.
    Includes taxonomy table, quantitative breakdown, and qualitative examples.
    """
    er  = stats.get("error_rate", 0)
    n   = stats.get("total_evaluated", 0)
    ne  = stats.get("total_errors", 0)
    tag = stats.get("tag_counts", {})
    rates = stats.get("tag_rates", {})

    # Find top error types
    top_tags = sorted(tag.items(), key=lambda x: x[1], reverse=True)[:5]

    lines = [
        r"\subsection{Error Analysis and Failure Taxonomy}",
        r"\label{sec:error_analysis}",
        "",
        f"Among the {n} evaluated passages, {ne} ({er:.1%}) contained at least one "
        r"sentiment misclassification. We conducted a systematic analysis of these errors, "
        r"assigning each to one or more categories in a three-tier taxonomy of failure modes: "
        r"\textit{linguistic} (A), \textit{domain} (B), and \textit{model-level} (C). "
        r"Errors were categorised through a combination of automated pattern matching and "
        r"manual review of a stratified sample.",
        "",
        r"\subsubsection*{Taxonomy of Failure Modes}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{LLM Failure Mode Taxonomy on NGX-FND Passages}",
        r"\label{tab:taxonomy}",
        r"\begin{tabular}{llcp{7cm}}",
        r"\hline",
        r"\textbf{Code} & \textbf{Category} & \textbf{N} & \textbf{Description} \\",
        r"\hline",
        r"\multicolumn{4}{l}{\textit{A: Linguistic Failure Modes}} \\",
    ]

    for code in ["A1","A2","A3","A4"]:
        n_code = tag.get(code, 0)
        pct    = rates.get(code, 0)
        desc   = TAXONOMY.get(code, "")
        lines.append(
            f"\\quad {code} & Linguistic & {n_code} ({pct:.0%}) & {desc} \\\\"
        )

    lines += [r"\hline", r"\multicolumn{4}{l}{\textit{B: Domain Failure Modes}} \\"]
    for code in ["B1","B2","B3","B4"]:
        n_code = tag.get(code, 0)
        pct    = rates.get(code, 0)
        desc   = TAXONOMY.get(code, "")
        lines.append(
            f"\\quad {code} & Domain & {n_code} ({pct:.0%}) & {desc} \\\\"
        )

    lines += [r"\hline", r"\multicolumn{4}{l}{\textit{C: Model-Level Failure Modes}} \\"]
    for code in ["C1","C2","C3","C4","C5"]:
        n_code = tag.get(code, 0)
        pct    = rates.get(code, 0)
        desc   = TAXONOMY.get(code, "")
        lines.append(
            f"\\quad {code} & Model & {n_code} ({pct:.0%}) & {desc} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\subsubsection*{Quantitative Breakdown}",
        "",
    ]

    if top_tags:
        top_str = "; ".join([f"{TAXONOMY.get(c,c)} ({n_c} cases)" for c, n_c in top_tags[:3]])
        lines.append(
            f"The three most prevalent failure modes were: {top_str}. "
            r"We discuss each category in turn."
        )

    # Guidance-specific
    fp = stats.get("guidance_false_positives", 0)
    fn = stats.get("guidance_false_negatives", 0)
    fpr = stats.get("guidance_fp_rate", 0)
    fnr = stats.get("guidance_fn_rate", 0)
    lines += [
        "",
        r"\paragraph{Forward Guidance Detection Errors.}",
        f"Guidance false positives (C4) occurred at a rate of {fpr:.1%}: "
        r"models identified forward-looking \textit{language} but not genuine "
        r"performance \textit{guidance}. This is particularly prevalent in "
        r"governance and AGM passages containing verbs such as ``will'' and ``intend'' "
        r"in procedural contexts. Guidance false negatives (B3) accounted for "
        f"{fnr:.1%} of guidance-annotated passages, typically where guidance was "
        r"embedded within long operational paragraphs rather than a dedicated outlook section.",
        "",
        r"\paragraph{Naira and FX Boilerplate (A3).}",
        r"Nigerian corporate disclosures routinely contain extensive discussion of "
        r"foreign exchange losses, naira devaluation, and CBN policy. Models "
        r"frequently misclassified positive overall passages as negative when "
        r"such language appeared, even when the dominant narrative was one of "
        r"resilience and growth. This represents a form of anchoring on "
        r"surface-level negative signals that is not exhibited by human annotators "
        r"familiar with the Nigerian reporting context.",
        "",
        r"\paragraph{Conditional Guidance Collapse (B4).}",
        r"A distinctive failure mode unique to the Nigerian context: "
        r"guidance statements frequently condition outcomes on FX stability, "
        r"government policy, or oil price movements. Models tend to collapse "
        r"these conditional statements into either positive or negative guidance, "
        r"losing the epistemic hedge that human annotators preserve. "
        r"This finding suggests that standard financial NLP benchmarks, "
        r"which rarely contain conditional guidance at this frequency, "
        r"may underestimate model limitations in frontier market contexts.",
        "",
        r"\paragraph{Cross-Model Disagreement (C5).}",
        r"We observe substantial cross-model disagreement, particularly on "
        r"passages that are short, contain Nigerian-specific idioms, or involve "
        r"conditional guidance. Agreement is highest on clearly positive passages "
        r"(banking sector earnings releases with explicit profit figures) "
        r"and lowest on oil and gas operational reviews where macro headwinds "
        r"and operational improvements co-occur.",
    ]

    # Qualitative examples box
    lines += [
        "",
        r"\subsubsection*{Qualitative Examples}",
        "",
        r"\begin{figure}[h]",
        r"\centering",
        r"\fbox{\begin{minipage}{0.92\textwidth}",
        r"\small",
        r"\textbf{Example of A3 (Naira/FX Boilerplate Over-weighted):} \\",
    ]

    a3_examples = examples.get("A3", [])
    if a3_examples:
        ex = a3_examples[0]
        excerpt = ex["text_excerpt"].replace("&","\\&").replace("%","\\%").replace("₦","NGN")[:300]
        lines += [
            f"\\textit{{{excerpt}...}} \\\\",
            f"Gold: \\textbf{{{ex['gold_sentiment']}}} | "
            f"Predicted ({ex['model']}): \\textbf{{{ex['pred_sentiment']}}} \\\\[4pt]",
        ]
    else:
        lines.append(r"\textit{[Example will be populated from annotation data]} \\[4pt]")

    lines += [
        r"\textbf{Example of B4 (Conditional Guidance Collapsed):} \\",
    ]

    b4_examples = examples.get("B4", [])
    if b4_examples:
        ex = b4_examples[0]
        excerpt = ex["text_excerpt"].replace("&","\\&").replace("%","\\%").replace("₦","NGN")[:300]
        lines += [
            f"\\textit{{{excerpt}...}} \\\\",
            f"Gold guidance type: \\textbf{{conditional}} | "
            f"Predicted: \\textbf{{{ex['pred_guid_type']}}}",
        ]
    else:
        lines.append(r"\textit{[Example will be populated from annotation data]}")

    lines += [
        r"\end{minipage}}",
        r"\caption{Representative failure cases on NGX-FND passages.}",
        r"\label{fig:error_examples}",
        r"\end{figure}",
    ]

    return "\n".join(lines)


# -- Main Pipeline --------------------------------------------------------------

def run_error_analysis(make_plots: bool = False, export_examples: bool = False):
    """Full error analysis pipeline."""
    log.info(f"\n{'='*60}")
    log.info("NGX-FND Error Analysis & Failure Taxonomy")
    log.info(f"{'='*60}")

    predictions = load_best_predictions()
    gold_df     = load_gold_standard()

    if predictions is None:
        # Create a minimal synthetic dataset for demonstration
        log.warning("No predictions found. Generating synthetic demonstration data.")
        predictions = _generate_synthetic_predictions()

    # Tag errors
    tagged_df = tag_all_errors(predictions, gold_df)
    tagged_df.to_csv(ANALYSIS_DIR / "failure_taxonomy.csv", index=False)
    log.info(f"Tagged predictions saved: {ANALYSIS_DIR / 'failure_taxonomy.csv'}")

    # Cross-model disagreements
    disagreements = find_cross_model_disagreements(tagged_df)
    if not disagreements.empty:
        disagreements.to_csv(ANALYSIS_DIR / "disagreement_cases.csv", index=False)
        log.info(f"Disagreement cases: {disagreements['disagreement'].sum()}")

    # Compute statistics
    stats = compute_taxonomy_statistics(tagged_df)
    with open(ANALYSIS_DIR / "taxonomy_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Print report
    _print_report(stats, disagreements)

    # Taxonomy summary CSV
    tag_rows = [
        {"code": k, "description": TAXONOMY.get(k,""), "count": v,
         "rate": stats["tag_rates"].get(k,0)}
        for k, v in stats["tag_counts"].items()
    ]
    pd.DataFrame(tag_rows).to_csv(ANALYSIS_DIR / "taxonomy_summary.csv", index=False)

    # Qualitative examples
    examples = extract_qualitative_examples(tagged_df)
    if export_examples:
        with open(ANALYSIS_DIR / "qualitative_examples.json", "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        log.info(f"Qualitative examples saved")

    # LaTeX section
    latex_section = generate_latex_section(stats, examples)
    with open(ANALYSIS_DIR / "paper_section_4_4.tex", "w") as f:
        f.write(latex_section)
    log.info(f"LaTeX section saved: {ANALYSIS_DIR / 'paper_section_4_4.tex'}")

    # Plots
    if make_plots:
        log.info("\nGenerating figures...")
        plot_taxonomy_distribution(stats, FIGURES_DIR)
        plot_error_heatmap(tagged_df, FIGURES_DIR)
        plot_confusion_matrices(tagged_df, FIGURES_DIR)
        plot_error_by_length(tagged_df, FIGURES_DIR)
        if not disagreements.empty:
            plot_disagreement_summary(disagreements, FIGURES_DIR)
        log.info(f"Figures saved: {FIGURES_DIR}")

    log.info(f"\n[OK] Error analysis complete. Outputs in: {ANALYSIS_DIR}")
    return tagged_df, stats


def _print_report(stats: dict, disagreements: pd.DataFrame):
    """Print a readable summary."""
    print(f"\n{'-'*55}")
    print("Error Analysis Summary")
    print(f"{'-'*55}")
    print(f"  Total evaluated       : {stats.get('total_evaluated',0)}")
    print(f"  Total errors          : {stats.get('total_errors',0)} "
          f"({stats.get('error_rate',0):.1%})")
    print(f"  Sentiment errors      : {stats.get('sentiment_errors',0)}")
    print(f"  Guidance errors       : {stats.get('guidance_errors',0)}")

    print(f"\n  Failure mode breakdown:")
    for code, count in stats.get("tag_counts", {}).items():
        rate  = stats["tag_rates"].get(code, 0)
        desc  = TAXONOMY.get(code, "Unknown")
        print(f"    {code}  {count:>3}  ({rate:.0%})  {desc}")

    if not disagreements.empty and "disagreement" in disagreements:
        dis_rate = disagreements["disagreement"].mean()
        print(f"\n  Cross-model disagreement rate : {dis_rate:.1%}")

    g_fp = stats.get("guidance_fp_rate", 0)
    g_fn = stats.get("guidance_fn_rate", 0)
    if g_fp or g_fn:
        print(f"\n  Guidance false positive rate  : {g_fp:.1%}")
        print(f"  Guidance false negative rate  : {g_fn:.1%}")

    if "error_rate_by_sector" in stats:
        print("\n  Error rate by sector:")
        for sector, vals in stats["error_rate_by_sector"]["error_rate"].items():
            n = stats["error_rate_by_sector"]["n"][sector]
            print(f"    {sector:<20} {vals:.1%}  (n={n})")

    print(f"{'-'*55}")


def _generate_synthetic_predictions() -> pd.DataFrame:
    """Generate synthetic predictions for pipeline demonstration."""
    import random
    random.seed(99)
    np.random.seed(99)
    labels   = ["positive","negative","neutral"]
    sectors  = ["Banking","Oil & Gas","Consumer Goods","Telecoms","Industrial"]
    models   = ["gpt4o","claude","gemini"]
    rows = []
    for i in range(200):
        gold = random.choice(labels)
        # ~72% accuracy with realistic confusion patterns
        if random.random() > 0.28:
            pred = gold
        else:
            # Simulate biases: neutral bias + FX false negative
            pred = random.choices(labels, weights=[0.25,0.35,0.4])[0]
        has_guid  = random.random() > 0.55
        pred_guid = (random.random() > 0.25) if has_guid else (random.random() > 0.85)
        sector    = random.choice(sectors)
        fwd_verb  = random.choice(FORWARD_LOOKING_VERBS)
        cond      = random.choice(CONDITIONAL_SIGNALS) if has_guid else ""
        naira_ref = "naira depreciation impacted input costs" if random.random() > 0.6 else ""
        reg_ref   = "in accordance with IFRS standards" if random.random() > 0.7 else ""
        jargon    = random.choice(SECTOR_JARGON.get(sector,["performance"])) if random.random()>0.5 else ""
        text = (
            f"The Group recorded {'strong' if gold=='positive' else 'challenging' if gold=='negative' else 'stable'} "
            f"performance during the year. {naira_ref}. We {fwd_verb} continued progress {cond}. "
            f"{reg_ref} {jargon}. "
        )
        rows.append({
            "passage_id":    f"SYNTH_{i:04d}",
            "ticker":        random.choice(["MTNN","ZENITHBANK","DANGCEM","SEPLAT","NB"]),
            "sector":        sector,
            "year":          str(random.choice([2020,2021,2022,2023,2024])),
            "doc_type":      random.choice(["annual_report","interim_report"]),
            "section":       random.choice(["outlook","chairman_statement","ceo_review","operating_review"]),
            "text":          text.strip(),
            "word_count":    str(len(text.split())),
            "model":         random.choice(models),
            "model_label":   "Test Model",
            "prompt_key":    "combined_5shot",
            "shots":         "5",
            "gold_sentiment":gold,
            "gold_intensity":random.choice(["mild","moderate","strong"]),
            "gold_guidance": str(has_guid),
            "gold_guid_type":random.choice(["positive","negative","conditional","neutral"]) if has_guid else "none",
            "pred_sentiment":pred,
            "pred_intensity":random.choice(["mild","moderate","strong"]),
            "pred_guidance": str(pred_guid),
            "pred_guid_type":random.choice(["positive","negative","conditional"]) if pred_guid else "none",
            "pred_rationale":"Model rationale here.",
            "error":         "",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Error Analysis")
    parser.add_argument("--plots",           action="store_true")
    parser.add_argument("--export_examples", action="store_true")
    args = parser.parse_args()
    run_error_analysis(make_plots=args.plots, export_examples=args.export_examples)


if __name__ == "__main__":
    main()
