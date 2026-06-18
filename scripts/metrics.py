"""
NGX-FND Metrics Calculator
============================
Computes evaluation metrics from LLM predictions and generates
publication-ready tables and plots for the paper.

Metrics computed:
  - Accuracy, Macro-F1, Weighted-F1, Cohen's Kappa (vs gold labels)
  - Per-class precision, recall, F1
  - Guidance detection: precision, recall, F1, AUROC
  - Inter-model agreement
  - Performance breakdowns: by sector, year, doc_type, section, shot count

Outputs:
  - results_summary.csv     -- per-model aggregated metrics
  - confusion_matrices/     -- per-model confusion matrix CSVs
  - paper_tables.txt        -- LaTeX-formatted tables
  - figures/                -- matplotlib plots

Usage:
  python metrics.py --run_id combined_5shot_20240611_1430
  python metrics.py --run_id combined_5shot_20240611_1430 --plots
  python metrics.py --compare run1 run2 run3   # compare across runs
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent
RESULTS_DIR  = BASE_DIR / "data" / "llm_results"
FIGURES_DIR  = BASE_DIR / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -- Label constants ------------------------------------------------------------
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
GUIDANCE_LABELS  = ["True", "False"]
INTENSITY_LABELS = ["mild", "moderate", "strong"]


# -- Loaders --------------------------------------------------------------------

def load_predictions(run_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / run_id / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path, dtype=str)
    # Drop rows with API errors
    df = df[df["error"].isna() | (df["error"] == "")]
    # Drop rows with missing predictions
    df = df[df["pred_sentiment"].notna() & (df["pred_sentiment"] != "")]
    return df


# -- Core Metrics ---------------------------------------------------------------

def compute_sentiment_metrics(df: pd.DataFrame,
                               gold_col: str = "gold_sentiment",
                               pred_col: str = "pred_sentiment") -> dict:
    """Compute full sentiment classification metrics."""
    valid = df[df[gold_col].isin(SENTIMENT_LABELS) &
               df[pred_col].isin(SENTIMENT_LABELS)].copy()

    if valid.empty:
        return {"error": "No valid gold/pred pairs found"}

    y_true = valid[gold_col].tolist()
    y_pred = valid[pred_col].tolist()

    metrics = {
        "n_samples":      len(valid),
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":       round(f1_score(y_true, y_pred, average="macro",    labels=SENTIMENT_LABELS, zero_division=0), 4),
        "weighted_f1":    round(f1_score(y_true, y_pred, average="weighted", labels=SENTIMENT_LABELS, zero_division=0), 4),
        "kappa":          round(cohen_kappa_score(y_true, y_pred), 4),
        "macro_precision":round(precision_score(y_true, y_pred, average="macro",    labels=SENTIMENT_LABELS, zero_division=0), 4),
        "macro_recall":   round(recall_score(y_true, y_pred, average="macro",       labels=SENTIMENT_LABELS, zero_division=0), 4),
    }

    # Per-class
    for label in SENTIMENT_LABELS:
        binary_true = [1 if y == label else 0 for y in y_true]
        binary_pred = [1 if y == label else 0 for y in y_pred]
        metrics[f"f1_{label}"]        = round(f1_score(binary_true, binary_pred, zero_division=0), 4)
        metrics[f"precision_{label}"] = round(precision_score(binary_true, binary_pred, zero_division=0), 4)
        metrics[f"recall_{label}"]    = round(recall_score(binary_true, binary_pred, zero_division=0), 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compute_guidance_metrics(df: pd.DataFrame,
                              gold_col: str = "gold_guidance",
                              pred_col: str = "pred_guidance") -> dict:
    """Compute binary guidance detection metrics."""
    valid = df[df[gold_col].isin(["True","False","true","false"]) &
               df[pred_col].isin(["True","False","true","false"])].copy()

    if valid.empty:
        return {"error": "No valid guidance labels found"}

    # Normalize to boolean int
    y_true = valid[gold_col].str.lower().map({"true": 1, "false": 0}).tolist()
    y_pred = valid[pred_col].str.lower().map({"true": 1, "false": 0}).tolist()

    metrics = {
        "n_samples":   len(valid),
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_true, y_pred, zero_division=0), 4),
        "kappa":       round(cohen_kappa_score(y_true, y_pred), 4),
        "prevalence":  round(sum(y_true) / len(y_true), 4),
    }

    # Guidance type accuracy (conditional on has_guidance=True)
    pos_rows = valid[valid[gold_col].str.lower() == "true"].copy()
    if len(pos_rows) > 0 and "gold_guid_type" in pos_rows and "pred_guid_type" in pos_rows:
        guid_types = ["positive","negative","neutral","conditional"]
        type_valid = pos_rows[pos_rows["gold_guid_type"].isin(guid_types) &
                              pos_rows["pred_guid_type"].isin(guid_types)]
        if len(type_valid) > 0:
            metrics["guidance_type_accuracy"] = round(
                accuracy_score(type_valid["gold_guid_type"], type_valid["pred_guid_type"]), 4
            )
            metrics["guidance_type_macro_f1"] = round(
                f1_score(type_valid["gold_guid_type"], type_valid["pred_guid_type"],
                         average="macro", zero_division=0), 4
            )

    return metrics


def compute_error_analysis(df: pd.DataFrame) -> dict:
    """Identify systematic failure patterns."""
    analysis = {}

    valid = df[df["gold_sentiment"].isin(SENTIMENT_LABELS) &
               df["pred_sentiment"].isin(SENTIMENT_LABELS)].copy()
    if valid.empty:
        return analysis

    valid["correct"] = valid["gold_sentiment"] == valid["pred_sentiment"]

    # Error rate by sector
    if "sector" in valid:
        analysis["error_by_sector"] = (
            valid.groupby("sector")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .round(4).to_dict()
        )

    # Error rate by year
    if "year" in valid:
        analysis["error_by_year"] = (
            valid.groupby("year")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .round(4).to_dict()
        )

    # Error rate by section type
    if "section" in valid:
        analysis["error_by_section"] = (
            valid.groupby("section")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .round(4).to_dict()
        )

    # Error rate by word count quartile
    if "word_count" in valid:
        valid["word_count_num"] = pd.to_numeric(valid["word_count"], errors="coerce")
        valid["wc_quartile"] = pd.qcut(valid["word_count_num"], q=4,
                                        labels=["Q1(short)","Q2","Q3","Q4(long)"],
                                        duplicates="drop")
        analysis["error_by_length"] = (
            valid.groupby("wc_quartile")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .round(4).to_dict()
        )

    # Most common misclassifications
    errors = valid[~valid["correct"]]
    if len(errors) > 0:
        analysis["common_errors"] = (
            errors.groupby(["gold_sentiment", "pred_sentiment"])
            .size().reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10).to_dict("records")
        )

    return analysis


def compute_shot_comparison(run_dir: Path) -> pd.DataFrame:
    """Compare 0-shot vs 3-shot vs 5-shot within same run directory."""
    rows = []
    for pred_file in run_dir.glob("*/predictions.csv"):
        run_id  = pred_file.parent.name
        df      = pd.read_csv(pred_file, dtype=str)
        df      = df[df["error"].isna() | (df["error"] == "")]

        for model_key in df["model"].unique():
            sub = df[df["model"] == model_key]
            shots_val = sub["shots"].iloc[0] if "shots" in sub else "?"
            m = compute_sentiment_metrics(sub)
            rows.append({
                "run_id":      run_id,
                "model":       model_key,
                "shots":       shots_val,
                "accuracy":    m.get("accuracy"),
                "macro_f1":    m.get("macro_f1"),
                "kappa":       m.get("kappa"),
                "n_samples":   m.get("n_samples"),
            })
    return pd.DataFrame(rows)


# -- LaTeX Table Generators -----------------------------------------------------

def latex_main_results_table(summary_df: pd.DataFrame) -> str:
    """Generate the main results table (Table 2 in the paper)."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{LLM Performance on NGX-FND: Sentiment Classification and Forward Guidance Detection}",
        r"\label{tab:main_results}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Shots} & \textbf{Acc} & \textbf{Macro-F1} & "
        r"\textbf{$\kappa$} & \textbf{F1$_{pos}$} & \textbf{F1$_{neg}$} & \textbf{F1$_{neu}$} \\",
        r"\hline",
        r"\multicolumn{8}{l}{\textit{Sentiment Classification}} \\",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row.get('model_label',''):<20} & {row.get('shots','')} & "
            f"{row.get('accuracy',''):.3f} & {row.get('macro_f1',''):.3f} & "
            f"{row.get('kappa',''):.3f} & "
            f"{row.get('f1_positive',''):.3f} & {row.get('f1_negative',''):.3f} & "
            f"{row.get('f1_neutral',''):.3f} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_guidance_table(guidance_df: pd.DataFrame) -> str:
    """Generate guidance detection results table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Forward Guidance Detection Performance on NGX-FND}",
        r"\label{tab:guidance_results}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Shots} & \textbf{Acc} & \textbf{Prec} & "
        r"\textbf{Rec} & \textbf{F1} \\",
        r"\hline",
    ]
    for _, row in guidance_df.iterrows():
        lines.append(
            f"{row.get('model_label',''):<20} & {row.get('shots','')} & "
            f"{row.get('accuracy',''):.3f} & {row.get('precision',''):.3f} & "
            f"{row.get('recall',''):.3f} & {row.get('f1',''):.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# -- Plots ----------------------------------------------------------------------

def plot_confusion_matrix(cm: list, model_label: str, out_dir: Path):
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    # Normalize
    cm_norm = cm_arr.astype(float) / (cm_arr.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=cm_arr, fmt="d", cmap="Blues",
                xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS,
                ax=ax, cbar=True, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Gold", fontsize=12)
    ax.set_title(f"Confusion Matrix -- {model_label}", fontsize=13)
    plt.tight_layout()
    fname = out_dir / f"cm_{model_label.lower().replace(' ','_')}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname


def plot_f1_comparison(summary_df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing macro-F1 across models and shot counts."""
    fig, ax = plt.subplots(figsize=(9, 5))
    models  = summary_df["model_label"].unique()
    shots   = sorted(summary_df["shots"].unique())
    x       = np.arange(len(models))
    width   = 0.25
    colors  = ["#2c7bb6", "#abd9e9", "#fdae61"]

    for i, shot in enumerate(shots):
        sub = summary_df[summary_df["shots"] == shot].set_index("model_label")
        vals = [sub.loc[m, "macro_f1"] if m in sub.index else 0 for m in models]
        ax.bar(x + i * width, vals, width, label=f"{shot}-shot", color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Sentiment Classification: Macro-F1 by Model and Shot Count\n(NGX-FND Dataset)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.6, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    fname = out_dir / "f1_comparison.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname


def plot_sector_accuracy(error_analysis: dict, model_label: str, out_dir: Path):
    """Plot accuracy breakdown by sector."""
    sector_data = error_analysis.get("error_by_sector", {})
    if not sector_data or "accuracy" not in sector_data:
        return None

    sectors  = list(sector_data["accuracy"].keys())
    accs     = [sector_data["accuracy"][s] for s in sectors]
    counts   = [sector_data["n"][s] for s in sectors]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sectors, accs, color="#4292c6", alpha=0.85)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(f"Sentiment Accuracy by Sector -- {model_label}", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"n={count}", va="center", fontsize=9)

    plt.tight_layout()
    fname = out_dir / f"sector_accuracy_{model_label.lower().replace(' ','_')}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname


# -- Main Pipeline --------------------------------------------------------------

def run_metrics(run_id: str, make_plots: bool = False):
    """Compute all metrics for a completed evaluation run."""
    import logging
    log = logging.getLogger(__name__)

    run_dir = RESULTS_DIR / run_id
    df      = load_predictions(run_id)
    fig_dir = FIGURES_DIR / run_id
    fig_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"NGX-FND Metrics Report: {run_id}")
    print(f"{'='*60}")
    print(f"Total prediction rows: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")

    sentiment_summary = []
    guidance_summary  = []

    for model_key in df["model"].unique():
        sub          = df[df["model"] == model_key].copy()
        model_label  = sub["model_label"].iloc[0] if "model_label" in sub else model_key
        shots_val    = sub["shots"].iloc[0]        if "shots" in sub else "?"

        print(f"\n{'-'*50}")
        print(f"Model: {model_label} ({shots_val}-shot) | n={len(sub)}")
        print(f"{'-'*50}")

        # Sentiment metrics
        sent_m = compute_sentiment_metrics(sub)
        if "error" not in sent_m:
            print(f"  Sentiment Accuracy  : {sent_m['accuracy']:.3f}")
            print(f"  Sentiment Macro-F1  : {sent_m['macro_f1']:.3f}")
            print(f"  Cohen's Kappa       : {sent_m['kappa']:.3f}")
            print(f"  F1 positive         : {sent_m['f1_positive']:.3f}")
            print(f"  F1 negative         : {sent_m['f1_negative']:.3f}")
            print(f"  F1 neutral          : {sent_m['f1_neutral']:.3f}")
            sentiment_summary.append({
                "model": model_key, "model_label": model_label, "shots": shots_val,
                **{k: v for k, v in sent_m.items() if k != "confusion_matrix"}
            })

            if make_plots and sent_m.get("confusion_matrix"):
                plot_confusion_matrix(sent_m["confusion_matrix"], model_label, fig_dir)

        # Guidance metrics
        guid_m = compute_guidance_metrics(sub)
        if "error" not in guid_m:
            print(f"  Guidance Accuracy   : {guid_m['accuracy']:.3f}")
            print(f"  Guidance F1         : {guid_m['f1']:.3f}")
            print(f"  Guidance Precision  : {guid_m['precision']:.3f}")
            print(f"  Guidance Recall     : {guid_m['recall']:.3f}")
            guidance_summary.append({
                "model": model_key, "model_label": model_label, "shots": shots_val,
                **guid_m
            })

        # Error analysis
        err = compute_error_analysis(sub)
        if err.get("error_by_sector"):
            print("\n  Accuracy by sector:")
            for sector, vals in err["error_by_sector"]["accuracy"].items():
                n = err["error_by_sector"]["n"][sector]
                print(f"    {sector:<20} {vals:.3f}  (n={n})")

            if make_plots:
                plot_sector_accuracy(err, model_label, fig_dir)

    # Save summary CSVs
    if sentiment_summary:
        sent_df = pd.DataFrame(sentiment_summary)
        sent_df.to_csv(run_dir / "sentiment_metrics.csv", index=False)

        if make_plots:
            plot_f1_comparison(sent_df, fig_dir)

    if guidance_summary:
        guid_df = pd.DataFrame(guidance_summary)
        guid_df.to_csv(run_dir / "guidance_metrics.csv", index=False)

    # Generate LaTeX tables
    paper_tables = []
    if sentiment_summary:
        paper_tables.append("% === TABLE 2: Main Results ===")
        paper_tables.append(latex_main_results_table(pd.DataFrame(sentiment_summary)))
    if guidance_summary:
        paper_tables.append("\n% === TABLE 3: Guidance Results ===")
        paper_tables.append(latex_guidance_table(pd.DataFrame(guidance_summary)))

    if paper_tables:
        tables_path = run_dir / "paper_tables.tex"
        with open(tables_path, "w") as f:
            f.write("\n\n".join(paper_tables))
        print(f"\n[doc] LaTeX tables saved: {tables_path}")

    if make_plots:
        print(f"[chart] Figures saved: {fig_dir}")

    print(f"\n{'='*60}")
    print("All metrics computed and saved.")
    print(f"{'='*60}\n")


def compare_runs(run_ids: list[str]):
    """Compare results across multiple evaluation runs."""
    all_rows = []
    for run_id in run_ids:
        try:
            df = load_predictions(run_id)
            for model_key in df["model"].unique():
                sub = df[df["model"] == model_key]
                m = compute_sentiment_metrics(sub)
                if "error" not in m:
                    all_rows.append({
                        "run_id":    run_id,
                        "model":     model_key,
                        "shots":     sub["shots"].iloc[0] if "shots" in sub else "?",
                        "accuracy":  m["accuracy"],
                        "macro_f1":  m["macro_f1"],
                        "kappa":     m["kappa"],
                        "n":         m["n_samples"],
                    })
        except Exception as e:
            print(f"[!] Could not load {run_id}: {e}")

    if all_rows:
        comp_df = pd.DataFrame(all_rows).sort_values(["model","shots"])
        print("\nCross-run comparison:")
        print(comp_df.to_string(index=False))
        comp_df.to_csv(RESULTS_DIR / "cross_run_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Metrics Calculator")
    parser.add_argument("--run_id",  help="Evaluation run ID to analyze")
    parser.add_argument("--plots",   action="store_true", help="Generate figures")
    parser.add_argument("--compare", nargs="+", metavar="RUN_ID",
                        help="Compare multiple run IDs")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
    elif args.run_id:
        run_metrics(args.run_id, make_plots=args.plots)
    else:
        # List available runs
        runs = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
        if runs:
            print("Available runs:")
            for r in sorted(runs):
                print(f"  {r}")
        else:
            print("No evaluation runs found. Run llm_evaluator.py first.")


if __name__ == "__main__":
    main()
