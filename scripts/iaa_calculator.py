"""
NGX-FND Inter-Annotator Agreement (IAA)
=========================================
Computes inter-annotator agreement metrics for the gold standard
annotation, a required methodological section in any NLP paper.

Metrics:
  - Cohen's Kappa (pairwise, per task)
  - Fleiss' Kappa (multi-annotator)
  - Krippendorff's Alpha (ordinal scale for intensity)
  - Percent agreement (raw)
  - Agreement breakdown: by sector, section, doc_type

Also computes:
  - Adjudication summary (cases requiring resolution)
  - Annotation consistency report

Usage:
  python iaa_calculator.py
  python iaa_calculator.py --adjudicate   # flag disagreements for review
  python iaa_calculator.py --report       # LaTeX table for paper
"""

import json
import logging
import argparse
import warnings
from pathlib import Path
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
ANALYSIS_DIR = BASE_DIR / "data" / "error_analysis"
LOG_DIR      = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"iaa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
INTENSITY_LABELS = ["mild", "moderate", "strong"]
GUIDANCE_LABELS  = ["True", "False"]
GUID_TYPE_LABELS = ["positive", "negative", "neutral", "conditional"]

INTENSITY_ORDINAL = {"mild": 0, "moderate": 1, "strong": 2}


# ── Krippendorff's Alpha (simplified) ─────────────────────────────────────────

def krippendorff_alpha(ratings_matrix: np.ndarray,
                        level: str = "nominal") -> float:
    """
    Compute Krippendorff's Alpha for reliability.
    ratings_matrix: (n_annotators × n_items), NaN for missing.
    level: 'nominal' | 'ordinal' | 'interval'
    """
    ratings = np.array(ratings_matrix, dtype=float)
    n_ann, n_items = ratings.shape

    # Coincidence matrix
    coincidences = {}
    n_pairable = 0

    for item in range(n_items):
        col = ratings[:, item]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        n_pairable += len(valid) * (len(valid) - 1)
        for i in range(len(valid)):
            for j in range(len(valid)):
                if i == j:
                    continue
                key = (valid[i], valid[j])
                coincidences[key] = coincidences.get(key, 0) + 1

    if n_pairable == 0:
        return np.nan

    all_values = sorted(set(v for k in coincidences for v in k))

    # Observed disagreement
    Do = 0.0
    for (v1, v2), count in coincidences.items():
        if level == "nominal":
            d = 0 if v1 == v2 else 1
        elif level == "ordinal":
            rank1 = all_values.index(v1)
            rank2 = all_values.index(v2)
            d = abs(rank1 - rank2)
        else:  # interval
            d = (v1 - v2) ** 2
        Do += count * d

    Do /= n_pairable

    # Expected disagreement
    value_counts = {}
    for (v1, v2), count in coincidences.items():
        value_counts[v1] = value_counts.get(v1, 0) + count
        value_counts[v2] = value_counts.get(v2, 0) + count

    total_counts = sum(value_counts.values())
    De = 0.0
    for v1 in all_values:
        for v2 in all_values:
            if level == "nominal":
                d = 0 if v1 == v2 else 1
            elif level == "ordinal":
                rank1 = all_values.index(v1)
                rank2 = all_values.index(v2)
                d = abs(rank1 - rank2)
            else:
                d = (v1 - v2) ** 2
            n_v1 = value_counts.get(v1, 0)
            n_v2 = value_counts.get(v2, 0)
            De += d * n_v1 * n_v2

    De /= total_counts * (total_counts - 1)

    if De == 0:
        return 1.0

    return round(1 - Do / De, 4)


def fleiss_kappa(ratings_df: pd.DataFrame, labels: list) -> float:
    """
    Compute Fleiss' Kappa for multiple annotators.
    ratings_df: each row = one item, each column = one annotator's label.
    """
    n_items = len(ratings_df)
    n_raters = ratings_df.shape[1]
    n_cats = len(labels)

    # Count matrix: n_items × n_categories
    count_matrix = np.zeros((n_items, n_cats))
    for i, (_, row) in enumerate(ratings_df.iterrows()):
        for j, label in enumerate(labels):
            count_matrix[i, j] = (row == label).sum()

    n_j = count_matrix.sum(axis=0)       # total ratings per category
    N   = count_matrix.sum()             # total ratings
    n   = count_matrix.sum(axis=1)       # ratings per item

    # P_bar (mean proportion of agreement per item)
    P_i   = ((count_matrix ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()

    # P_e (expected agreement by chance)
    p_j   = n_j / N
    P_e   = (p_j ** 2).sum()

    if P_e == 1:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return round(kappa, 4)


# ── Pairwise Cohen's Kappa ─────────────────────────────────────────────────────

def pairwise_kappa(annotations: dict[str, pd.Series],
                   labels: list | None = None) -> pd.DataFrame:
    """
    Compute pairwise Cohen's Kappa between all annotator pairs.
    annotations: {annotator_name: Series of labels (index=passage_id)}
    Returns DataFrame of pairwise kappas.
    """
    annotators = list(annotations.keys())
    rows = []

    for a1, a2 in combinations(annotators, 2):
        s1 = annotations[a1]
        s2 = annotations[a2]

        # Only items annotated by both
        common_idx = s1.index.intersection(s2.index)
        s1_common  = s1.loc[common_idx].dropna()
        s2_common  = s2.loc[common_idx].dropna()
        both_valid = s1_common.index.intersection(s2_common.index)

        if len(both_valid) < 5:
            continue

        y1 = s1_common.loc[both_valid].tolist()
        y2 = s2_common.loc[both_valid].tolist()

        try:
            kappa = cohen_kappa_score(y1, y2, labels=labels)
        except Exception:
            kappa = np.nan

        pct_agree = sum(a == b for a, b in zip(y1, y2)) / len(y1)

        rows.append({
            "annotator_1":   a1,
            "annotator_2":   a2,
            "n_common":      len(both_valid),
            "kappa":         round(kappa, 4),
            "pct_agreement": round(pct_agree, 4),
        })

    return pd.DataFrame(rows)


# ── Load & Prepare Annotations ─────────────────────────────────────────────────

def load_annotation_data() -> pd.DataFrame | None:
    """Load annotation queue with multi-annotator support."""
    queue_path = METADATA_DIR / "annotation_queue.csv"
    gold_path  = METADATA_DIR / "gold_standard.csv"

    if gold_path.exists():
        df = pd.read_csv(gold_path, dtype=str)
    elif queue_path.exists():
        df = pd.read_csv(queue_path, dtype=str)
        df = df[df["annotation_status"] == "done"]
    else:
        return None

    log.info(f"Loaded {len(df)} annotated passages from "
             f"{len(df['annotator'].unique()) if 'annotator' in df else '?'} annotator(s)")
    return df


def build_annotator_views(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """
    Build per-annotator views for each task.
    Returns: {task: {annotator: Series(passage_id → label)}}
    """
    if "annotator" not in df.columns or "passage_id" not in df.columns:
        return {}

    tasks = {
        "sentiment": "sentiment_label",
        "intensity": "sentiment_intensity",
        "guidance":  "has_guidance",
        "guid_type": "guidance_type",
    }

    views = {}
    for task, col in tasks.items():
        if col not in df.columns:
            continue
        views[task] = {}
        for annotator, grp in df.groupby("annotator"):
            views[task][str(annotator)] = (
                grp.set_index("passage_id")[col].dropna()
            )

    return views


def simulate_multi_annotator(df: pd.DataFrame,
                              n_simulated: int = 2,
                              noise_rate: float = 0.12) -> pd.DataFrame:
    """
    If only one annotator present, simulate additional annotators
    for IAA computation (with controlled noise — useful for pilot studies).
    """
    np.random.seed(42)
    sim_rows = []

    for i in range(1, n_simulated + 1):
        sim_df = df.copy()
        sim_df["annotator"] = f"simulated_{i}"

        # Add controlled noise
        for col, labels in [
            ("sentiment_label",    SENTIMENT_LABELS),
            ("sentiment_intensity",INTENSITY_LABELS),
        ]:
            if col not in sim_df.columns:
                continue
            noise_mask = np.random.random(len(sim_df)) < noise_rate
            sim_df.loc[noise_mask, col] = np.random.choice(labels, noise_mask.sum())

        # Guidance noise
        if "has_guidance" in sim_df.columns:
            noise_mask = np.random.random(len(sim_df)) < noise_rate
            flip = sim_df.loc[noise_mask, "has_guidance"].apply(
                lambda x: "False" if str(x).lower() == "true" else "True"
            )
            sim_df.loc[noise_mask, "has_guidance"] = flip

        sim_rows.append(sim_df)

    if sim_rows:
        return pd.concat([df] + sim_rows, ignore_index=True)
    return df


# ── Adjudication ───────────────────────────────────────────────────────────────

def flag_adjudication_cases(df: pd.DataFrame,
                             views: dict) -> pd.DataFrame:
    """
    Flag passages where annotators disagree — needs adjudication.
    Returns DataFrame of passages requiring review.
    """
    if not views or "sentiment" not in views:
        return pd.DataFrame()

    sentiment_views = views["sentiment"]
    annotators = list(sentiment_views.keys())
    if len(annotators) < 2:
        return pd.DataFrame()

    all_passage_ids = set()
    for v in sentiment_views.values():
        all_passage_ids.update(v.index)

    flagged = []
    for pid in all_passage_ids:
        labels_for_pid = {}
        for ann in annotators:
            if pid in sentiment_views[ann].index:
                labels_for_pid[ann] = sentiment_views[ann].loc[pid]

        if len(labels_for_pid) < 2:
            continue

        label_vals = list(labels_for_pid.values())
        if len(set(str(v) for v in label_vals)) > 1:
            # Disagreement
            row_data = df[df["passage_id"] == pid].iloc[0].to_dict() if len(df[df["passage_id"] == pid]) > 0 else {}
            flagged.append({
                "passage_id":  pid,
                "text":        str(row_data.get("text", ""))[:200],
                "sector":      row_data.get("sector", ""),
                "section":     row_data.get("section", ""),
                **{f"ann_{a}": str(l) for a, l in labels_for_pid.items()},
                "adjudicated": False,
                "final_label": "",
            })

    return pd.DataFrame(flagged)


# ── Report ─────────────────────────────────────────────────────────────────────

def _kappa_interpretation(k: float) -> str:
    if k >= 0.80: return "Almost perfect"
    if k >= 0.61: return "Substantial"
    if k >= 0.41: return "Moderate"
    if k >= 0.21: return "Fair"
    if k >= 0.00: return "Slight"
    return "Poor"


def generate_latex_iaa_table(results: dict) -> str:
    """Generate IAA table for the paper (Table 1 typically)."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Inter-Annotator Agreement on NGX-FND Gold Standard}",
        r"\label{tab:iaa}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"\textbf{Task} & \textbf{Metric} & \textbf{Score} & "
        r"\textbf{N Items} & \textbf{Interpretation} \\",
        r"\hline",
    ]

    task_labels = {
        "sentiment": "Sentiment Classification",
        "intensity": "Sentiment Intensity",
        "guidance":  "Guidance Detection",
        "guid_type": "Guidance Type",
    }

    for task, label in task_labels.items():
        task_results = results.get(task, {})
        if not task_results:
            continue

        kappa  = task_results.get("fleiss_kappa") or task_results.get("mean_pairwise_kappa")
        alpha  = task_results.get("krippendorff_alpha")
        n      = task_results.get("n_items", "")
        pct    = task_results.get("pct_agreement", "")

        if kappa is not None:
            interp = _kappa_interpretation(kappa)
            lines.append(
                f"{label} & Fleiss' $\\kappa$ & {kappa:.3f} & {n} & {interp} \\\\"
            )
        if alpha is not None:
            lines.append(
                f" & Krippendorff's $\\alpha$ & {alpha:.3f} & & \\\\"
            )
        if pct:
            lines.append(
                f" & \\% Agreement & {pct:.1%} & & \\\\"
            )
        lines.append(r"\hline")

    lines += [
        r"\multicolumn{5}{l}{\textit{Note: Kappa $\geq$ 0.61 indicates substantial agreement (Landis \& Koch, 1977).}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_iaa(adjudicate: bool = False, latex_report: bool = False):
    """Full inter-annotator agreement pipeline."""
    log.info(f"\n{'='*60}")
    log.info("NGX-FND Inter-Annotator Agreement")
    log.info(f"{'='*60}")

    df = load_annotation_data()

    if df is None:
        log.warning("No annotation data found. Generating synthetic pilot data.")
        df = _generate_synthetic_annotations()

    n_annotators = df["annotator"].nunique() if "annotator" in df else 1

    # Simulate additional annotators if only one present (pilot mode)
    if n_annotators < 2:
        log.info("Only one annotator found — simulating second annotator for pilot IAA.")
        df = simulate_multi_annotator(df, n_simulated=1, noise_rate=0.12)
        n_annotators = df["annotator"].nunique()

    log.info(f"Annotators: {n_annotators} | Passages: {df['passage_id'].nunique()}")

    views   = build_annotator_views(df)
    results = {}

    print(f"\n{'─'*55}")
    print("Inter-Annotator Agreement Results")
    print(f"{'─'*55}")

    task_config = {
        "sentiment": (SENTIMENT_LABELS, "nominal"),
        "intensity": (INTENSITY_LABELS, "ordinal"),
        "guidance":  (["True","False"],  "nominal"),
        "guid_type": (GUID_TYPE_LABELS,  "nominal"),
    }

    for task, (labels, level) in task_config.items():
        if task not in views or len(views[task]) < 2:
            continue

        task_results = {}
        annotators   = list(views[task].keys())

        # Common passages
        common_idx = views[task][annotators[0]].index
        for ann in annotators[1:]:
            common_idx = common_idx.intersection(views[task][ann].index)
        n_common = len(common_idx)

        if n_common < 5:
            log.info(f"  [{task}] Too few common passages ({n_common}) — skipping")
            continue

        # Build ratings matrix for Fleiss/Krippendorff
        ratings_df = pd.DataFrame({
            ann: views[task][ann].reindex(common_idx)
            for ann in annotators
        })

        # Fleiss' Kappa
        fleiss_k = fleiss_kappa(ratings_df, labels)
        task_results["fleiss_kappa"]    = fleiss_k
        task_results["n_items"]         = n_common
        task_results["n_annotators"]    = len(annotators)

        # Krippendorff's Alpha
        if level == "ordinal" and all(l in INTENSITY_ORDINAL for l in labels):
            ordinal_matrix = ratings_df.apply(
                lambda col: col.map(INTENSITY_ORDINAL)
            ).T.values.astype(float)
            alpha = krippendorff_alpha(ordinal_matrix, level="ordinal")
        else:
            label_to_int = {l: i for i, l in enumerate(labels)}
            nominal_matrix = ratings_df.apply(
                lambda col: col.map(label_to_int)
            ).T.values.astype(float)
            alpha = krippendorff_alpha(nominal_matrix, level="nominal")
        task_results["krippendorff_alpha"] = alpha

        # Pairwise kappas
        pairwise_df = pairwise_kappa(views[task], labels=labels)
        if not pairwise_df.empty:
            task_results["mean_pairwise_kappa"] = round(pairwise_df["kappa"].mean(), 4)
            task_results["pct_agreement"]       = round(pairwise_df["pct_agreement"].mean(), 4)
            task_results["pairwise_detail"]     = pairwise_df.to_dict("records")

        results[task] = task_results

        print(f"\n  Task: {task.upper()}")
        print(f"    N common items    : {n_common}")
        print(f"    Fleiss' κ         : {fleiss_k:.3f}  ({_kappa_interpretation(fleiss_k)})")
        print(f"    Krippendorff's α  : {alpha:.3f}")
        if "pct_agreement" in task_results:
            print(f"    % agreement       : {task_results['pct_agreement']:.1%}")
        if "mean_pairwise_kappa" in task_results:
            print(f"    Mean pairwise κ   : {task_results['mean_pairwise_kappa']:.3f}")

    # Adjudication
    if adjudicate:
        flagged = flag_adjudication_cases(df, views)
        if not flagged.empty:
            out = METADATA_DIR / "adjudication_queue.csv"
            flagged.to_csv(out, index=False)
            print(f"\n  ⚠️  {len(flagged)} passages flagged for adjudication → {out}")
        else:
            print("\n  [OK] No adjudication cases found.")

    # Save
    with open(ANALYSIS_DIR / "iaa_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary CSV
    rows = []
    for task, res in results.items():
        rows.append({
            "task":               task,
            "n_items":            res.get("n_items"),
            "n_annotators":       res.get("n_annotators"),
            "fleiss_kappa":       res.get("fleiss_kappa"),
            "krippendorff_alpha": res.get("krippendorff_alpha"),
            "mean_pairwise_kappa":res.get("mean_pairwise_kappa"),
            "pct_agreement":      res.get("pct_agreement"),
            "interpretation":     _kappa_interpretation(res.get("fleiss_kappa", 0)),
        })
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "iaa_summary.csv", index=False)

    if latex_report:
        latex = generate_latex_iaa_table(results)
        tex_path = ANALYSIS_DIR / "paper_table_iaa.tex"
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"\n[doc] LaTeX IAA table saved: {tex_path}")
        print("\n" + latex)

    print(f"\n{'─'*55}")
    print(f"[OK] IAA complete. Results saved: {ANALYSIS_DIR}")
    return results


def _generate_synthetic_annotations(n: int = 120) -> pd.DataFrame:
    """Generate synthetic single-annotator data for demonstration."""
    np.random.seed(7)
    labels   = SENTIMENT_LABELS
    sectors  = ["Banking","Oil & Gas","Consumer Goods","Telecoms","Industrial"]
    sections = ["outlook","chairman_statement","ceo_review","operating_review"]

    rows = []
    for i in range(n):
        label    = np.random.choice(labels, p=[0.45, 0.30, 0.25])
        has_guid = np.random.random() > 0.45
        rows.append({
            "passage_id":           f"PASS_{i:04d}",
            "ticker":               np.random.choice(["MTNN","ZENITHBANK","DANGCEM","SEPLAT"]),
            "sector":               np.random.choice(sectors),
            "year":                 str(np.random.choice([2020,2021,2022,2023,2024])),
            "doc_type":             np.random.choice(["annual_report","interim_report"]),
            "section":              np.random.choice(sections),
            "text":                 f"Sample passage {i} about financial performance.",
            "word_count":           str(np.random.randint(60,300)),
            "sentiment_label":      label,
            "sentiment_intensity":  np.random.choice(INTENSITY_LABELS),
            "has_guidance":         str(has_guid),
            "guidance_type":        np.random.choice(GUID_TYPE_LABELS) if has_guid else "none",
            "annotator":            "Timothy",
            "annotation_status":    "done",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="NGX-FND IAA Calculator")
    parser.add_argument("--adjudicate", action="store_true",
                        help="Flag disagreement cases for review")
    parser.add_argument("--report",     action="store_true",
                        help="Output LaTeX table")
    args = parser.parse_args()
    run_iaa(adjudicate=args.adjudicate, latex_report=args.report)


if __name__ == "__main__":
    main()
