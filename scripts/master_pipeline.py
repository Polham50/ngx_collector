"""
NGX-FND Master Pipeline
========================
Runs the full research pipeline end-to-end and assembles
all outputs into a single paper-ready package.

Pipeline stages:
  Stage 1 -- Data Collection       (ngx_collector.py)
  Stage 2 -- Text Cleaning         (text_cleaner.py)
  Stage 3 -- Corpus Validation     (corpus_validator.py)
  Stage 4 -- LLM Evaluation        (llm_evaluator.py)
  Stage 5 -- Metrics               (metrics.py)
  Stage 6 -- Price Data            (price_collector.py)
  Stage 7 -- Abnormal Returns      (abnormal_returns.py)
  Stage 8 -- Return Regression     (return_regression.py)
  Stage 9 -- Error Analysis        (error_analysis.py)
  Stage 10 -- Paper Assembly       (this script)

Usage:
  python master_pipeline.py --stage all          # full run
  python master_pipeline.py --stage eval         # stages 4-5 only
  python master_pipeline.py --stage returns      # stages 6-8 only
  python master_pipeline.py --stage errors       # stage 9 only
  python master_pipeline.py --stage assemble     # stage 10: build paper package
  python master_pipeline.py --status            # check what's been completed
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = BASE_DIR / "scripts"
DATA_DIR     = BASE_DIR / "data"
LOG_DIR      = BASE_DIR / "logs"
PAPER_DIR    = BASE_DIR / "paper_package"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Pipeline Stage Definitions ─────────────────────────────────────────────────
STAGES = {
    "collect": {
        "label":  "Stage 1: Data Collection",
        "script": "ngx_collector.py",
        "args":   ["--mode", "full"],
        "output_check": DATA_DIR / "metadata" / "collection_registry.json",
        "group": "data",
    },
    "clean": {
        "label":  "Stage 2: Text Cleaning",
        "script": "text_cleaner.py",
        "args":   [],
        "output_check": DATA_DIR / "metadata" / "annotation_queue.csv",
        "group": "data",
    },
    "validate": {
        "label":  "Stage 3: Corpus Validation",
        "script": "corpus_validator.py",
        "args":   ["--report"],
        "output_check": None,
        "group": "data",
    },
    "evaluate": {
        "label":  "Stage 4: LLM Evaluation",
        "script": "llm_evaluator.py",
        "args":   ["--task", "combined", "--shots", "5"],
        "output_check": None,   # run_id varies
        "group": "eval",
        "requires_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
    },
    "metrics": {
        "label":  "Stage 5: Metrics Computation",
        "script": "metrics.py",
        "args":   [],           # --run_id filled dynamically
        "output_check": None,
        "group": "eval",
    },
    "prices": {
        "label":  "Stage 6: Price Data Collection",
        "script": "price_collector.py",
        "args":   ["--full"],
        "output_check": DATA_DIR / "prices" / "log_returns.csv",
        "group": "returns",
    },
    "abnormal": {
        "label":  "Stage 7: Abnormal Returns",
        "script": "abnormal_returns.py",
        "args":   [],
        "output_check": DATA_DIR / "prices" / "abnormal_returns.csv",
        "group": "returns",
    },
    "regression": {
        "label":  "Stage 8: Return-Sentiment Regression",
        "script": "return_regression.py",
        "args":   ["--car", "CAR[0,+3]", "--plots"],
        "output_check": DATA_DIR / "returns_analysis" / "regression_results.csv",
        "group": "returns",
    },
    "errors": {
        "label":  "Stage 9: Error Analysis",
        "script": "error_analysis.py",
        "args":   ["--plots", "--export_examples"],
        "output_check": DATA_DIR / "error_analysis" / "taxonomy_summary.csv",
        "group": "errors",
    },
}

STAGE_GROUPS = {
    "all":     list(STAGES.keys()),
    "data":    ["collect", "clean", "validate"],
    "eval":    ["evaluate", "metrics"],
    "returns": ["prices", "abnormal", "regression"],
    "errors":  ["errors"],
    "assemble":[], # handled separately
}


# ── Status Checker ─────────────────────────────────────────────────────────────

def check_pipeline_status() -> dict:
    """Check which stages have been completed."""
    status = {}
    for stage_id, cfg in STAGES.items():
        check = cfg.get("output_check")
        if check is None:
            status[stage_id] = "unknown"
        elif check.exists():
            status[stage_id] = "complete"
        else:
            status[stage_id] = "pending"
    return status


def print_status():
    status = check_pipeline_status()
    print(f"\n{'='*55}")
    print("NGX-FND Pipeline Status")
    print(f"{'='*55}")
    for stage_id, cfg in STAGES.items():
        s = status[stage_id]
        icon = "[OK]" if s == "complete" else ("[?]" if s == "unknown" else "[...]")
        print(f"  {icon}  {cfg['label']}")

    # Check API keys
    print(f"\n  API Keys:")
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
        set_icon = "[OK]" if os.environ.get(key) else "[ERR]"
        print(f"    {set_icon}  {key}")

    # Count available data
    raw_pdf_count  = len(list((DATA_DIR / "raw_pdfs").rglob("*.pdf"))) if (DATA_DIR / "raw_pdfs").exists() else 0
    text_count     = len(list((DATA_DIR / "extracted_text").rglob("*.json"))) if (DATA_DIR / "extracted_text").exists() else 0
    ann_queue_path = DATA_DIR / "metadata" / "annotation_queue.csv"
    gold_path      = DATA_DIR / "metadata" / "gold_standard.csv"

    print(f"\n  Data inventory:")
    print(f"    PDFs downloaded    : {raw_pdf_count}")
    print(f"    Texts extracted    : {text_count}")
    if ann_queue_path.exists():
        import pandas as pd
        q = pd.read_csv(ann_queue_path)
        done = (q["annotation_status"] == "done").sum() if "annotation_status" in q else 0
        print(f"    Annotation queue   : {len(q)} passages ({done} annotated)")
    if gold_path.exists():
        import pandas as pd
        g = pd.read_csv(gold_path)
        print(f"    Gold standard      : {len(g)} passages")
    print(f"{'='*55}\n")


# ── Stage Runner ───────────────────────────────────────────────────────────────

def run_stage(stage_id: str, extra_args: list = None) -> bool:
    """Run a single pipeline stage. Returns True on success."""
    cfg    = STAGES.get(stage_id)
    if not cfg:
        log.error(f"Unknown stage: {stage_id}")
        return False

    log.info(f"\n{'='*55}")
    log.info(f"Running: {cfg['label']}")
    log.info(f"{'='*55}")

    # Check API keys
    for key in cfg.get("requires_keys", []):
        if not os.environ.get(key):
            log.warning(f"  {key} not set -- stage may fail or be skipped")

    script = SCRIPTS_DIR / cfg["script"]
    if not script.exists():
        log.error(f"Script not found: {script}")
        return False

    args = [sys.executable, str(script)] + cfg["args"] + (extra_args or [])
    log.info(f"  Running: {' '.join(args)}")

    result = subprocess.run(args, capture_output=False, cwd=str(SCRIPTS_DIR))
    if result.returncode != 0:
        log.error(f"  Stage FAILED (exit code {result.returncode})")
        return False

    log.info(f"  Stage complete [OK]")
    return True


def find_latest_run_id() -> str | None:
    """Find the most recent LLM evaluation run ID."""
    results_dir = DATA_DIR / "llm_results"
    if not results_dir.exists():
        return None
    runs = sorted(
        [d.name for d in results_dir.iterdir()
         if d.is_dir() and (d / "predictions.csv").exists()],
        reverse=True
    )
    return runs[0] if runs else None


# ── Paper Package Assembler ────────────────────────────────────────────────────

def assemble_paper_package():
    """
    Collect all paper-ready outputs into a single organised directory.

    Structure:
      paper_package/
        tables/           <- all LaTeX .tex table files
        figures/          <- all PDF figures
        data/             <- key CSV files for replication
        section_4_4.tex   <- error analysis section draft
        paper_tables.tex  <- combined tables file
        README.md         <- replication guide
    """
    log.info(f"\n{'='*55}")
    log.info("Assembling Paper Package")
    log.info(f"{'='*55}")

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir  = PAPER_DIR / "tables"
    figures_dir = PAPER_DIR / "figures"
    replication_dir = PAPER_DIR / "replication_data"
    for d in [tables_dir, figures_dir, replication_dir]:
        d.mkdir(exist_ok=True)

    copied = []
    missing = []

    def try_copy(src: Path, dest: Path, label: str):
        if src.exists():
            shutil.copy2(src, dest)
            copied.append(label)
            log.info(f"  [OK] {label}")
        else:
            missing.append(label)
            log.warning(f"  [ERR] {label} -- not found ({src})")

    # ── LaTeX tables ──
    run_id = find_latest_run_id()
    if run_id:
        try_copy(DATA_DIR / "llm_results" / run_id / "paper_tables.tex",
                 tables_dir / "table_2_3_llm_results.tex",
                 "Tables 2-3: LLM Evaluation Results")

    try_copy(DATA_DIR / "returns_analysis" / "paper_tables.tex",
             tables_dir / "table_4_5_regression.tex",
             "Tables 4-5: Return Regression Results")

    try_copy(DATA_DIR / "error_analysis" / "paper_section_4_4.tex",
             tables_dir / "section_4_4_error_analysis.tex",
             "Section 4.4: Error Analysis Draft")

    # ── Figures ──
    fig_sources = [
        (DATA_DIR / "figures", "llm_eval_figures"),
        (DATA_DIR / "figures" / "error_analysis", "error_analysis_figures"),
    ]
    for src_dir, label in fig_sources:
        if src_dir.exists():
            for pdf in src_dir.glob("*.pdf"):
                dest = figures_dir / pdf.name
                shutil.copy2(pdf, dest)
                copied.append(f"Figure: {pdf.name}")

    # ── Replication data ──
    replication_files = [
        (DATA_DIR / "metadata" / "corpus_index.csv",          "corpus_index.csv"),
        (DATA_DIR / "metadata" / "gold_standard.csv",         "gold_standard.csv"),
        (DATA_DIR / "prices"   / "abnormal_returns.csv",      "abnormal_returns.csv"),
        (DATA_DIR / "returns_analysis" / "regression_results.csv", "regression_results.csv"),
        (DATA_DIR / "returns_analysis" / "group_tests.csv",    "group_tests.csv"),
        (DATA_DIR / "error_analysis"   / "taxonomy_summary.csv","taxonomy_summary.csv"),
        (DATA_DIR / "error_analysis"   / "qualitative_examples.json","qualitative_examples.json"),
        (DATA_DIR / "companies.json",                          "companies.json"),
    ]
    if run_id:
        replication_files += [
            (DATA_DIR / "llm_results" / run_id / "predictions.csv",        "llm_predictions.csv"),
            (DATA_DIR / "llm_results" / run_id / "sentiment_metrics.csv",  "sentiment_metrics.csv"),
            (DATA_DIR / "llm_results" / run_id / "guidance_metrics.csv",   "guidance_metrics.csv"),
        ]
    for src, dest_name in replication_files:
        try_copy(src, replication_dir / dest_name, f"Data: {dest_name}")

    # ── Combine all LaTeX tables ──
    all_tex = []
    for tex_file in sorted(tables_dir.glob("*.tex")):
        all_tex.append(f"% === {tex_file.name} ===")
        all_tex.append(tex_file.read_text(encoding="utf-8"))

    if all_tex:
        combined_tex = PAPER_DIR / "all_paper_outputs.tex"
        combined_tex.write_text("\n\n".join(all_tex), encoding="utf-8")
        log.info(f"\n  Combined LaTeX: {combined_tex}")

    # ── README ──
    readme = _generate_package_readme(run_id, copied, missing)
    (PAPER_DIR / "README.md").write_text(readme, encoding="utf-8")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"Paper Package Summary")
    print(f"{'='*55}")
    print(f"  Output directory : {PAPER_DIR}")
    print(f"  Files copied     : {len(copied)}")
    print(f"  Files missing    : {len(missing)}")
    if missing:
        print(f"\n  Missing (run remaining stages):")
        for m in missing:
            print(f"    [ERR] {m}")
    print(f"{'='*55}\n")


def _generate_package_readme(run_id: str | None, copied: list, missing: list) -> str:
    return f"""# NGX-FND Paper Package
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Paper
**Title:** Extracting Sentiment and Forward Guidance Signals from Nigerian Exchange
Corporate Disclosures Using Large Language Models

**Authors:** Timothy Popoola Oladapo (Dataqury Analytics / WorldQuant University)

**Target venues:** FinNLP @ EMNLP 2026 | AfricaNLP @ ACL 2026

---

## Directory Structure

```
paper_package/
├── tables/
│   ├── table_2_3_llm_results.tex      <- Main evaluation results (Tables 2-3)
│   ├── table_4_5_regression.tex       <- Return regression results (Tables 4-5)
│   └── section_4_4_error_analysis.tex <- Error analysis section draft
├── figures/
│   ├── f1_comparison.pdf              <- Model comparison bar chart
│   ├── confusion_matrices_*.pdf       <- Per-model confusion matrices
│   ├── taxonomy_distribution.pdf      <- Failure mode taxonomy chart
│   ├── error_heatmap_*.pdf            <- Error rate heatmap
│   └── ...
├── replication_data/
│   ├── corpus_index.csv               <- NGX-FND corpus index
│   ├── gold_standard.csv              <- Human-annotated passages
│   ├── llm_predictions.csv            <- All model predictions
│   ├── sentiment_metrics.csv          <- Per-model metrics
│   ├── abnormal_returns.csv           <- Event study results
│   ├── regression_results.csv         <- Regression coefficients
│   └── taxonomy_summary.csv           <- Error taxonomy counts
├── all_paper_outputs.tex              <- All LaTeX tables combined
└── README.md                          <- This file
```

---

## Replication

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."

# 3. Run full pipeline
python scripts/master_pipeline.py --stage all

# 4. Assemble paper outputs
python scripts/master_pipeline.py --stage assemble
```

---

## LLM Evaluation Run
Run ID: `{run_id or 'not yet completed'}`

---

## Files Included: {len(copied)}
## Files Missing: {len(missing)}
{chr(10).join(f'- [ERR] {m}' for m in missing) if missing else '*(All files present)*'}
"""


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NGX-FND Master Pipeline")
    parser.add_argument("--stage", default="status",
                        choices=list(STAGES.keys()) + list(STAGE_GROUPS.keys()) + ["assemble","status"],
                        help="Pipeline stage(s) to run")
    parser.add_argument("--run_id", help="LLM evaluation run ID (for metrics stage)")
    args = parser.parse_args()

    if args.stage == "status":
        print_status()
        return

    if args.stage == "assemble":
        assemble_paper_package()
        return

    # Resolve stages to run
    stages_to_run = STAGE_GROUPS.get(args.stage, [args.stage])

    log.info(f"\nNGX-FND Pipeline -- Running: {args.stage}")
    log.info(f"Stages: {stages_to_run}")

    results = {}
    for stage_id in stages_to_run:
        extra = []
        if stage_id == "metrics":
            run_id = args.run_id or find_latest_run_id()
            if run_id:
                extra = ["--run_id", run_id, "--plots"]
            else:
                log.warning("No run_id found for metrics stage -- skipping")
                continue
        success = run_stage(stage_id, extra_args=extra)
        results[stage_id] = "[OK]" if success else "[ERR]"

    print(f"\n{'='*55}")
    print("Pipeline Run Summary")
    print(f"{'='*55}")
    for stage_id, result in results.items():
        print(f"  {result}  {STAGES[stage_id]['label']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
