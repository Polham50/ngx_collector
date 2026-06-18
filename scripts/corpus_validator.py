"""
NGX-FND Corpus Validator
=========================
Validates the collected corpus for research readiness:
  - Coverage checks (sectors, years, document types)
  - Quality thresholds
  - Annotation readiness report
  - Corpus statistics for the paper's data section

Usage:
  python corpus_validator.py
  python corpus_validator.py --report   # Generate full LaTeX-ready stats table
"""

import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
CLEANED_DIR  = BASE_DIR / "data" / "cleaned_text"

# ── Thresholds (targets for publication-ready corpus) ─────────────────────────
TARGETS = {
    "min_companies":          30,
    "min_documents":         150,
    "min_good_passages":     300,
    "min_sectors":             4,
    "min_years":               3,
    "min_docs_per_sector":    20,
    "min_annotated_passages": 150,
}


# ── Validators ─────────────────────────────────────────────────────────────────

def load_corpus_index() -> pd.DataFrame | None:
    path = METADATA_DIR / "corpus_index.csv"
    if not path.exists():
        print("[!] corpus_index.csv not found — run ngx_collector.py --mode extract first")
        return None
    return pd.read_csv(path)


def load_annotation_queue() -> pd.DataFrame | None:
    path = METADATA_DIR / "annotation_queue.csv"
    if not path.exists():
        print("[!] annotation_queue.csv not found — run text_cleaner.py first")
        return None
    return pd.read_csv(path)


def load_collection_registry() -> dict | None:
    path = METADATA_DIR / "collection_registry.json"
    if not path.exists():
        print("[!] collection_registry.json not found — run ngx_collector.py --mode discover first")
        return None
    with open(path) as f:
        return json.load(f)


def validate_coverage(corpus_df: pd.DataFrame) -> dict:
    """Check sector, year, and document type coverage."""
    results = {}

    companies    = corpus_df["ticker"].nunique() if "ticker" in corpus_df else 0
    sectors      = corpus_df["sector"].nunique() if "sector" in corpus_df else 0
    years        = corpus_df["year"].nunique()   if "year"   in corpus_df else 0
    total_docs   = len(corpus_df)

    results["companies"]  = companies
    results["sectors"]    = sectors
    results["years"]      = years
    results["total_docs"] = total_docs

    results["coverage_ok"] = (
        companies  >= TARGETS["min_companies"] and
        total_docs >= TARGETS["min_documents"] and
        sectors    >= TARGETS["min_sectors"]   and
        years      >= TARGETS["min_years"]
    )

    if "sector" in corpus_df:
        results["docs_per_sector"] = corpus_df.groupby("sector").size().to_dict()
        results["sector_ok"] = all(
            v >= TARGETS["min_docs_per_sector"]
            for v in results["docs_per_sector"].values()
        )
    else:
        results["docs_per_sector"] = {}
        results["sector_ok"] = False

    if "year" in corpus_df:
        results["docs_per_year"] = corpus_df.groupby("year").size().to_dict()

    if "doc_type" in corpus_df:
        results["docs_per_type"] = corpus_df.groupby("doc_type").size().to_dict()

    return results


def validate_text_quality(passages_df: pd.DataFrame) -> dict:
    """Check passage quality distribution."""
    results = {}

    total     = len(passages_df)
    good      = (passages_df["quality"] == "good").sum()      if "quality" in passages_df else 0
    acceptable= (passages_df["quality"] == "acceptable").sum() if "quality" in passages_df else 0
    poor      = (passages_df["quality"] == "poor").sum()       if "quality" in passages_df else 0
    annotated = (passages_df["annotation_status"] == "done").sum() if "annotation_status" in passages_df else 0

    results.update({
        "total_passages":      total,
        "good_passages":       good,
        "acceptable_passages": acceptable,
        "poor_passages":       poor,
        "annotated_passages":  annotated,
        "quality_ratio":       round((good + acceptable) / total * 100, 1) if total else 0,
        "annotation_progress": round(annotated / TARGETS["min_annotated_passages"] * 100, 1),
    })

    results["quality_ok"] = (good + acceptable) >= TARGETS["min_good_passages"]
    results["annotation_ok"] = annotated >= TARGETS["min_annotated_passages"]

    if "section" in passages_df:
        results["passages_per_section"] = passages_df.groupby("section").size().to_dict()

    if "sector" in passages_df:
        results["passages_per_sector"] = passages_df.groupby("sector")[
            ["passage_id"]
        ].count().rename(columns={"passage_id":"count"}).to_dict()["count"] if "passage_id" in passages_df else {}

    if "word_count" in passages_df:
        results["avg_word_count"] = round(passages_df["word_count"].mean(), 1)
        results["median_word_count"] = passages_df["word_count"].median()

    return results


def check_missing_gaps(corpus_df: pd.DataFrame) -> list[str]:
    """Identify notable gaps in the corpus."""
    gaps = []
    required_sectors = {"Banking", "Oil & Gas", "Consumer Goods", "Telecoms", "Industrial"}
    target_years     = {2020, 2021, 2022, 2023, 2024}

    if "sector" in corpus_df:
        present_sectors = set(corpus_df["sector"].unique())
        missing_sectors = required_sectors - present_sectors
        if missing_sectors:
            gaps.append(f"Missing sectors: {missing_sectors}")

        for sector in present_sectors:
            count = (corpus_df["sector"] == sector).sum()
            if count < TARGETS["min_docs_per_sector"]:
                gaps.append(f"Low coverage in {sector}: only {count} docs (target: {TARGETS['min_docs_per_sector']})")

    if "year" in corpus_df:
        present_years = set(corpus_df["year"].dropna().astype(int))
        missing_years = target_years - present_years
        if missing_years:
            gaps.append(f"Missing years: {sorted(missing_years)}")

    if "ticker" in corpus_df:
        single_doc_companies = corpus_df.groupby("ticker").size()
        few = single_doc_companies[single_doc_companies == 1].index.tolist()
        if len(few) > 5:
            gaps.append(f"{len(few)} companies with only 1 document (ideally 3+ per company for longitudinal analysis)")

    return gaps


def generate_paper_stats_table(coverage: dict, quality: dict) -> str:
    """Generate a LaTeX-ready statistics table for the paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{NGX-FND Corpus Statistics}",
        r"\label{tab:corpus_stats}",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"\textbf{Statistic} & \textbf{Value} \\",
        r"\hline",
        f"Companies covered & {coverage.get('companies', 0)} \\\\",
        f"Total documents & {coverage.get('total_docs', 0)} \\\\",
        f"Sectors & {coverage.get('sectors', 0)} \\\\",
        f"Years covered & {coverage.get('years', 0)} (2020--2024) \\\\",
        f"Total passages & {quality.get('total_passages', 0)} \\\\",
        f"High-quality passages & {quality.get('good_passages', 0)} \\\\",
        f"Annotated passages (gold) & {quality.get('annotated_passages', 0)} \\\\",
        f"Avg. passage length (words) & {quality.get('avg_word_count', 0)} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_validation_report(coverage: dict, quality: dict,
                             gaps: list, registry: dict | None,
                             latex: bool = False):
    """Print a human-readable validation report."""
    PASS = "[OK]"
    FAIL = "[ERR]"
    WARN = "[!] "

    print("\n" + "="*60)
    print("NGX-FND CORPUS VALIDATION REPORT")
    print("="*60)

    # Collection progress
    if registry:
        print("\n[+] Collection Progress")
        print(f"   Discovered : {len(registry.get('discovered', []))}")
        print(f"   Downloaded : {len(registry.get('downloaded', []))}")
        print(f"   Extracted  : {len(registry.get('extracted', []))}")
        print(f"   Failed     : {len(registry.get('failed', []))}")

    # Coverage checks
    print("\n[~] Coverage")
    c = coverage
    print(f"   {PASS if c.get('companies',0) >= TARGETS['min_companies'] else FAIL} "
          f"Companies: {c.get('companies',0)} / {TARGETS['min_companies']} target")
    print(f"   {PASS if c.get('total_docs',0) >= TARGETS['min_documents'] else FAIL} "
          f"Documents: {c.get('total_docs',0)} / {TARGETS['min_documents']} target")
    print(f"   {PASS if c.get('sectors',0) >= TARGETS['min_sectors'] else FAIL} "
          f"Sectors: {c.get('sectors',0)} / {TARGETS['min_sectors']} target")
    print(f"   {PASS if c.get('years',0) >= TARGETS['min_years'] else FAIL} "
          f"Years: {c.get('years',0)} / {TARGETS['min_years']} target")

    if "docs_per_sector" in c:
        print("\n   Docs by sector:")
        for sector, count in sorted(c["docs_per_sector"].items()):
            icon = PASS if count >= TARGETS["min_docs_per_sector"] else WARN
            print(f"     {icon} {sector:<20} {count}")

    if "docs_per_year" in c:
        print("\n   Docs by year:")
        for year, count in sorted(c["docs_per_year"].items()):
            print(f"      {year}: {count}")

    if "docs_per_type" in c:
        print("\n   Docs by type:")
        for dtype, count in sorted(c["docs_per_type"].items()):
            print(f"      {dtype:<25} {count}")

    # Quality
    print("\n[*] Text Quality")
    q = quality
    print(f"   Total passages     : {q.get('total_passages', 0)}")
    print(f"   Good quality       : {q.get('good_passages', 0)}")
    print(f"   Acceptable quality : {q.get('acceptable_passages', 0)}")
    print(f"   Poor quality       : {q.get('poor_passages', 0)}")
    print(f"   Quality ratio      : {q.get('quality_ratio', 0)}%")
    print(f"   Avg passage length : {q.get('avg_word_count', 0)} words")

    print(f"\n   {PASS if q.get('quality_ok') else FAIL} "
          f"Passage quality: {q.get('good_passages',0) + q.get('acceptable_passages',0)} usable "
          f"/ {TARGETS['min_good_passages']} target")

    # Annotation
    print(f"\n[-] Annotation Progress")
    ann = q.get("annotated_passages", 0)
    print(f"   {PASS if q.get('annotation_ok') else WARN} "
          f"Annotated: {ann} / {TARGETS['min_annotated_passages']} target "
          f"({q.get('annotation_progress',0)}%)")

    # Gaps
    if gaps:
        print("\n[!] Identified Gaps")
        for gap in gaps:
            print(f"   • {gap}")

    # Overall readiness
    ready = (
        c.get("coverage_ok", False) and
        q.get("quality_ok", False)
    )
    print(f"\n{'='*60}")
    print(f"Overall Readiness: {'[READY FOR EVALUATION]' if ready else '[IN PROGRESS]'}")
    print("="*60 + "\n")

    if latex:
        print("\n--- LaTeX Statistics Table ---\n")
        print(generate_paper_stats_table(coverage, quality))


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Corpus Validator")
    parser.add_argument("--report", action="store_true", help="Include LaTeX stats table")
    args = parser.parse_args()

    corpus_df   = load_corpus_index()
    passages_df = load_annotation_queue()
    registry    = load_collection_registry()

    coverage = validate_coverage(corpus_df)   if corpus_df   is not None else {}
    quality  = validate_text_quality(passages_df) if passages_df is not None else {}
    gaps     = check_missing_gaps(corpus_df)  if corpus_df   is not None else []

    print_validation_report(coverage, quality, gaps, registry, latex=args.report)


if __name__ == "__main__":
    main()
