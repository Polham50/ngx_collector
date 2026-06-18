# NGX-FND: Nigerian Exchange Financial Narratives Dataset
## Data Collection Toolkit

**Research project:** *Extracting Sentiment and Forward Guidance Signals from Nigerian Exchange Corporate Disclosures Using Large Language Models*  
**Author:** Timothy Popoola Oladapo | Dataqury Analytics  
**Affiliation:** WorldQuant University (MSc Financial Engineering)

---

## Project Structure

```
ngx_collector/
├── data/
│   ├── companies.json          ← 50 target companies across 5 sectors
│   ├── raw_pdfs/               ← Downloaded PDF filings (by ticker)
│   ├── extracted_text/         ← JSON files with extracted narrative sections
│   ├── cleaned_text/           ← Cleaned + segmented passages
│   └── metadata/
│       ├── collection_registry.json  ← Download tracker
│       ├── corpus_index.csv          ← Master document index
│       ├── annotation_queue.csv      ← All passages + annotation fields
│       └── gold_standard.csv         ← Completed human annotations
├── logs/                       ← Timestamped run logs
└── scripts/
    ├── ngx_collector.py        ← Phase 1-3: Discover, download, extract
    ├── text_cleaner.py         ← Phase 4: Clean + segment passages
    ├── corpus_validator.py     ← Phase 5: Quality + coverage checks
    └── annotator.py            ← Phase 6: Human annotation CLI
```

---

## Quickstart

### Step 1 — Discover available filings
```bash
cd scripts
python ngx_collector.py --mode discover
```
Scrapes NGX portal, Proshare, and company IR pages for PDF links across all 50 companies.

### Step 2 — Download PDFs
```bash
python ngx_collector.py --mode download
```
Downloads all discovered PDFs (skips already-downloaded files).

### Step 3 — Extract narrative text
```bash
python ngx_collector.py --mode extract
```
Uses `pdfplumber` to extract narrative sections (Chairman's Statement, CEO Review, Outlook, etc.) from each PDF.

### Run all three steps at once
```bash
python ngx_collector.py --mode full
```

### Single company test run
```bash
python ngx_collector.py --mode full --ticker MTNN
python ngx_collector.py --mode full --ticker ZENITHBANK
```

---

### Step 4 — Clean and segment text
```bash
python text_cleaner.py
```
Normalizes Unicode, removes page headers/footers and boilerplate, fixes OCR artifacts, and segments passages into annotation-ready chunks. Outputs `annotation_queue.csv`.

---

### Step 5 — Validate corpus
```bash
python corpus_validator.py           # Check coverage and quality
python corpus_validator.py --report  # Include LaTeX stats table for paper
```

---

### Step 6 — Annotate passages
```bash
# Start your annotation session
python annotator.py --annotator "Timothy"

# Annotate a specific sector
python annotator.py --annotator "Timothy" --sector Banking

# Check progress
python annotator.py --stats

# Export gold standard for LLM evaluation
python annotator.py --export
```

---

## Target Corpus

| Sector | Target Companies | Target Docs |
|---|---|---|
| Banking | 10 | 50 |
| Oil & Gas | 8 | 40 |
| Consumer Goods | 14 | 70 |
| Industrial | 8 | 40 |
| Telecoms | 2 | 10 |
| **Total** | **50** | **210** |

**Document types:** Annual Reports, Interim Reports, Earnings Releases, AGM Statements  
**Years:** 2020 – 2024  
**Target passages for annotation:** 300+ (gold standard: 150+ fully annotated)

---

## Annotation Schema

Each passage is labeled with:

| Field | Values |
|---|---|
| `sentiment_label` | `positive` / `negative` / `neutral` |
| `sentiment_intensity` | `mild` / `moderate` / `strong` |
| `has_guidance` | `True` / `False` |
| `guidance_type` | `positive` / `negative` / `neutral` / `conditional` |
| `guidance_span` | Key sentence containing guidance |

---

## Data Sources

- **NGX Group portal:** `ngxgroup.com/exchange/data/company-filings/`
- **SEC Nigeria:** `sec.gov.ng`
- **Proshare Nigeria:** `proshareng.com`
- **Company IR pages:** Direct investor relations pages per company

---

## Citation (Forthcoming)

```
Oladapo, T.P. (2025). Extracting Sentiment and Forward Guidance Signals 
from Nigerian Exchange Corporate Disclosures Using Large Language Models. 
[Target: FinNLP @ EMNLP 2026 / AfricaNLP @ ACL 2026]
```

---

## Notes

- All PDFs are publicly available regulatory filings
- PDF size cap: 50MB per document
- Request delay: 2 seconds between requests (respectful scraping)
- Logs are timestamped and stored in `/logs/`
- The `companies.json` registry can be extended with additional tickers
