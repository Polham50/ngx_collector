import os
import json
import shutil
from pathlib import Path
from ngx_collector import extract_narrative_sections

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    COMPANIES_JSON = BASE_DIR / "data" / "companies.json"
    
    with open(COMPANIES_JSON, "r", encoding="utf-8") as f:
        companies_data = json.load(f)
    ticker_to_sector = {c["ticker"]: c["sector"] for c in companies_data["companies"]}
    
    TEXT_DIR = BASE_DIR / "data" / "extracted_text"
    RAW_DIR = BASE_DIR / "data" / "raw_pdfs"

    print("Step 1: Moving PDFs to the correct folder...")
    pdfs_in_text = list(TEXT_DIR.rglob("*.pdf"))
    for pdf in pdfs_in_text:
        ticker = pdf.parent.name
        dest_dir = RAW_DIR / ticker
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / pdf.name
        # Use shutil.move to handle overwrites or just skip if exists
        if dest_path.exists() and str(dest_path.resolve()) != str(pdf.resolve()):
            dest_path.unlink()
        
        if str(dest_path.resolve()) != str(pdf.resolve()):
            shutil.move(str(pdf), str(dest_path))
            print(f"  Moved {pdf.name} to raw_pdfs/{ticker}/")

    print("\nStep 2: Extracting text from manually added PDFs...")
    all_pdfs = list(RAW_DIR.rglob("*.pdf"))
    extracted_count = 0

    for pdf in all_pdfs:
        ticker = pdf.parent.name
        json_path = TEXT_DIR / ticker / (pdf.stem + ".json")
        
        if not json_path.exists():
            print(f"  Extracting: {pdf.name} (Ticker: {ticker})")
            sections = extract_narrative_sections(pdf)
            
            # Build mock record for text_cleaner
            sector = ticker_to_sector.get(ticker, "Manual")
            record = {
                "ticker": ticker,
                "company_name": ticker,
                "sector": sector,
                "year": "Unknown",
                "doc_type": "manual_pdf",
                "source": "manual",
                "local_path": str(pdf),
                "sections": sections,
                "char_counts": {k: len(v) for k, v in sections.items() if isinstance(v, str)}
            }
            
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            extracted_count += 1

    print(f"\nDone! Extracted {extracted_count} new PDFs to JSON.")

if __name__ == "__main__":
    main()
