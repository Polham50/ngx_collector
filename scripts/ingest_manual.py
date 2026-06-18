import os
import sys
from pathlib import Path
import json

# Add scripts dir to path to import from ngx_collector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ngx_collector import MetadataTracker, classify_document, infer_year

def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_pdf_dir = base_dir / "data" / "raw_pdfs"
    companies_file = base_dir / "data" / "companies.json"
    
    with open(companies_file, "r") as f:
        companies_data = json.load(f)["companies"]
    
    # Create lookup mappings
    ticker_info = {c["ticker"]: c for c in companies_data}
    
    tracker = MetadataTracker()
    
    added_count = 0
    for pdf_path in raw_pdf_dir.rglob("*.pdf"):
        folder_ticker = pdf_path.parent.name
        
        # Check if already tracked by looking for this local path
        already_tracked = False
        for r in tracker.data["discovered"]:
            if r.get("local_path") == str(pdf_path):
                already_tracked = True
                break
        
        if already_tracked:
            continue
            
        print(f"Registering new manual file: {pdf_path.name} for {folder_ticker}")
        
        # Attempt to map to proper ticker if user used a name like "DANGOTE CEMENT"
        comp = ticker_info.get(folder_ticker.upper(), {})
        if not comp:
            # Try fuzzy match against company names
            for t, info in ticker_info.items():
                if folder_ticker.upper() in t or folder_ticker.replace(" ", "").upper() in info["name"].replace(" ", "").upper():
                    comp = info
                    break
                    
        actual_ticker = comp.get("ticker", folder_ticker.upper())
        actual_sector = comp.get("sector", "Unknown")
        actual_name = comp.get("name", actual_ticker)
        
        fake_url = f"manual://{actual_ticker}/{pdf_path.name}"
        
        title = pdf_path.stem
        text_for_class = title.replace("_", " ").lower()
        
        record = {
            "source": "manual",
            "ticker": actual_ticker,
            "url": fake_url,
            "title": title,
            "doc_type": classify_document(text_for_class),
            "company_name": actual_name,
            "sector": actual_sector,
            "year": infer_year(text_for_class, fake_url) or "2023",
        }
        
        tracker.add_discovered(record)
        tracker.mark_downloaded(fake_url, str(pdf_path))
        added_count += 1
        
    tracker.save()
    print(f"\nSuccessfully registered {added_count} manual PDFs.")

if __name__ == "__main__":
    main()
