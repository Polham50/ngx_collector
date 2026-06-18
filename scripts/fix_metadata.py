import json
import shutil
from pathlib import Path

base_dir = Path("c:/Users/USER/Desktop/DataQUry/ngx_collector")
raw_dir = base_dir / "data" / "raw_pdfs"

renames = {
    "GTBANK": "GTCO",
    "NIGERIABREWRIES": "NB",
    "DANGOTE": "DANGCEM",
    "SEPLATENERGY": "SEPLAT"
}

# 1. Rename directories
for old, new in renames.items():
    old_p = raw_dir / old
    new_p = raw_dir / new
    if old_p.exists():
        if not new_p.exists():
            old_p.rename(new_p)
        else:
            for f in old_p.glob("*"):
                f.rename(new_p / f.name)
            try:
                old_p.rmdir()
            except Exception as e:
                print(f"Could not remove {old_p}: {e}")

# Load company info
with open(base_dir / "data" / "companies.json") as f:
    comps = {c["ticker"]: c for c in json.load(f)["companies"]}

# 2. Patch registry
reg_file = base_dir / "data" / "metadata" / "collection_registry.json"
with open(reg_file) as f:
    reg = json.load(f)

for r in reg.get("discovered", []):
    if r.get("source") == "manual":
        parts = r["url"].replace("manual://", "").split("/")
        old_t = parts[0]
        if old_t in renames:
            new_t = renames[old_t]
            new_url = f"manual://{new_t}/{parts[1]}"
            r["url"] = new_url
            r["ticker"] = new_t
            r["company_name"] = comps[new_t]["name"]
            r["sector"] = comps[new_t]["sector"]
            if "local_path" in r:
                r["local_path"] = r["local_path"].replace(f"\\{old_t}\\", f"\\{new_t}\\").replace(f"/{old_t}/", f"/{new_t}/")
            if "extraction_stats" in r and "text_path" in r["extraction_stats"]:
                 pass # text path stays same for now

extracted_new = []
for url in reg.get("extracted", []):
    if url.startswith("manual://"):
        parts = url.replace("manual://", "").split("/")
        old_t = parts[0]
        if old_t in renames:
            extracted_new.append(f"manual://{renames[old_t]}/{parts[1]}")
            continue
    extracted_new.append(url)
reg["extracted"] = extracted_new

downloaded_new = []
for url in reg.get("downloaded", []):
    if url.startswith("manual://"):
        parts = url.replace("manual://", "").split("/")
        old_t = parts[0]
        if old_t in renames:
            new_t = renames[old_t]
            new_url = f"manual://{new_t}/{parts[1]}"
            downloaded_new.append(new_url)
            continue
    downloaded_new.append(url)
reg["downloaded"] = downloaded_new

with open(reg_file, "w") as f:
    json.dump(reg, f, indent=2)

# 3. Patch json files in extracted_text and cleaned_text
for text_dir in [base_dir / "data" / "extracted_text", base_dir / "data" / "cleaned_text"]:
    if not text_dir.exists(): continue
    for jf in text_dir.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        changed = False
        meta = data.get("metadata", {})
        if meta.get("source") == "manual":
            old_t = meta.get("ticker")
            if old_t in renames:
                new_t = renames[old_t]
                meta["ticker"] = new_t
                meta["company_name"] = comps[new_t]["name"]
                meta["sector"] = comps[new_t]["sector"]
                
                parts = meta["url"].replace("manual://", "").split("/")
                meta["url"] = f"manual://{new_t}/{parts[1]}"
                changed = True
                
        if changed:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

print("Metadata fixed successfully.")
