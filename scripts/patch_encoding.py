"""Patch all scripts to use ASCII-safe characters for Windows cp1252 compatibility."""
import pathlib

SCRIPTS = pathlib.Path("scripts")

replacements = [
    ("\u03b2=", "coef="),          # beta
    ("R\u00b2=", "R2="),           # R-squared
    ("Adj-R\u00b2=", "AdjR2="),    # Adj R-squared
    ("R\u00b2", "R2"),
    # Greek rho in group label
    ("Spearman \u03c1 (sentiment vs CAR)", "Spearman rho (sentiment vs CAR)"),
    (".read_text(encoding="utf-8")", '.read_text(encoding="utf-8")'),
]

for f in SCRIPTS.glob("*.py"):
    txt = f.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)
            changed = True
    if changed:
        f.write_text(txt, encoding="utf-8")
        print(f"Patched: {f.name}")

print("Patch complete.")
