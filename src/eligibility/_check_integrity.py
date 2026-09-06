import json, sys
from pathlib import Path

root = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
sys.path.insert(0, str(root))

raw_files = list((root / "data/documents/raw").glob("*.html"))
with open(root / "data/documents/raw/metadata.json", "r", encoding="utf-8") as f:
    raw_meta = json.load(f)

proc_files = list((root / "data/processed/documents").glob("*.txt"))
with open(root / "data/processed/corpus_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(root / "data/processed/corpus_lock_report.json", "r", encoding="utf-8") as f:
    lock_report = json.load(f)

from src.rag.vector_store import VectorStoreManager
vs = VectorStoreManager()
v_count = vs.collection.count()

print("=== FINAL DATA INTEGRITY VERIFICATION ===")
print(f"Raw HTML files: {len(raw_files)} (Expected: 25)")
print(f"Raw metadata entries: {len(raw_meta)} (Expected: 40)")
print(f"Processed TXT files: {len(proc_files)} (Expected: 16)")
print(f"Corpus manifest entries: {len(manifest)} (Expected: 16)")
print(f"Lock report schemes: {lock_report.get('scheme_coverage_count')} (Expected: 14)")
print(f"Vector store count: {v_count} (Expected: 166)")
all_ok = (len(raw_files) == 25 and len(raw_meta) == 40 and len(proc_files) == 16 and len(manifest) == 16 and v_count == 166)
print(f"Integrity check: {'PASS' if all_ok else 'FAIL'}")
