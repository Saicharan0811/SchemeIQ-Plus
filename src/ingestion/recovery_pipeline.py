"""Recovery pipeline: downloads replacement documents for missing/weak corpus entries.

Reads source_recovery_candidates.csv (VERIFIED_REPLACEMENT rows only),
downloads each document via the existing fetch_official_document() function,
appends results to data/documents/raw/metadata.json, and does NOT modify
data/schemes.csv or source_manifest.csv.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.downloader import DownloadResult, fetch_official_document
from src.ingestion.manifest_loader import ManifestEntry

# -- Paths ------------------------------------------------------------------
# __file__ is src/ingestion/recovery_pipeline.py -> .parent = src/ingestion
#                                                -> .parent.parent = src
#                                                -> .parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CANDIDATES_CSV = PROJECT_ROOT / "data" / "documents" / "source_recovery_candidates.csv"
RAW_DIR = PROJECT_ROOT / "data" / "documents" / "raw"
METADATA_JSON = RAW_DIR / "metadata.json"


def _load_existing_metadata(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_metadata(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def run_recovery_pipeline() -> dict:
    """Download VERIFIED_REPLACEMENT candidates and append provenance to metadata.json."""

    # -- 1. Load candidates ---------------------------------------------------
    candidates: list[dict] = []
    with open(CANDIDATES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["classification"].strip() in ("VERIFIED_REPLACEMENT", "VERIFIED_DOWNLOADABLE"):
                candidates.append(row)

    print(f"[Recovery] {len(candidates)} VERIFIED_REPLACEMENT candidates to attempt.")

    # -- 2. Load existing metadata so we can track already-downloaded files --
    existing_records = _load_existing_metadata(METADATA_JSON)
    existing_urls = {r["official_url"] for r in existing_records}

    # -- 3. Compute a sequential index starting after existing successful files
    #       to avoid filename collisions (REC prefix distinguishes recovery files)
    results: list[DownloadResult] = []
    rec_index = 30  # start at 30 to avoid collisions with manifested 01-28 slot

    successes = 0
    failures = 0

    for cand in candidates:
        url = cand["replacement_url"].strip()
        scheme_id = cand["scheme_id"].strip()

        if url in existing_urls:
            print(f"  [SKIP] Already have: {url}")
            continue

        try:
            entry = ManifestEntry(
                scheme_id=scheme_id,
                scheme_name=cand["scheme_name"].strip(),
                document_title=cand["document_title"].strip(),
                document_type=cand["document_type"].strip(),
                official_url=url,
                source_authority=cand["source_authority"].strip(),
                publication_or_update_date="2026-08-15",
            )
        except Exception as validation_err:
            print(f"  [BLOCKED] {url}: {validation_err}")
            failures += 1
            # Record failure for audit
            existing_records.append({
                "scheme_id": scheme_id,
                "scheme_name": cand["scheme_name"].strip(),
                "document_title": cand["document_title"].strip(),
                "document_type": cand["document_type"].strip(),
                "official_url": url,
                "source_authority": cand["source_authority"].strip(),
                "local_filename": "",
                "file_format": "",
                "file_size_bytes": 0,
                "sha256": "",
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "BLOCKED",
                "error_message": str(validation_err),
                "recovery": True,
                "old_url": cand["old_url"].strip(),
                "replacement_reason": cand["replacement_reason"].strip(),
            })
            continue

        print(f"  [DL] {scheme_id} -> {url}")
        result = fetch_official_document(entry, RAW_DIR, index=rec_index)

        rec_index += 1
        results.append(result)

        record = result.model_dump()
        record["recovery"] = True
        record["old_url"] = cand["old_url"].strip()
        record["replacement_reason"] = cand["replacement_reason"].strip()
        existing_records.append(record)

        if result.status == "SUCCESS":
            successes += 1
            print(f"    [OK] Saved: {result.local_filename} ({result.file_size_bytes:,} bytes)")
        else:
            failures += 1
            print(f"    [FAIL] {result.error_message}")

    # -- 4. Persist updated metadata -----------------------------------------
    _save_metadata(METADATA_JSON, existing_records)
    print(f"\n[Recovery] Done. Successes: {successes}  Failures: {failures}")
    print(f"[Recovery] Updated metadata.json with {len(existing_records)} total records.")

    return {
        "recovery_attempted": len(candidates),
        "successes": successes,
        "failures": failures,
        "total_metadata_records": len(existing_records),
    }


if __name__ == "__main__":
    import sys
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    summary = run_recovery_pipeline()
    print(summary)
