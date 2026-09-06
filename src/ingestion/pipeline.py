"""Ingestion pipeline runner and metadata catalog generator."""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

from src.ingestion.downloader import DownloadResult, fetch_official_document
from src.ingestion.manifest_loader import load_source_manifest


def run_ingestion_pipeline(
    manifest_path: str | Path = "data/documents/source_manifest.csv",
    output_dir: str | Path = "data/documents/raw",
    timeout_seconds: int = 20,
) -> Dict[str, any]:
    """
    Execute the official document ingestion pipeline.
    Preserves original source URL, authority, and SHA256 checksums in metadata.json.
    """
    manifest_file = Path(manifest_path).resolve()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    entries = load_source_manifest(manifest_file)
    results: List[DownloadResult] = []

    for idx, entry in enumerate(entries, start=1):
        res = fetch_official_document(
            entry=entry,
            target_dir=target_dir,
            index=idx,
            timeout_seconds=timeout_seconds,
        )
        results.append(res)

    # Save provenance metadata catalog to metadata.json
    metadata_path = target_dir / "metadata.json"
    records = [res.model_dump() for res in results]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in results if r.status == "SUCCESS")
    failed_count = sum(1 for r in results if r.status == "FAILED")

    return {
        "total_manifest_entries": len(entries),
        "successful_downloads": success_count,
        "failed_downloads": failed_count,
        "raw_directory": str(target_dir),
        "metadata_catalog": str(metadata_path),
        "results": records,
    }
