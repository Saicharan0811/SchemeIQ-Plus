"""Document and metadata ingestion module for SchemeIQ+."""

from src.ingestion.corpus_validator import (
    extract_html_text,
    validate_corpus,
    write_validation_reports,
)
from src.ingestion.downloader import DownloadResult, fetch_official_document
from src.ingestion.manifest_loader import (
    ManifestEntry,
    is_official_domain,
    load_source_manifest,
)
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.recovery_pipeline import run_recovery_pipeline

__all__ = [
    "ManifestEntry",
    "is_official_domain",
    "load_source_manifest",
    "DownloadResult",
    "fetch_official_document",
    "run_ingestion_pipeline",
    "run_recovery_pipeline",
    "extract_html_text",
    "validate_corpus",
    "write_validation_reports",
]
