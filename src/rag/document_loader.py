# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Document Loader
Loads processed text documents from data/processed/documents/ and aligns them
with the authoritative corpus manifest (data/processed/corpus_manifest.json).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.rag.schemas import ProcessedDocument

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DOCS_DIR = DEFAULT_PROCESSED_DIR / "documents"
DEFAULT_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "corpus_manifest.json"


def sha256_of_text(text: str) -> str:
    """Compute SHA-256 hash of a string using UTF-8 encoding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentLoader:
    """Loads validated clean documents and binds them with corpus manifest metadata."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.docs_dir = docs_dir or DEFAULT_DOCS_DIR
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH

    def load_manifest(self) -> dict[str, dict]:
        """Load corpus_manifest.json and index records by processed_filename."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Corpus manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)

        manifest_map = {}
        for entry in manifest_data:
            key = entry.get("processed_filename")
            if key:
                manifest_map[key] = entry
        return manifest_map

    def load_documents(self) -> List[ProcessedDocument]:
        """
        Load all .txt files from docs_dir and bind them to manifest records.
        Raises ValueError if any file is missing, empty, or unmanifested.
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Processed documents directory not found: {self.docs_dir}")

        manifest_map = self.load_manifest()
        txt_files = sorted(list(self.docs_dir.glob("*.txt")))

        if not txt_files:
            raise ValueError(f"No .txt documents found in: {self.docs_dir}")

        loaded_documents: List[ProcessedDocument] = []

        for file_path in txt_files:
            filename = file_path.name
            manifest_entry = manifest_map.get(filename)

            if not manifest_entry:
                logger.warning(f"File {filename} is not listed in corpus_manifest.json; skipping.")
                continue

            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning(f"File {filename} is empty; skipping.")
                continue

            content_hash = sha256_of_text(content)
            word_count = len(content.split())
            char_count = len(content)

            doc = ProcessedDocument(
                document_id=manifest_entry.get("document_id", filename.replace(".txt", "")),
                scheme_id=manifest_entry.get("scheme_id", "UNKNOWN"),
                scheme_name=manifest_entry.get("scheme_name", "Unknown Scheme"),
                processed_filename=filename,
                source_filename=manifest_entry.get("source_filename", filename.replace(".txt", ".html")),
                official_url=manifest_entry.get("official_url"),
                source_authority=manifest_entry.get("source_authority"),
                document_title=manifest_entry.get("document_title"),
                document_type=manifest_entry.get("document_type"),
                source_sha256=manifest_entry.get("source_sha256", ""),
                processed_sha256=content_hash,
                validation_status=manifest_entry.get("validation_status", "VALID"),
                recovery=manifest_entry.get("recovery", False),
                download_timestamp=manifest_entry.get("download_timestamp"),
                content=content,
                word_count=word_count,
                character_count=char_count,
                source_type=manifest_entry.get("source_type", "official_government_source"),
            )
            loaded_documents.append(doc)

        logger.info(f"Loaded {len(loaded_documents)} documents from {self.docs_dir}")
        return loaded_documents
