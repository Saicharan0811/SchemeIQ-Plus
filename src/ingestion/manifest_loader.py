"""Manifest loader and domain validator for official scheme documents."""

from pathlib import Path
from typing import List
from urllib.parse import urlparse
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Strict whitelist of official government and statutory organization domains
ALLOWED_DOMAIN_SUFFIXES = [
    ".gov.in",
    ".nic.in",
]

EXPLICIT_ALLOWED_DOMAINS = {
    "pfrda.org.in",
    "www.pfrda.org.in",
    "npscra.nsdl.co.in",
    "www.npscra.nsdl.co.in",
    "mudra.org.in",
    "www.mudra.org.in",
    "telanganadiagnostics.com",
    "www.telanganadiagnostics.com",
}


def is_official_domain(url: str) -> bool:
    """Validate whether the URL belongs to a verified official government domain."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False

        if hostname in EXPLICIT_ALLOWED_DOMAINS:
            return True

        for suffix in ALLOWED_DOMAIN_SUFFIXES:
            if hostname.endswith(suffix):
                return True

        return False
    except Exception:
        return False


class ManifestEntry(BaseModel):
    """Schema representing a verified document source entry."""

    scheme_id: str
    scheme_name: str
    document_title: str
    document_type: str
    official_url: str
    source_authority: str
    publication_or_update_date: str = Field(default="NOT SPECIFIED")

    @field_validator("official_url")
    @classmethod
    def validate_url_domain(cls, v: str) -> str:
        if not is_official_domain(v):
            raise ValueError(f"URL does not belong to a verified official government source: {v}")
        return v


def load_source_manifest(manifest_path: str | Path) -> List[ManifestEntry]:
    """Load and validate all entries from the source manifest CSV."""
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {path}")

    df = pd.read_csv(path)
    required_cols = [
        "scheme_id",
        "scheme_name",
        "document_title",
        "document_type",
        "official_url",
        "source_authority",
        "publication_or_update_date",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in manifest: {missing_cols}")

    entries: List[ManifestEntry] = []
    for idx, row in df.iterrows():
        entry = ManifestEntry(
            scheme_id=str(row["scheme_id"]).strip(),
            scheme_name=str(row["scheme_name"]).strip(),
            document_title=str(row["document_title"]).strip(),
            document_type=str(row["document_type"]).strip(),
            official_url=str(row["official_url"]).strip(),
            source_authority=str(row["source_authority"]).strip(),
            publication_or_update_date=str(row["publication_or_update_date"]).strip(),
        )
        entries.append(entry)

    return entries
