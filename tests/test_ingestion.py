"""Automated tests for the official document ingestion pipeline."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.ingestion.downloader import compute_sha256, sanitize_filename
from src.ingestion.manifest_loader import (
    ManifestEntry,
    is_official_domain,
    load_source_manifest,
)


def test_domain_security_whitelist():
    """Verify that only authorized official government domains are permitted."""
    valid_urls = [
        "https://farmerswelfarecommission.telangana.gov.in/",
        "https://pmkisan.gov.in/Documents/OperationalGuidelines.pdf",
        "https://nha.gov.in/PM-JAY",
        "https://pfrda.org.in/web/pfrda/schemes/atal-pension-yojana-apy",
        "https://telanganadiagnostics.com/",
        "https://telanganaepass.cgg.gov.in/FAQ.do",
        "https://www.kviconline.gov.in/pmegpeportal/",
    ]

    for url in valid_urls:
        assert is_official_domain(url) is True, f"Official URL rejected: {url}"

    unauthorized_urls = [
        "https://en.wikipedia.org/wiki/Rythu_Bandhu",
        "https://someblog.wordpress.com/schemes",
        "https://fakegovernment-schemes.com",
        "https://gov.in.scam-site.org",
        "https://newsportal.com/telangana-schemes",
    ]

    for url in unauthorized_urls:
        assert is_official_domain(url) is False, f"Unauthorized URL was not rejected: {url}"


def test_manifest_entry_validation_security():
    """Verify that creating a ManifestEntry with an unauthorized domain raises an error."""
    with pytest.raises(ValidationError):
        ManifestEntry(
            scheme_id="TS001",
            scheme_name="Test Scheme",
            document_title="Unofficial Document",
            document_type="Blog",
            official_url="https://unauthorized-blog.com/test.pdf",
            source_authority="Unknown Source",
            publication_or_update_date="2024-01-01",
        )


def test_load_source_manifest_integrity():
    """Verify that the official source_manifest.csv loads and validates completely."""
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "documents" / "source_manifest.csv"
    assert manifest_path.exists(), "source_manifest.csv is missing"

    entries = load_source_manifest(manifest_path)
    assert len(entries) == 28, f"Expected 28 entries, got {len(entries)}"

    scheme_ids = set(e.scheme_id for e in entries)
    assert len(scheme_ids) == 14, f"Expected 14 unique schemes, got {len(scheme_ids)}"

    for entry in entries:
        assert entry.scheme_id.startswith(("TS", "CT"))
        assert len(entry.document_title) > 0
        assert is_official_domain(entry.official_url) is True


def test_sha256_computation_and_filename_sanitization():
    """Verify SHA256 checksum computation and filename sanitization utilities."""
    sample_bytes = b"SchemeIQ+ Official Government Document Test Content"
    checksum = compute_sha256(sample_bytes)
    assert len(checksum) == 64
    assert isinstance(checksum, str)

    raw_title = "G.O. Ms. No. 3: Transport & R&B (Free Bus Travel) / 2024"
    cleaned = sanitize_filename(raw_title)
    assert "/" not in cleaned
    assert ":" not in cleaned
    assert " " not in cleaned
