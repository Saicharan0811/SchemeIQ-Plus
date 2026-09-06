"""Document downloader for official government PDFs and web pages."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel
import requests

from src.ingestion.manifest_loader import ManifestEntry, is_official_domain

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 SchemeIQPlus/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,te;q=0.8",
}


def sanitize_filename(name: str) -> str:
    """Sanitize title or identifier to a safe filesystem filename."""
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", cleaned)


def compute_sha256(data: bytes) -> str:
    """Calculate SHA256 hexadecimal hash for byte content."""
    return hashlib.sha256(data).hexdigest()


class DownloadResult(BaseModel):
    """Result schema for an ingested document."""

    scheme_id: str
    scheme_name: str
    document_title: str
    document_type: str
    official_url: str
    source_authority: str
    local_filename: str
    file_format: str
    file_size_bytes: int
    sha256: str
    download_timestamp: str
    status: str
    error_message: Optional[str] = None


def fetch_official_document(
    entry: ManifestEntry,
    target_dir: Path | str,
    index: int = 1,
    timeout_seconds: int = 20,
) -> DownloadResult:
    """
    Download or fetch an official document/webpage from a verified government domain.
    """
    if not is_official_domain(entry.official_url):
        raise ValueError(f"Security Alert: Blocked download from untrusted source: {entry.official_url}")

    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    slug = sanitize_filename(entry.document_title)[:50]

    try:
        response = requests.get(
            entry.official_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout_seconds,
            allow_redirects=True,
            verify=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        is_pdf = (
            entry.official_url.lower().endswith(".pdf")
            or "application/pdf" in content_type
            or response.content.startswith(b"%PDF-")
        )

        if is_pdf:
            ext = "pdf"
            file_format = "application/pdf"
            content_bytes = response.content
        else:
            ext = "html"
            file_format = "text/html"
            content_bytes = response.content

        filename = f"{entry.scheme_id}_{index:02d}_{slug}.{ext}"
        target_path = out_dir / filename

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        file_hash = compute_sha256(content_bytes)
        file_size = len(content_bytes)

        return DownloadResult(
            scheme_id=entry.scheme_id,
            scheme_name=entry.scheme_name,
            document_title=entry.document_title,
            document_type=entry.document_type,
            official_url=entry.official_url,
            source_authority=entry.source_authority,
            local_filename=filename,
            file_format=file_format,
            file_size_bytes=file_size,
            sha256=file_hash,
            download_timestamp=timestamp,
            status="SUCCESS",
            error_message=None,
        )

    except requests.exceptions.SSLError:
        # Retry with verified exception logging if a government portal has legacy SSL certificate chains
        try:
            response = requests.get(
                entry.official_url,
                headers=DEFAULT_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
                verify=False,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            is_pdf = entry.official_url.lower().endswith(".pdf") or "application/pdf" in content_type
            ext = "pdf" if is_pdf else "html"
            file_format = "application/pdf" if is_pdf else "text/html"
            filename = f"{entry.scheme_id}_{index:02d}_{slug}.{ext}"
            target_path = out_dir / filename

            with open(target_path, "wb") as f:
                f.write(response.content)

            file_hash = compute_sha256(response.content)
            return DownloadResult(
                scheme_id=entry.scheme_id,
                scheme_name=entry.scheme_name,
                document_title=entry.document_title,
                document_type=entry.document_type,
                official_url=entry.official_url,
                source_authority=entry.source_authority,
                local_filename=filename,
                file_format=file_format,
                file_size_bytes=len(response.content),
                sha256=file_hash,
                download_timestamp=timestamp,
                status="SUCCESS",
                error_message="Fetched with fallback SSL handling",
            )
        except Exception as fallback_err:
            return DownloadResult(
                scheme_id=entry.scheme_id,
                scheme_name=entry.scheme_name,
                document_title=entry.document_title,
                document_type=entry.document_type,
                official_url=entry.official_url,
                source_authority=entry.source_authority,
                local_filename="",
                file_format="",
                file_size_bytes=0,
                sha256="",
                download_timestamp=timestamp,
                status="FAILED",
                error_message=str(fallback_err),
            )
    except Exception as err:
        return DownloadResult(
            scheme_id=entry.scheme_id,
            scheme_name=entry.scheme_name,
            document_title=entry.document_title,
            document_type=entry.document_type,
            official_url=entry.official_url,
            source_authority=entry.source_authority,
            local_filename="",
            file_format="",
            file_size_bytes=0,
            sha256="",
            download_timestamp=timestamp,
            status="FAILED",
            error_message=str(err),
        )
