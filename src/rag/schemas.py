# -*- coding: utf-8 -*-
"""
SchemeIQ+ — RAG Data Schemas
Defines core data models for processed documents, chunks, metadata, and ingestion results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessedDocument(BaseModel):
    """Represents a validated, clean source document ready for chunking."""
    document_id: str
    scheme_id: str
    scheme_name: str
    processed_filename: str
    source_filename: str
    official_url: Optional[str] = None
    source_authority: Optional[str] = None
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    source_sha256: str
    processed_sha256: str
    validation_status: str = "VALID"
    recovery: bool = False
    download_timestamp: Optional[str] = None
    content: str
    word_count: int
    character_count: int
    source_type: str = "official_government_source"


class DocumentChunk(BaseModel):
    """Represents an atomic, provenance-preserving chunk for vector search."""
    chunk_id: str
    scheme_id: str
    scheme_name: str
    source_filename: str
    official_url: str
    source_authority: str
    document_title: str
    document_type: str
    recovery: bool
    chunk_index: int
    total_chunks_in_document: int
    document_content_hash: str
    chunk_content_hash: str
    source_type: str = "official_government_source"
    text: str
    word_count: int
    character_count: int

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to a flat dict format required by ChromaDB."""
        return {
            "chunk_id": self.chunk_id,
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "source_filename": self.source_filename,
            "official_url": self.official_url,
            "source_authority": self.source_authority,
            "document_title": self.document_title,
            "document_type": self.document_type,
            "recovery": self.recovery,
            "chunk_index": self.chunk_index,
            "total_chunks_in_document": self.total_chunks_in_document,
            "document_content_hash": self.document_content_hash,
            "chunk_content_hash": self.chunk_content_hash,
            "source_type": self.source_type,
            "word_count": self.word_count,
            "character_count": self.character_count,
        }


class RetrievalResult(BaseModel):
    """Represents a single retrieved chunk match for a query."""
    rank: int
    chunk_id: str
    similarity_score: float
    distance: float
    scheme_id: str
    scheme_name: str
    source_filename: str
    official_url: str
    source_authority: str
    document_title: str
    text_preview: str
    text: str


class SmokeTestResult(BaseModel):
    """Represents smoke test query execution results."""
    query_id: int
    query_text: str
    top_matches: List[RetrievalResult]


class IngestionReport(BaseModel):
    """Complete summary report of the RAG ingestion pipeline run."""
    ingestion_timestamp: str
    source_directory: str
    documents_expected: int
    documents_loaded: int
    scheme_coverage_expected: int
    scheme_coverage_indexed: int
    total_chunks_created: int
    total_chunks_embedded: int
    total_chunks_skipped: int
    total_empty_chunks_rejected: int
    duplicate_chunk_ids: int
    failed_chunks: int
    vector_store: Dict[str, Any]
    embedding: Dict[str, Any]
    integrity_status: str
    per_document_chunks: Dict[str, int]
    per_scheme_chunks: Dict[str, int]
    smoke_tests: List[SmokeTestResult]


# ---------------------------------------------------------------------------
# Phase 5 — Answer Generation Schemas
# ---------------------------------------------------------------------------

class GroundingStatus(str):
    """
    Grounding status enum values for GroundedAnswer.

    Values:
        GROUNDED               – answer fully supported by retrieved context
        PARTIALLY_GROUNDED     – answer partly supported; some gaps flagged
        INSUFFICIENT_CONTEXT   – retrieved context did not contain enough info
        CONFLICTING_SOURCES    – retrieved chunks contain conflicting information
        GENERATION_FAILED      – LLM call failed or returned empty answer
    """
    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    CONFLICTING_SOURCES = "CONFLICTING_SOURCES"
    GENERATION_FAILED = "GENERATION_FAILED"


class SourceCitation(BaseModel):
    """A deduplicated, human-readable citation for an official retrieved source."""
    source_id: str                          # e.g. "1", "2", "3"
    scheme_id: str
    scheme_name: str
    document_title: Optional[str] = None
    source_filename: str
    official_url: str
    chunk_id: str

    def as_readable_label(self) -> str:
        """Return a formatted label for inclusion in the answer text."""
        title = self.document_title or self.scheme_name
        return f"[{self.source_id}] {self.scheme_name} — {title}\n    {self.official_url}"


class GroundedAnswer(BaseModel):
    """
    A fully-structured, grounded answer from the RAG pipeline.

    IMPORTANT: grounding_status=GROUNDED means the answer is derived from
    retrieved official context. It does NOT guarantee factual correctness —
    the source documents themselves may be outdated or incomplete.
    """
    question: str
    answer: str
    detected_scheme_id: Optional[str] = None
    detected_scheme_name: Optional[str] = None
    retrieval_count: int = 0
    retrieved_scheme_ids: List[str] = Field(default_factory=list)
    sources: List[SourceCitation] = Field(default_factory=list)
    grounding_status: str = GroundingStatus.GROUNDED
    grounding_note: str = ""
    model_name: str = ""
    provider_name: str = ""
    generation_timestamp: str = ""
    context_chunks_used: int = 0

