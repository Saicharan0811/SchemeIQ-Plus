# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Structure-Aware Document Chunker
Recursively chunks clean government scheme documents while preserving complete
provenance, section structure, eligibility clauses, financial metrics, and hashes.
"""

import hashlib
import os
import re
from typing import List, Optional

from src.rag.schemas import DocumentChunk, ProcessedDocument

DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))


def sha256_of_text(text: str) -> str:
    """Compute SHA-256 hash of a UTF-8 text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StructureAwareChunker:
    """
    Structure-aware recursive text chunker.
    Splits text hierarchically across section breaks, paragraphs, and sentence boundaries
    to preserve self-contained, context-rich chunks for dense embedding retrieval.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarchical split separators
        self.separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using decreasing hierarchy of separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if s.strip():
                if len(s) < self.chunk_size:
                    good_splits.append(s)
                else:
                    if new_separators:
                        other_splits = self._split_text_recursive(s, new_separators)
                        good_splits.extend(other_splits)
                    else:
                        good_splits.append(s)

        # Merge small splits up to chunk_size with chunk_overlap
        current_chunk: List[str] = []
        current_length = 0

        for split in good_splits:
            split_len = len(split)
            sep_len = len(separator) if current_chunk else 0

            if current_length + split_len + sep_len <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_len + sep_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                
                # Handle overlap from previous chunk
                overlap_accum: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_accum.insert(0, item)
                        overlap_len += len(item) + len(separator)
                    else:
                        break

                current_chunk = overlap_accum + [split]
                current_length = sum(len(x) for x in current_chunk) + (len(current_chunk) - 1) * len(separator)

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Split a ProcessedDocument into atomic DocumentChunk objects with complete provenance.
        """
        content = document.content.strip()
        if not content:
            return []

        doc_content_hash = document.processed_sha256 or sha256_of_text(content)

        # Perform recursive splitting
        raw_chunks = self._split_text_recursive(content, self.separators)

        # Filter out empty or whitespace-only chunks
        valid_chunk_texts = [c.strip() for c in raw_chunks if c.strip()]
        total_chunks = len(valid_chunk_texts)

        chunks: List[DocumentChunk] = []
        for idx, chunk_text in enumerate(valid_chunk_texts):
            chunk_hash = sha256_of_text(chunk_text)
            
            # Deterministic chunk ID: {scheme_id}_{document_id}_{chunk_index:03d}_{chunk_hash[:8]}
            chunk_id = f"{document.scheme_id}_{document.document_id}_{idx:03d}_{chunk_hash[:8]}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                scheme_id=document.scheme_id,
                scheme_name=document.scheme_name,
                source_filename=document.source_filename,
                official_url=document.official_url or "",
                source_authority=document.source_authority or "",
                document_title=document.document_title or "",
                document_type=document.document_type or "",
                recovery=document.recovery,
                chunk_index=idx,
                total_chunks_in_document=total_chunks,
                document_content_hash=doc_content_hash,
                chunk_content_hash=chunk_hash,
                source_type=document.source_type,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                character_count=len(chunk_text),
            )
            chunks.append(chunk)

        return chunks
