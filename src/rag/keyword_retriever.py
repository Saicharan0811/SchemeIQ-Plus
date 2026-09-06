# -*- coding: utf-8 -*-
"""
keyword_retriever.py — BM25-based keyword retrieval over the locked processed corpus.

Builds a BM25 index from the 16 processed .txt documents and provides
ranked retrieval at the chunk level. The index is built once at init time
and mirrors the exact documents in data/processed/documents/.

Design decisions:
- Index is built over the same chunks produced by chunker.py to ensure
  chunk IDs align with the ChromaDB vector store.
- Tokenization: lowercase, strip punctuation, remove stopwords.
- Returns KeywordResult objects (parallel to RetrievalResult) for easy RRF merging.
"""
from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
CORPUS_MANIFEST = PROJECT_ROOT / "data" / "processed" / "corpus_manifest.json"

# ---------------------------------------------------------------------------
# Minimal stopword list (English + common Indian scheme terminology stopwords)
# We keep scheme-specific terms like 'scheme', 'yojana', 'pradhan' as signals.
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "it", "its", "this", "that",
    "these", "those", "as", "up", "out", "if", "about", "into", "than",
    "then", "all", "each", "more", "also", "such", "not", "no", "so",
    "any", "which", "who", "whom", "there", "their", "they", "we", "i",
    "you", "he", "she", "him", "her", "our", "your", "my",
}


@dataclass
class KeywordResult:
    """A single BM25 retrieval result."""
    rank: int
    chunk_id: str
    scheme_id: str
    scheme_name: str
    source_filename: str
    bm25_score: float
    chunk_text: str
    text_preview: str = field(init=False)

    def __post_init__(self) -> None:
        self.text_preview = self.chunk_text[:200]


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return tokens


class KeywordRetriever:
    """
    BM25 retrieval engine over the 16 locked processed documents.

    Chunks are produced using the same parameters as chunker.py
    (CHUNK_SIZE=800, CHUNK_OVERLAP=120) to produce chunk IDs that align
    with the ChromaDB vector store.
    """

    def __init__(
        self,
        docs_dir: Path = PROCESSED_DOCS_DIR,
        manifest_path: Path = CORPUS_MANIFEST,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self._docs_dir = docs_dir
        self._manifest_path = manifest_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunks: list[dict] = []   # list of {chunk_id, scheme_id, scheme_name, source_filename, text}
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Load all processed documents, split into chunks, build BM25."""
        manifest = self._load_manifest()
        # Build lookup by processed_filename stem (e.g. "CT001_18_PM_KISAN_...") -> manifest entry
        filename_map: dict[str, dict] = {}
        for entry in manifest:
            proc_fn = entry.get("processed_filename", "")
            if proc_fn:
                stem = Path(proc_fn).stem
                filename_map[stem] = entry

        for txt_file in sorted(self._docs_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            stem = txt_file.stem
            meta = filename_map.get(stem, {})
            scheme_id = meta.get("scheme_id", "UNKNOWN")
            scheme_name = meta.get("scheme_name", "Unknown")

            chunks = self._split_text(text)
            for idx, chunk_text in enumerate(chunks):
                chunk_hash = self._short_hash(chunk_text)
                chunk_id = f"{scheme_id}_{stem}_{idx:03d}_{chunk_hash}"
                self._chunks.append({
                    "chunk_id": chunk_id,
                    "scheme_id": scheme_id,
                    "scheme_name": scheme_name,
                    "source_filename": txt_file.name,
                    "official_url": meta.get("official_url", ""),
                    "document_title": meta.get("document_title", ""),
                    "text": chunk_text,
                })
                self._tokenized_corpus.append(_tokenize(chunk_text))

        if not self._chunks:
            raise RuntimeError(f"No processed documents found in {self._docs_dir}")

        self._bm25 = BM25Okapi(self._tokenized_corpus)


    def _load_manifest(self) -> list[dict]:
        if not self._manifest_path.exists():
            return []
        with open(self._manifest_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
        """
        Simple recursive character splitter matching chunker.py behaviour.
        Splits on paragraphs, then sentences, then words.
        """
        text = text.strip()
        if not text:
            return []

        # If text fits in one chunk, return as-is
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        separators = ["\n\n", "\n", ". ", " "]

        def _split(t: str, sep_idx: int) -> list[str]:
            if len(t) <= chunk_size or sep_idx >= len(separators):
                return [t] if t.strip() else []
            sep = separators[sep_idx]
            parts = t.split(sep)
            result: list[str] = []
            current = ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        if len(current) > chunk_size:
                            result.extend(_split(current, sep_idx + 1))
                        else:
                            result.append(current.strip())
                    current = part
            if current.strip():
                if len(current) > chunk_size:
                    result.extend(_split(current, sep_idx + 1))
                else:
                    result.append(current.strip())
            return result

        raw_chunks = _split(text, 0)

        # Apply overlap: each chunk gets the tail of the previous chunk prepended
        if chunk_overlap <= 0 or len(raw_chunks) <= 1:
            return [c for c in raw_chunks if c.strip()]

        overlapped: list[str] = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev = raw_chunks[i - 1]
            tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            merged = tail + "\n" + raw_chunks[i]
            overlapped.append(merged[:chunk_size].strip())

        return [c for c in overlapped if c.strip()]

    @staticmethod
    def _short_hash(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        scheme_id_filter: Optional[str] = None,
    ) -> list[KeywordResult]:
        """
        Retrieve top_k chunks by BM25 score.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            scheme_id_filter: If set, restrict results to this scheme_id only.

        Returns:
            List of KeywordResult sorted by BM25 score descending.
        """
        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Build (score, idx) pairs, filter if needed
        scored = [
            (score, idx)
            for idx, score in enumerate(scores)
            if (scheme_id_filter is None or
                self._chunks[idx]["scheme_id"] == scheme_id_filter)
        ]

        # Sort descending by BM25 score, then ascending by idx for stability
        scored.sort(key=lambda x: (-x[0], x[1]))

        results: list[KeywordResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            chunk = self._chunks[idx]
            results.append(
                KeywordResult(
                    rank=rank,
                    chunk_id=chunk["chunk_id"],
                    scheme_id=chunk["scheme_id"],
                    scheme_name=chunk["scheme_name"],
                    source_filename=chunk["source_filename"],
                    bm25_score=float(score),
                    chunk_text=chunk["text"],
                )
            )

        return results

    def get_index_stats(self) -> dict:
        """Return index statistics for diagnostics."""
        scheme_counts: dict[str, int] = {}
        for c in self._chunks:
            scheme_counts[c["scheme_id"]] = scheme_counts.get(c["scheme_id"], 0) + 1
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len({c["source_filename"] for c in self._chunks}),
            "per_scheme_chunks": scheme_counts,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = KeywordRetriever()
    stats = retriever.get_index_stats()
    print(f"BM25 index built: {stats['total_chunks']} chunks, {stats['total_documents']} documents")
    print(f"Per-scheme: {stats['per_scheme_chunks']}\n")

    test_cases = [
        ("What benefits are provided under the MCH Kit Scheme?", "TS007"),
        ("What pension is available under Aasara?", "TS005"),
        ("Who is eligible for Rythu Bharosa?", "TS001"),
    ]

    for query, expected in test_cases:
        results = retriever.retrieve(query, top_k=5)
        top_scheme = results[0].scheme_id if results else "NONE"
        correct = expected in [r.scheme_id for r in results[:3]]
        correct_rank = next((r.rank for r in results if r.scheme_id == expected), None)
        status = "OK" if correct_rank == 1 else ("TOP-3" if correct else "FAIL")
        print(f"Query: {query}")
        print(f"  Expected: {expected} | Top-1: {top_scheme} | Correct rank: {correct_rank} | {status}")
        for r in results[:5]:
            marker = " <-- CORRECT" if r.scheme_id == expected else ""
            safe = r.text_preview[:80].encode("ascii", errors="replace").decode("ascii")
            print(f"    [{r.rank}] {r.scheme_id:<8} BM25={r.bm25_score:.4f}  {safe}{marker}")
        print()
