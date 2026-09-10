# -*- coding: utf-8 -*-
"""
hybrid_retriever.py — Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion.

Architecture:
    Query
      │
      ├─ [Query Detector] detect scheme_id from aliases (deterministic)
      │
      ├─ [Dense Retrieval] ChromaDB cosine similarity (top dense_candidates)
      │    └── optional: metadata filter by scheme_id when detected
      │
      ├─ [Keyword Retrieval] BM25 over processed documents (top keyword_candidates)
      │    └── optional: scheme_id filter when detected
      │
      ├─ [RRF Merge] Reciprocal Rank Fusion (rrf_k=60)
      │
      └─ [Top K] return final_top_k results

Config (all overridable):
    dense_candidates    = 20   (candidates pulled from ChromaDB)
    keyword_candidates  = 20   (candidates pulled from BM25)
    rrf_k               = 60   (RRF smoothing constant)
    final_top_k         = 5    (results returned to caller)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.rag.query_detector import detect_scheme_with_details
from src.rag.keyword_retriever import KeywordRetriever
from src.rag.vector_store import VectorStoreManager
from src.rag.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_DENSE_CANDIDATES: int = 20
DEFAULT_KEYWORD_CANDIDATES: int = 20
DEFAULT_RRF_K: int = 60
DEFAULT_FINAL_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class HybridResult:
    """A single result from hybrid retrieval."""
    rank: int
    chunk_id: str
    scheme_id: str
    scheme_name: str
    source_filename: str
    rrf_score: float
    dense_rank: Optional[int]       # rank in dense-only results (None if not retrieved)
    keyword_rank: Optional[int]     # rank in BM25 results (None if not retrieved)
    dense_score: Optional[float]    # cosine similarity score (None if not retrieved)
    bm25_score: Optional[float]     # BM25 raw score (None if not retrieved)
    chunk_text: str
    text_preview: str = field(init=False)
    detection: dict = field(default_factory=dict)
    official_url: str = ""
    document_title: str = ""

    def __post_init__(self) -> None:
        self.text_preview = self.chunk_text[:200]


# ---------------------------------------------------------------------------
# RRF utility
# ---------------------------------------------------------------------------
def _rrf_merge(
    dense_results: list,
    keyword_results: list,
    rrf_k: int = 60,
) -> list[tuple[str, float, Optional[int], Optional[int], Optional[float], Optional[float]]]:
    """
    Reciprocal Rank Fusion of dense and keyword ranked lists.

    Returns list of (chunk_id, rrf_score, dense_rank, keyword_rank, dense_score, bm25_score)
    sorted by rrf_score descending.
    """
    scores: dict[str, dict] = {}

    for result in dense_results:
        cid = result.chunk_id
        if cid not in scores:
            scores[cid] = {
                "rrf": 0.0,
                "dense_rank": None,
                "keyword_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }
        scores[cid]["rrf"] += 1.0 / (rrf_k + result.rank)
        scores[cid]["dense_rank"] = result.rank
        scores[cid]["dense_score"] = result.similarity_score

    for result in keyword_results:
        cid = result.chunk_id
        if cid not in scores:
            scores[cid] = {
                "rrf": 0.0,
                "dense_rank": None,
                "keyword_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }
        scores[cid]["rrf"] += 1.0 / (rrf_k + result.rank)
        scores[cid]["keyword_rank"] = result.rank
        scores[cid]["bm25_score"] = result.bm25_score

    merged = sorted(scores.items(), key=lambda x: -x[1]["rrf"])
    return [
        (
            cid,
            info["rrf"],
            info["dense_rank"],
            info["keyword_rank"],
            info["dense_score"],
            info["bm25_score"],
        )
        for cid, info in merged
    ]


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Hybrid dense + BM25 retriever with deterministic scheme detection and RRF.

    Usage::

        retriever = HybridRetriever()
        results = retriever.retrieve("What pension is available under Aasara?")
        for r in results:
            print(r.rank, r.scheme_id, r.rrf_score, r.text_preview)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        dense_candidates: int = DEFAULT_DENSE_CANDIDATES,
        keyword_candidates: int = DEFAULT_KEYWORD_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        self._vector_store = vector_store or VectorStoreManager()
        self._keyword_retriever = keyword_retriever or KeywordRetriever()
        self._emb_provider = get_embedding_provider()
        self.dense_candidates = dense_candidates
        self.keyword_candidates = keyword_candidates
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k

        # Build a lookup from chunk_id → full chunk text (from keyword index)
        self._chunk_text_map: dict[str, str] = {}
        for c in self._keyword_retriever._chunks:
            self._chunk_text_map[c["chunk_id"]] = c["text"]

        # Build chunk_id → meta lookup from keyword index chunks list
        self._chunk_meta_map: dict[str, dict] = {
            c["chunk_id"]: c for c in self._keyword_retriever._chunks
        }

    def retrieve(
        self,
        query: str,
        final_top_k: Optional[int] = None,
        use_scheme_filter: bool = True,
    ) -> list[HybridResult]:
        """
        Run hybrid retrieval for the given query.

        Args:
            query: Natural language query.
            final_top_k: Override final top-k (default: self.final_top_k).
            use_scheme_filter: If True and a scheme is detected, apply metadata
                               filter to both dense and keyword retrieval.

        Returns:
            List of HybridResult sorted by RRF score descending.
        """
        top_k = final_top_k if final_top_k is not None else self.final_top_k
        detection = detect_scheme_with_details(query)
        detected_scheme = detection["scheme_id"] if detection["detected"] else None

        logger.debug(
            "Query='%s' | Detection=%s | filter=%s",
            query, detection, use_scheme_filter and detected_scheme,
        )

        scheme_filter = detected_scheme if use_scheme_filter else None

        # ---- Dense retrieval ----
        t_emb_start = time.perf_counter()
        q_emb = self._emb_provider.embed_query(query)
        logger.info(
            "HybridRetriever: query embedded in %.3fs (dim=%d)",
            time.perf_counter() - t_emb_start,
            len(q_emb),
        )
        dense_top_k = self.dense_candidates

        t_dense_start = time.perf_counter()
        dense_raw: list = []
        if scheme_filter:
            # Use ChromaDB where_filter to restrict to detected scheme
            try:
                dense_raw = self._vector_store.query_similar(
                    q_emb,
                    top_k=dense_top_k,
                    where_filter={"scheme_id": scheme_filter},
                )
            except Exception as e:
                logger.error(f"Filtered dense query failed: {e}. Falling back to unfiltered query.")
                # Fallback to unfiltered if filter fails
                try:
                    dense_raw = self._vector_store.query_similar(q_emb, top_k=dense_top_k)
                except Exception as e2:
                    logger.error(f"Unfiltered dense query also failed: {e2}. Falling back to BM25-only retrieval.")
                    dense_raw = []
        else:
            try:
                dense_raw = self._vector_store.query_similar(q_emb, top_k=dense_top_k)
            except Exception as e:
                logger.error(f"Dense query failed: {e}. Falling back to BM25-only retrieval.")
                dense_raw = []
        logger.info(
            "HybridRetriever: Chroma dense retrieval completed in %.3fs (%d chunks, filter=%s)",
            time.perf_counter() - t_dense_start,
            len(dense_raw),
            scheme_filter,
        )

        # ---- BM25 keyword retrieval ----
        t_bm25_start = time.perf_counter()
        keyword_raw = self._keyword_retriever.retrieve(
            query,
            top_k=self.keyword_candidates,
            scheme_id_filter=scheme_filter,
        )
        logger.info(
            "HybridRetriever: BM25 retrieval completed in %.3fs (%d chunks, filter=%s)",
            time.perf_counter() - t_bm25_start,
            len(keyword_raw),
            scheme_filter,
        )

        # ---- RRF merge ----
        t_rrf_start = time.perf_counter()
        merged = _rrf_merge(dense_raw, keyword_raw, rrf_k=self.rrf_k)
        logger.info(
            "HybridRetriever: RRF merge completed in %.3fs (%d candidates merged)",
            time.perf_counter() - t_rrf_start,
            len(merged),
        )

        # ---- Build HybridResult list ----
        results: list[HybridResult] = []
        # For dense results, build chunk_id → scheme/name lookup
        dense_meta: dict[str, dict] = {}
        for dr in dense_raw:
            dense_meta[dr.chunk_id] = {
                "scheme_id": dr.scheme_id,
                "scheme_name": dr.scheme_name,
                "source_filename": dr.source_filename,
                "official_url": getattr(dr, "official_url", ""),
                "document_title": getattr(dr, "document_title", ""),
                "text": getattr(dr, "text_preview", "")[:800],
            }

        for rank, (chunk_id, rrf_score, d_rank, k_rank, d_score, b_score) in enumerate(
            merged[:top_k], start=1
        ):
            # Prefer keyword chunk text (full text) over dense text_preview
            chunk_text = self._chunk_text_map.get(chunk_id, "")
            meta = (
                self._chunk_meta_map.get(chunk_id)
                or dense_meta.get(chunk_id, {})
            )
            # Merge fields from both sources if one has missing official_url
            official_url = meta.get("official_url") or dense_meta.get(chunk_id, {}).get("official_url", "")
            document_title = meta.get("document_title") or dense_meta.get(chunk_id, {}).get("document_title", "")

            results.append(
                HybridResult(
                    rank=rank,
                    chunk_id=chunk_id,
                    scheme_id=meta.get("scheme_id", "UNKNOWN"),
                    scheme_name=meta.get("scheme_name", "Unknown"),
                    source_filename=meta.get("source_filename", ""),
                    rrf_score=rrf_score,
                    dense_rank=d_rank,
                    keyword_rank=k_rank,
                    dense_score=d_score,
                    bm25_score=b_score,
                    chunk_text=chunk_text,
                    detection=detection,
                    official_url=official_url,
                    document_title=document_title,
                )
            )

        return results

    def retrieve_as_dicts(self, query: str, **kwargs) -> list[dict]:
        """Convenience wrapper returning list of dicts for JSON serialisation."""
        results = self.retrieve(query, **kwargs)
        return [
            {
                "rank": r.rank,
                "chunk_id": r.chunk_id,
                "scheme_id": r.scheme_id,
                "scheme_name": r.scheme_name,
                "source_filename": r.source_filename,
                "rrf_score": round(r.rrf_score, 6),
                "dense_rank": r.dense_rank,
                "keyword_rank": r.keyword_rank,
                "dense_score": round(r.dense_score, 4) if r.dense_score is not None else None,
                "bm25_score": round(r.bm25_score, 4) if r.bm25_score is not None else None,
                "detection": r.detection,
                "text_preview": r.text_preview[:200],
            }
            for r in results
        ]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    retriever = HybridRetriever()

    test_cases = [
        ("Who is eligible for Rythu Bharosa?",                              "TS001"),
        ("What health coverage is provided under Ayushman Bharat PM-JAY?", "CT002"),
        ("What is the maximum project cost under PMEGP?",                  "CT005"),
        ("What subsidy is available under PMEGP?",                         "CT005"),
        ("What are the loan categories under MUDRA?",                      "CT004"),
        ("Who can apply for Telangana ePASS scholarships?",                "TS006"),
        ("What benefits are provided under the MCH Kit Scheme?",           "TS007"),
        ("What pension is available under Aasara?",                        "TS005"),
    ]

    print(f"{'#':<4}{'Query':<62}{'Exp':<8}{'Top-1':<8}{'Rank':<6}{'OK'}")
    print("-" * 95)
    passed = 0
    for i, (query, expected) in enumerate(test_cases, 1):
        results = retriever.retrieve(query, final_top_k=10)
        top1 = results[0].scheme_id if results else "NONE"
        correct_rank = next((r.rank for r in results if r.scheme_id == expected), None)
        ok = top1 == expected
        if ok:
            passed += 1
        q_short = query[:60]
        print(f"{i:<4}{q_short:<62}{expected:<8}{top1:<8}{str(correct_rank):<6}{'OK' if ok else 'FAIL'}")

    print(f"\nTop-1 Accuracy: {passed}/{len(test_cases)}")
