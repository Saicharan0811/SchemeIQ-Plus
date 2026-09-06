# -*- coding: utf-8 -*-
"""
rag_service.py — Phase 5H: Production-facing RAG Orchestrator

Provides a single clean interface: answer_query(query, top_k) -> GroundedAnswer

Pipeline:
  1. Validate query (non-empty, reasonable length)
  2. Detect scheme from query_detector
  3. Retrieve via HybridRetriever (NEVER bypassed)
  4. Validate retrieved results
  5. Build grounded context
  6. Generate answer via AnswerGenerator
  7. Attach deduplicated source citations
  8. Return GroundedAnswer
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from src.rag.answer_generator import AnswerGenerator, BaseLLMProvider, get_llm_provider
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.keyword_retriever import KeywordRetriever
from src.rag.query_detector import detect_scheme_with_details
from src.rag.schemas import GroundedAnswer, GroundingStatus
from src.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------
MIN_QUERY_CHARS: int = 5
MAX_QUERY_CHARS: int = 2000


def _validate_query(query: str) -> str | None:
    """
    Validate the query string.
    Returns None if valid, or an error message string if invalid.
    """
    if not query or not query.strip():
        return "Query is empty."
    if len(query.strip()) < MIN_QUERY_CHARS:
        return f"Query is too short (minimum {MIN_QUERY_CHARS} characters)."
    if len(query.strip()) > MAX_QUERY_CHARS:
        return f"Query is too long (maximum {MAX_QUERY_CHARS} characters)."
    return None


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class RAGService:
    """
    Production-facing RAG orchestration service.

    Wraps HybridRetriever + AnswerGenerator into a single answer_query() call.

    NOTE: Dense-only retrieval is NOT used here. HybridRetriever is always invoked.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        dense_candidates: int = 20,
        keyword_candidates: int = 20,
        rrf_k: int = 60,
        max_answer_chunks: int = 5,
        max_context_chars: int = 6000,
        max_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> None:
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            keyword_retriever=keyword_retriever,
            dense_candidates=dense_candidates,
            keyword_candidates=keyword_candidates,
            rrf_k=rrf_k,
            final_top_k=max_answer_chunks,
        )
        self._generator = AnswerGenerator(
            llm_provider=llm_provider or get_llm_provider(allow_fallback=allow_fallback),
            max_context_chars=max_context_chars,
            max_chunks=max_answer_chunks,
            max_tokens=max_tokens,
            allow_fallback=allow_fallback,
        )

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> GroundedAnswer:
        """
        Full RAG pipeline: retrieve → generate → return GroundedAnswer.

        Args:
            query: User's natural language question.
            top_k: Maximum number of retrieved chunks to pass to the generator.

        Returns:
            GroundedAnswer with answer, sources, and grounding metadata.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # ---- Step 1: Validate query ----
        error = _validate_query(query)
        if error:
            logger.warning(f"Query validation failed: {error}")
            return GroundedAnswer(
                question=query,
                answer=f"Query validation error: {error}",
                grounding_status=GroundingStatus.GENERATION_FAILED,
                grounding_note=error,
                generation_timestamp=timestamp,
            )

        # ---- Step 2: Detect scheme ----
        detection = detect_scheme_with_details(query)
        detected_scheme_id = detection["scheme_id"] if detection["detected"] else None
        detected_alias = detection.get("matched_alias")

        logger.info(
            "RAGService.answer_query | query=%r | detected_scheme=%s (alias=%r)",
            query[:80],
            detected_scheme_id,
            detected_alias,
        )

        # ---- Step 3: Retrieve via HybridRetriever ----
        try:
            retrieved_chunks = self._retriever.retrieve(query, final_top_k=top_k)
        except Exception as e:
            logger.error(f"HybridRetriever failed: {e}")
            return GroundedAnswer(
                question=query,
                answer=(
                    "A technical error occurred during document retrieval. "
                    "Please try again or contact support."
                ),
                detected_scheme_id=detected_scheme_id,
                grounding_status=GroundingStatus.GENERATION_FAILED,
                grounding_note=f"Retrieval failed: {e}",
                generation_timestamp=timestamp,
            )

        # ---- Step 4: Validate retrieval ----
        if not retrieved_chunks:
            logger.warning("HybridRetriever returned no results for query: %r", query[:80])
            return GroundedAnswer(
                question=query,
                answer=(
                    "I could not find sufficient information in the official SchemeIQ+ sources "
                    "to answer that question. No relevant official documents were retrieved."
                ),
                detected_scheme_id=detected_scheme_id,
                grounding_status=GroundingStatus.INSUFFICIENT_CONTEXT,
                grounding_note="HybridRetriever returned no results.",
                generation_timestamp=timestamp,
            )

        # ---- Step 5+6: Generate grounded answer ----
        # Resolve scheme_name from detected chunks
        detected_scheme_name: Optional[str] = None
        if detected_scheme_id:
            for chunk in retrieved_chunks:
                if chunk.scheme_id == detected_scheme_id:
                    detected_scheme_name = chunk.scheme_name
                    break

        grounded = self._generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            detected_scheme_id=detected_scheme_id,
            detected_scheme_name=detected_scheme_name,
            detection=detection,
        )

        logger.info(
            "RAGService answer | status=%s | sources=%d | chunks_used=%d",
            grounded.grounding_status,
            len(grounded.sources),
            grounded.context_chunks_used,
        )

        return grounded
