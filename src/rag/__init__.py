# -*- coding: utf-8 -*-
"""
SchemeIQ+ — RAG Package
Provides document loading, structure-aware chunking, embeddings, vector storage,
ingestion, Phase 4 hybrid retrieval, and Phase 5 grounded answer generation.
"""

from src.rag.chunker import StructureAwareChunker
from src.rag.document_loader import DocumentLoader
from src.rag.embeddings import EmbeddingProvider, get_embedding_provider
from src.rag.hybrid_retriever import HybridResult, HybridRetriever
from src.rag.ingest import RAGIngestionPipeline
from src.rag.keyword_retriever import KeywordResult, KeywordRetriever
from src.rag.query_detector import (
    SCHEME_ALIASES,
    detect_scheme,
    detect_scheme_with_details,
)
from src.rag.schemas import (
    DocumentChunk,
    GroundedAnswer,
    GroundingStatus,
    IngestionReport,
    ProcessedDocument,
    RetrievalResult,
    SmokeTestResult,
    SourceCitation,
)
from src.rag.vector_store import VectorStoreManager
from src.rag.answer_generator import (
    AnswerGenerator,
    BaseLLMProvider,
    OpenAILLMProvider,
    LocalTemplateLLMProvider,
    get_llm_provider,
    build_context_block,
    build_source_citations,
    build_prompt,
)
from src.rag.rag_service import RAGService

__all__ = [
    # Phase 3 — ingestion
    "DocumentLoader",
    "StructureAwareChunker",
    "EmbeddingProvider",
    "get_embedding_provider",
    "VectorStoreManager",
    "RAGIngestionPipeline",
    # Schemas
    "ProcessedDocument",
    "DocumentChunk",
    "RetrievalResult",
    "SmokeTestResult",
    "IngestionReport",
    # Phase 4 — hybrid retrieval
    "detect_scheme",
    "detect_scheme_with_details",
    "SCHEME_ALIASES",
    "KeywordRetriever",
    "KeywordResult",
    "HybridRetriever",
    "HybridResult",
    # Phase 5 schemas
    "GroundingStatus",
    "SourceCitation",
    "GroundedAnswer",
    # Phase 5 — answer generation
    "BaseLLMProvider",
    "OpenAILLMProvider",
    "LocalTemplateLLMProvider",
    "get_llm_provider",
    "AnswerGenerator",
    "build_context_block",
    "build_source_citations",
    "build_prompt",
    # Phase 5 — orchestration
    "RAGService",
]

