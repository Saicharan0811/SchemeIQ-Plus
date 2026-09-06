# -*- coding: utf-8 -*-
"""
SchemeIQ+ — RAG Ingestion Pipeline
Orchestrates structure-aware chunking, dense vector embedding, persistent ChromaDB indexing,
idempotent incremental caching, integrity validation, and retrieval smoke testing.
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.chunker import StructureAwareChunker
from src.rag.document_loader import DocumentLoader
from src.rag.embeddings import EmbeddingProvider, get_embedding_provider
from src.rag.schemas import (
    DocumentChunk,
    IngestionReport,
    ProcessedDocument,
    RetrievalResult,
    SmokeTestResult,
)
from src.rag.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
PROCESSED_DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_manifest.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store" / "chroma_db"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
INGESTION_REPORT_PATH = REPORTS_DIR / "rag_ingestion_report.json"

SMOKE_TEST_QUERIES = [
    (1, "Who is eligible for Rythu Bharosa?"),
    (2, "What health coverage is provided under Ayushman Bharat PM-JAY?"),
    (3, "What is the maximum project cost under PMEGP?"),
    (4, "What subsidy is available under PMEGP?"),
    (5, "What are the loan categories under MUDRA?"),
    (6, "Who can apply for Telangana ePASS scholarships?"),
    (7, "What benefits are provided under the MCH Kit Scheme?"),
    (8, "What pension is available under Aasara?"),
]


class RAGIngestionPipeline:
    """End-to-end production RAG ingestion and indexing pipeline."""

    def __init__(
        self,
        docs_dir: Path = PROCESSED_DOCS_DIR,
        manifest_path: Path = MANIFEST_PATH,
        vector_store_dir: Path = VECTOR_STORE_DIR,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.docs_dir = docs_dir
        self.manifest_path = manifest_path
        self.vector_store_dir = vector_store_dir
        self.loader = DocumentLoader(docs_dir=self.docs_dir, manifest_path=self.manifest_path)
        self.chunker = StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStoreManager(persist_directory=self.vector_store_dir)
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def run(self) -> IngestionReport:
        """Execute complete ingestion pipeline with idempotent vector indexing and smoke testing."""
        logger.info("=" * 70)
        logger.info("Starting SchemeIQ+ RAG Ingestion Pipeline")
        logger.info("=" * 70)

        # 1. Load Processed Documents
        logger.info(f"Loading documents from {self.docs_dir}...")
        documents: List[ProcessedDocument] = self.loader.load_documents()
        docs_loaded_count = len(documents)
        logger.info(f"Successfully loaded {docs_loaded_count} documents.")

        # 2. Chunk Documents
        logger.info("Chunking documents using structure-aware recursive chunker...")
        all_chunks: List[DocumentChunk] = []
        per_doc_chunks: Dict[str, int] = {}
        per_scheme_chunks: Dict[str, int] = {}

        for doc in documents:
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            per_doc_chunks[doc.processed_filename] = len(doc_chunks)
            per_scheme_chunks[doc.scheme_id] = per_scheme_chunks.get(doc.scheme_id, 0) + len(doc_chunks)

        total_chunks_created = len(all_chunks)
        logger.info(f"Total chunks created: {total_chunks_created}")

        # Check for chunk ID duplicates or empty chunks
        seen_ids = set()
        duplicate_chunk_ids = 0
        empty_chunks_rejected = 0
        valid_chunks: List[DocumentChunk] = []

        for chk in all_chunks:
            if not chk.text or not chk.text.strip():
                empty_chunks_rejected += 1
                continue
            if chk.chunk_id in seen_ids:
                duplicate_chunk_ids += 1
                continue
            seen_ids.add(chk.chunk_id)
            valid_chunks.append(chk)

        # 3. Idempotent Caching Check with ChromaDB
        existing_hashes = self.vector_store.get_existing_chunk_hashes()
        logger.info(f"Found {len(existing_hashes)} existing vectors in ChromaDB collection.")

        chunks_to_embed: List[DocumentChunk] = []
        chunks_skipped: List[DocumentChunk] = []

        for chk in valid_chunks:
            stored_hash = existing_hashes.get(chk.chunk_id)
            if stored_hash and stored_hash == chk.chunk_content_hash:
                chunks_skipped.append(chk)
            else:
                chunks_to_embed.append(chk)

        total_skipped = len(chunks_skipped)
        total_to_embed = len(chunks_to_embed)
        logger.info(f"Chunks to embed & index: {total_to_embed} | Unchanged chunks skipped: {total_skipped}")

        # 4. Generate Embeddings & Upsert
        failed_chunks = 0
        if chunks_to_embed:
            logger.info(f"Generating dense embeddings using provider '{self.embedding_provider.provider_name}' (model: {self.embedding_provider.model_name})...")
            texts = [c.text for c in chunks_to_embed]
            try:
                embeddings = self.embedding_provider.embed_documents(texts)
                logger.info(f"Generated {len(embeddings)} embedding vectors.")
                self.vector_store.upsert_chunks(chunks_to_embed, embeddings)
            except Exception as e:
                logger.error(f"Error during embedding generation or vector upsert: {e}")
                failed_chunks = len(chunks_to_embed)

        # 5. Integrity Verification
        integrity = self.vector_store.verify_integrity()
        logger.info(f"Vector Store Integrity Check: {integrity.get('status')}")

        indexed_schemes_count = integrity.get("unique_scheme_count", 0)

        # 6. Retrieval Smoke Testing
        logger.info("\n" + "=" * 70)
        logger.info("Running Retrieval Smoke Tests (8 Queries)")
        logger.info("=" * 70)

        smoke_test_results: List[SmokeTestResult] = []
        for q_id, q_text in SMOKE_TEST_QUERIES:
            q_emb = self.embedding_provider.embed_query(q_text)
            top_matches = self.vector_store.query_similar(q_emb, top_k=3)
            smoke_test_results.append(SmokeTestResult(query_id=q_id, query_text=q_text, top_matches=top_matches))

            top = top_matches[0] if top_matches else None
            if top:
                logger.info(f"Query {q_id}: '{q_text}'")
                logger.info(f"  -> Top 1: [{top.scheme_id} - {top.scheme_name}] Score: {top.similarity_score} | Source: {top.source_filename}")
                logger.info(f"     Preview: {top.text_preview[:120]}...\n")

        # 7. Build Ingestion Report
        integrity_status = "PASS"
        if docs_loaded_count != 16 or indexed_schemes_count != 14 or duplicate_chunk_ids > 0 or failed_chunks > 0 or integrity.get("status") != "PASS":
            integrity_status = "FAIL"

        report = IngestionReport(
            ingestion_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_directory=str(self.docs_dir),
            documents_expected=16,
            documents_loaded=docs_loaded_count,
            scheme_coverage_expected=14,
            scheme_coverage_indexed=indexed_schemes_count,
            total_chunks_created=total_chunks_created,
            total_chunks_embedded=total_to_embed,
            total_chunks_skipped=total_skipped,
            total_empty_chunks_rejected=empty_chunks_rejected,
            duplicate_chunk_ids=duplicate_chunk_ids,
            failed_chunks=failed_chunks,
            vector_store={
                "provider": "ChromaDB",
                "collection_name": self.vector_store.collection_name,
                "persist_directory": str(self.vector_store.persist_directory),
                "total_stored_vectors": integrity.get("total_stored_vectors", 0),
            },
            embedding={
                "provider": self.embedding_provider.provider_name,
                "model": self.embedding_provider.model_name,
                "dimension": self.embedding_provider.get_dimensions(),
            },
            integrity_status=integrity_status,
            per_document_chunks=per_doc_chunks,
            per_scheme_chunks=per_scheme_chunks,
            smoke_tests=smoke_test_results,
        )

        # Write report artifact
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(INGESTION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ingestion report to: {INGESTION_REPORT_PATH}")

        return report


def main():
    pipeline = RAGIngestionPipeline()
    report = pipeline.run()
    print("\n" + "=" * 70)
    print(f"INGESTION PIPELINE STATUS: {report.integrity_status}")
    print(f"Documents Loaded: {report.documents_loaded} / {report.documents_expected}")
    print(f"Schemes Indexed:  {report.scheme_coverage_indexed} / {report.scheme_coverage_expected}")
    print(f"Total Chunks:     {report.total_chunks_created} (Embedded: {report.total_chunks_embedded}, Skipped: {report.total_chunks_skipped})")
    print(f"Total Stored:     {report.vector_store['total_stored_vectors']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
