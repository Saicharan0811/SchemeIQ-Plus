# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Persistent ChromaDB Vector Store Manager
Handles collection lifecycle, idempotent upserts, metadata indexing, and dense similarity search.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.rag.schemas import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path("data/vector_store/chroma_db")
DEFAULT_COLLECTION_NAME = "schemeiq_official_schemes"


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage for official scheme chunks."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        query_timeout: float = 5.0,
    ):
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIR
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._query_timeout = query_timeout

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

        # Get or create collection with cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "SchemeIQ+ Official Government Scheme Corpus"}
        )
        logger.info(f"Initialized ChromaDB collection '{self.collection_name}' at: {self.persist_directory}")

    def get_existing_chunk_hashes(self) -> Dict[str, str]:
        """
        Retrieve a mapping of all existing chunk_ids to their stored chunk_content_hash.
        Used for idempotent incremental ingestion.
        """
        count = self.collection.count()
        if count == 0:
            return {}

        results = self.collection.get(
            include=["metadatas"]
        )

        id_hash_map: Dict[str, str] = {}
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        for chunk_id, meta in zip(ids, metadatas):
            if meta and "chunk_content_hash" in meta:
                id_hash_map[chunk_id] = meta["chunk_content_hash"]
            else:
                id_hash_map[chunk_id] = ""

        return id_hash_map

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert a batch of chunks and their pre-computed embeddings into ChromaDB.
        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_chroma_metadata() for chunk in chunks]

        # Batch upsert into collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Successfully upserted {len(chunks)} chunks into '{self.collection_name}'")
        return len(chunks)

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Perform dense similarity search using query embedding vector.
        Thread-safe: queries are serialized with self._lock and bounded by timeout.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        effective_timeout = timeout if timeout is not None else self._query_timeout

        with self._lock:
            if effective_timeout and effective_timeout > 0:
                res_box: List[Any] = [None]
                err_box: List[Optional[Exception]] = [None]

                def _query_target():
                    try:
                        res_box[0] = self.collection.query(**kwargs)
                    except Exception as e:
                        err_box[0] = e

                t = threading.Thread(target=_query_target, daemon=True)
                t.start()
                t.join(timeout=effective_timeout)

                if t.is_alive():
                    raise TimeoutError(f"ChromaDB query timed out after {effective_timeout:.1f}s")

                if err_box[0] is not None:
                    raise err_box[0]

                results = res_box[0]
            else:
                results = self.collection.query(**kwargs)

        retrieval_results: List[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (chunk_id, doc_text, meta, distance) in enumerate(zip(ids, documents, metadatas, distances), 1):
            # In cosine distance: similarity = 1.0 - distance
            similarity = max(0.0, 1.0 - distance)
            meta = meta or {}

            # Generate short 150-char preview
            preview = doc_text.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."

            res = RetrievalResult(
                rank=rank,
                chunk_id=chunk_id,
                similarity_score=round(similarity, 4),
                distance=round(distance, 4),
                scheme_id=meta.get("scheme_id", "UNKNOWN"),
                scheme_name=meta.get("scheme_name", "Unknown Scheme"),
                source_filename=meta.get("source_filename", ""),
                official_url=meta.get("official_url", ""),
                source_authority=meta.get("source_authority", ""),
                document_title=meta.get("document_title", ""),
                text_preview=preview,
                text=doc_text,
            )
            retrieval_results.append(res)

        return retrieval_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return collection level statistics and health metrics."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_records": count,
        }

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify that:
        - No empty documents exist
        - No duplicate IDs exist
        - All chunks contain required metadata keys
        """
        count = self.collection.count()
        if count == 0:
            return {"status": "EMPTY", "issues": ["Collection is empty"]}

        results = self.collection.get(include=["documents", "metadatas"])
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        issues = []
        if len(ids) != len(set(ids)):
            issues.append(f"Duplicate IDs detected: {len(ids)} total vs {len(set(ids))} unique")

        empty_docs = sum(1 for d in documents if not d or not d.strip())
        if empty_docs > 0:
            issues.append(f"Found {empty_docs} empty documents in vector database")

        required_keys = [
            "scheme_id", "scheme_name", "source_filename", "official_url",
            "chunk_id", "chunk_index", "chunk_content_hash"
        ]
        missing_metadata_count = 0
        for m in metadatas:
            if not m or any(k not in m for k in required_keys):
                missing_metadata_count += 1

        if missing_metadata_count > 0:
            issues.append(f"Found {missing_metadata_count} records with missing required metadata keys")

        unique_schemes = sorted(list({m.get("scheme_id") for m in metadatas if m and m.get("scheme_id")}))

        return {
            "status": "PASS" if not issues else "FAIL",
            "total_stored_vectors": count,
            "unique_scheme_count": len(unique_schemes),
            "indexed_scheme_ids": unique_schemes,
            "empty_documents_count": empty_docs,
            "issues": issues,
        }
