# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Lightweight Deterministic Vector Store Manager
Replaces heavy ChromaDB / SQLite / Tokio runtime with a deterministic,
in-memory vector store powered by NumPy cosine similarity.

Storage:
  data/vector_store/dense_index.json
  (Precomputed 166 chunk vectors, text, and metadata).

Zero-cost, zero-deadlock, zero-process architecture for Render Free tier.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.schemas import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/vector_store")
DEFAULT_INDEX_FILE = DEFAULT_INDEX_DIR / "dense_index.json"
DEFAULT_COLLECTION_NAME = "schemeiq_official_schemes"
EXPECTED_VECTOR_COUNT = 166


class _LocalCollectionAdapter:
    """
    Lightweight backward-compatible adapter exposing Chroma-like Collection methods
    (.count(), .get(), .query(), .upsert()) over the local VectorStoreManager.
    """

    def __init__(self, manager: VectorStoreManager):
        self._manager = manager

    def count(self) -> int:
        return len(self._manager._ids)

    def get(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        result: Dict[str, Any] = {
            "ids": list(self._manager._ids),
        }
        if "documents" in include:
            result["documents"] = list(self._manager._documents)
        if "metadatas" in include:
            result["metadatas"] = [dict(m) for m in self._manager._metadatas]
        if "embeddings" in include:
            result["embeddings"] = self._manager._embeddings.tolist() if self._manager._embeddings is not None else []
        return result

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._manager._execute_collection_query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self._manager._apply_upsert(ids, embeddings, documents, metadatas)


class VectorStoreManager:
    """
    Manages lightweight in-memory vector storage for official scheme chunks.
    Uses pure NumPy cosine similarity for sub-millisecond, deadlock-free search.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        query_timeout: float = 5.0,
        auto_load: bool = True,
    ):
        self.persist_directory = persist_directory or DEFAULT_INDEX_DIR
        self.collection_name = collection_name
        self._lock = threading.Lock()
        self._query_timeout = query_timeout

        # In-memory index structures
        self._ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._norms: Optional[np.ndarray] = None

        # Adapter for code that calls .collection.*
        self.collection = _LocalCollectionAdapter(self)

        if auto_load:
            self._load_or_build_index()

    # ---------------------------------------------------------------------------
    # Index Persistence & Initialization
    # ---------------------------------------------------------------------------

    def _get_index_file(self) -> Path:
        if self.persist_directory.is_file():
            return self.persist_directory
        return self.persist_directory / "dense_index.json"

    def _load_or_build_index(self) -> None:
        """Load from dense_index.json if available, or build deterministically."""
        index_file = self._get_index_file()
        if index_file.exists():
            try:
                self._load_from_json(index_file)
                logger.info(
                    "Loaded %d vector records from %s", len(self._ids), index_file
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to load %s: %s. Attempting deterministic rebuild...",
                    index_file, e,
                )

        # Auto-rebuild fallback if file missing or corrupted
        try:
            self._build_and_save_index(index_file)
        except Exception as e:
            logger.error("Failed to build vector index: %s", e)

    def _load_from_json(self, path: Path) -> None:
        """Parse dense_index.json into memory arrays."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records", [])
        self._ids = [r["chunk_id"] for r in records]
        self._documents = [r["text"] for r in records]
        self._metadatas = [r["metadata"] for r in records]

        embs = [r["embedding"] for r in records]
        if embs:
            self._embeddings = np.array(embs, dtype=np.float32)
            self._norms = np.linalg.norm(self._embeddings, axis=1)
            # Replace zero norms with 1e-9 to avoid division by zero
            self._norms[self._norms == 0] = 1e-9
        else:
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._norms = np.empty((0,), dtype=np.float32)

    def _build_and_save_index(self, output_path: Path) -> None:
        """Deterministically rebuild from canonical documents."""
        logger.info("Building vector index deterministically from canonical corpus...")
        from scripts.build_vector_index import build_vector_index
        build_vector_index(output_path=output_path)
        self._load_from_json(output_path)

    # ---------------------------------------------------------------------------
    # Health & Diagnostics
    # ---------------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if index contains expected 166 vectors and is ready."""
        try:
            return len(self._ids) == EXPECTED_VECTOR_COUNT and self._embeddings is not None
        except Exception:
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return collection level statistics."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_records": len(self._ids),
        }

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify vector store integrity invariants."""
        count = len(self._ids)
        if count == 0:
            return {"status": "EMPTY", "issues": ["Collection is empty"]}

        issues = []
        if len(self._ids) != len(set(self._ids)):
            issues.append(f"Duplicate IDs detected: {len(self._ids)} total vs {len(set(self._ids))} unique")

        empty_docs = sum(1 for d in self._documents if not d or not d.strip())
        if empty_docs > 0:
            issues.append(f"Found {empty_docs} empty documents in vector database")

        required_keys = [
            "scheme_id", "scheme_name", "source_filename", "official_url",
            "chunk_id", "chunk_index", "chunk_content_hash"
        ]
        missing_metadata_count = 0
        for m in self._metadatas:
            if not m or any(k not in m for k in required_keys):
                missing_metadata_count += 1

        if missing_metadata_count > 0:
            issues.append(f"Found {missing_metadata_count} records with missing required metadata keys")

        unique_schemes = sorted(list({m.get("scheme_id") for m in self._metadatas if m and m.get("scheme_id")}))

        return {
            "status": "PASS" if not issues else "FAIL",
            "total_stored_vectors": count,
            "unique_scheme_count": len(unique_schemes),
            "indexed_scheme_ids": unique_schemes,
            "empty_documents_count": empty_docs,
            "issues": issues,
        }

    def get_existing_chunk_hashes(self) -> Dict[str, str]:
        """Mapping of chunk_id to stored chunk_content_hash."""
        id_hash_map: Dict[str, str] = {}
        for chunk_id, meta in zip(self._ids, self._metadatas):
            id_hash_map[chunk_id] = meta.get("chunk_content_hash", "") if meta else ""
        return id_hash_map

    # ---------------------------------------------------------------------------
    # Dense Retrieval (NumPy Cosine Similarity)
    # ---------------------------------------------------------------------------

    def _execute_collection_query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Internal vector search returning raw Chroma-compatible query results."""
        if not query_embeddings or len(query_embeddings) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        if self._embeddings is None or len(self._ids) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # 1. Filter candidates by where_filter if present
        candidate_indices: List[int]
        if where:
            candidate_indices = []
            for idx, meta in enumerate(self._metadatas):
                matches = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        matches = False
                        break
                if matches:
                    candidate_indices.append(idx)
            if not candidate_indices:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        else:
            candidate_indices = list(range(len(self._ids)))

        # 2. Compute cosine similarities
        cand_idx_arr = np.array(candidate_indices, dtype=np.int64)
        cand_embeddings = self._embeddings[cand_idx_arr]
        cand_norms = self._norms[cand_idx_arr]

        q = np.array(query_embeddings[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q))

        if q_norm == 0.0:
            similarities = np.zeros(len(candidate_indices), dtype=np.float32)
        else:
            dots = np.dot(cand_embeddings, q)
            similarities = dots / (cand_norms * q_norm)

        # 3. Sort descending by similarity
        actual_top_k = min(n_results, len(candidate_indices))
        sorted_order = np.argsort(-similarities)[:actual_top_k]

        ids = [self._ids[candidate_indices[ord_idx]] for ord_idx in sorted_order]
        documents = [self._documents[candidate_indices[ord_idx]] for ord_idx in sorted_order]
        metadatas = [self._metadatas[candidate_indices[ord_idx]] for ord_idx in sorted_order]
        distances = [round(max(0.0, 1.0 - float(similarities[ord_idx])), 4) for ord_idx in sorted_order]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Perform dense similarity search using query embedding vector.
        Thread-safe: queries are serialized with self._lock synchronously.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        with self._lock:
            results = self.collection.query(**kwargs)

        retrieval_results: List[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (chunk_id, doc_text, meta, distance) in enumerate(zip(ids, documents, metadatas, distances), 1):
            similarity = max(0.0, 1.0 - distance)
            meta = meta or {}

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

    # ---------------------------------------------------------------------------
    # Mutation & Ingestion
    # ---------------------------------------------------------------------------

    def _apply_upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Internal upsert helper updating memory and disk JSON."""
        with self._lock:
            existing_index_map = {cid: idx for idx, cid in enumerate(self._ids)}

            new_records = []
            for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                if cid in existing_index_map:
                    idx = existing_index_map[cid]
                    self._documents[idx] = doc
                    self._metadatas[idx] = meta
                    self._embeddings[idx] = np.array(emb, dtype=np.float32)
                else:
                    existing_index_map[cid] = len(self._ids)
                    self._ids.append(cid)
                    self._documents.append(doc)
                    self._metadatas.append(meta)
                    new_records.append(emb)

            if new_records:
                new_arr = np.array(new_records, dtype=np.float32)
                if self._embeddings is None or len(self._embeddings) == 0:
                    self._embeddings = new_arr
                else:
                    self._embeddings = np.vstack([self._embeddings, new_arr])

            if self._embeddings is not None and len(self._embeddings) > 0:
                self._norms = np.linalg.norm(self._embeddings, axis=1)
                self._norms[self._norms == 0] = 1e-9

            # Save updated index back to JSON
            records = [
                {
                    "chunk_id": cid,
                    "text": doc,
                    "embedding": [round(float(v), 6) for v in emb],
                    "metadata": meta,
                }
                for cid, doc, meta, emb in zip(
                    self._ids, self._documents, self._metadatas, self._embeddings.tolist()
                )
            ]
            index_data = {
                "version": "1.0.0",
                "collection_name": self.collection_name,
                "total_records": len(records),
                "records": records,
            }
            index_file = self._get_index_file()
            index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert a batch of chunks into the local vector index."""
        if not chunks:
            return 0
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_chroma_metadata() for chunk in chunks]
        self._apply_upsert(ids, embeddings, documents, metadatas)
        return len(chunks)
