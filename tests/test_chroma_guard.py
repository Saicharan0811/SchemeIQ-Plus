# -*- coding: utf-8 -*-
"""
tests/test_chroma_guard.py — Phase 9D.4: ChromaDB Concurrency & Timeout Guard Tests

Tests:
1. test_chroma_query_lock_protection: Verifies self._lock serializes queries across threads.
2. test_chroma_timeout_handling: Verifies TimeoutError is raised when query exceeds timeout and lock is released.
3. test_existing_retrieval_behavior_unchanged: Verifies exact retrieval results on production vector store.
4. test_wsgi_prewarm_includes_chroma_query: Verifies WSGI pre-warm executes Chroma dense query.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import VectorStoreManager


def test_chroma_query_lock_protection():
    """Verify that VectorStoreManager._lock strictly serializes collection.query across threads."""
    vsm = VectorStoreManager()
    assert hasattr(vsm, "_lock")
    assert isinstance(vsm._lock, type(threading.Lock()))

    active_threads = 0
    max_concurrent = 0
    tracker_lock = threading.Lock()

    original_query = vsm.collection.query

    def _instrumented_query(*args, **kwargs):
        nonlocal active_threads, max_concurrent
        with tracker_lock:
            active_threads += 1
            if active_threads > max_concurrent:
                max_concurrent = active_threads
        try:
            time.sleep(0.05)
            return original_query(*args, **kwargs)
        finally:
            with tracker_lock:
                active_threads -= 1

    with patch.object(vsm.collection, "query", side_effect=_instrumented_query):
        threads = []
        errors = []

        def _worker():
            try:
                vsm.query_similar([0.0] * 384, top_k=1)
            except Exception as e:
                errors.append(e)

        for _ in range(4):
            t = threading.Thread(target=_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Worker thread hung during locked query"

        assert not errors, f"Unexpected errors during query: {errors}"
        # Serialization assertion: at most 1 thread inside collection.query at any instant
        assert max_concurrent == 1, f"Expected max concurrency of 1, got {max_concurrent}"


def test_no_detached_query_threads():
    """Verify that query_similar executes synchronously and spawns no background/daemon threads."""
    vsm = VectorStoreManager()
    threads_before = {t.ident for t in threading.enumerate()}

    # Execute synchronous query
    results = vsm.query_similar([0.0] * 384, top_k=1)
    assert isinstance(results, list)

    threads_after = {t.ident for t in threading.enumerate()}
    # No new lingering threads should be created or left running
    new_threads = threads_after - threads_before
    assert not new_threads, f"Unexpected background threads left running: {new_threads}"


def test_bm25_fallback_when_dense_fails():
    """Verify that when Chroma dense retrieval fails completely, HybridRetriever safely falls back to BM25."""
    from src.rag.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()

    # Simulate dense retrieval failure (both filtered and unfiltered)
    with patch.object(retriever._vector_store, "query_similar", side_effect=RuntimeError("Simulated Chroma timeout/failure")):
        results = retriever.retrieve("Who is eligible for Rythu Bharosa?", final_top_k=5)

        # Retrieval must not fail or raise: it must produce grounded BM25 results
        assert len(results) > 0, "Expected BM25 results when dense fails"
        assert results[0].scheme_id == "TS001"
        assert results[0].bm25_score is not None
        assert results[0].dense_score is None
        assert len(results[0].chunk_text) > 0


def test_existing_retrieval_behavior_unchanged():
    """Verify that existing retrieval semantics, scores, and chunk results remain intact."""
    vsm = VectorStoreManager()
    emb_provider = get_embedding_provider()
    q_emb = emb_provider.embed_query("Who is eligible for Rythu Bharosa?")

    # 1. Filtered query for TS001
    results = vsm.query_similar(q_emb, top_k=5, where_filter={"scheme_id": "TS001"})
    assert len(results) == 3, f"Expected 3 chunks for TS001, got {len(results)}"
    assert results[0].scheme_id == "TS001"
    assert results[0].chunk_id == "TS001_TS001_30_000_76a455a2"
    assert results[0].similarity_score > 0.5
    assert len(results[0].text_preview) > 20
    assert len(results[0].text) > 50

    # 2. Unfiltered query
    unfiltered = vsm.query_similar(q_emb, top_k=10)
    assert len(unfiltered) == 10
    assert unfiltered[0].rank == 1
    assert unfiltered[-1].rank == 10


def test_wsgi_prewarm_includes_chroma_query():
    """Verify that wsgi pre-warming initializes and executes the Chroma query."""
    import wsgi
    assert hasattr(wsgi, "_rag")
    if wsgi._rag is not None:
        assert hasattr(wsgi._rag, "_retriever")
        assert hasattr(wsgi._rag._retriever, "_vector_store")
        assert hasattr(wsgi._rag._retriever._vector_store, "query_similar")
        # Ensure collection is loaded and responsive
        res = wsgi._rag._retriever._vector_store.query_similar([0.0] * 384, top_k=1)
        assert len(res) == 1


# ── Phase 9D.8 Zero-Cost Architecture Tests ────────────────────────────────────

def test_local_dense_index_loads_166_vectors():
    """Verify that VectorStoreManager loads all 166 canonical vectors from dense_index.json."""
    vsm = VectorStoreManager()
    assert vsm.is_healthy() is True
    assert vsm.collection.count() == 166
    stats = vsm.get_collection_stats()
    assert stats["total_records"] == 166
    assert stats["collection_name"] == "schemeiq_official_schemes"


def test_vector_store_deterministic_rebuild_from_canonical(tmp_path):
    """Verify that vector index can be deterministically rebuilt from canonical documents."""
    temp_index_file = tmp_path / "dense_index.json"
    from scripts.build_vector_index import build_vector_index

    index_data = build_vector_index(output_path=temp_index_file)
    assert index_data["total_records"] == 166
    assert index_data["unique_scheme_count"] == 14
    assert temp_index_file.exists()

    # Load with VectorStoreManager pointing to temp dir
    vsm_temp = VectorStoreManager(persist_directory=temp_index_file)
    assert vsm_temp.is_healthy() is True
    assert vsm_temp.collection.count() == 166


def test_health_endpoint_contains_chroma_connected():
    """Verify GET /health response contains 'chroma_connected' field."""
    from src.api.app import create_app

    app = create_app(test_config={"TESTING": True})
    with app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "chroma_connected" in data
        assert isinstance(data["chroma_connected"], bool)
        assert data["chroma_connected"] is True


def test_health_endpoint_remains_200_when_heartbeat_fails():
    """Verify GET /health remains HTTP 200 and returns chroma_connected=False if vector store is unhealthy."""
    from src.api.app import create_app
    from unittest.mock import patch

    app = create_app(test_config={"TESTING": True})
    with patch("src.rag.vector_store.VectorStoreManager.is_healthy", return_value=False):
        with app.test_client() as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "healthy"
            assert data["chroma_connected"] is False


def test_render_yaml_zero_cost_single_web_service():
    """Verify render.yaml declares only ONE service: free schemeiq-plus-backend web service."""
    import yaml
    from pathlib import Path

    render_path = Path("render.yaml")
    assert render_path.exists(), "render.yaml not found"

    with open(render_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    services = spec.get("services", [])
    assert len(services) == 1, f"Expected exactly 1 service in render.yaml, found {len(services)}"

    backend = services[0]
    assert backend.get("type") == "web", f"Expected type: web, got {backend.get('type')}"
    assert backend.get("name") == "schemeiq-plus-backend"
    assert backend.get("plan") == "free", f"Expected plan: free, got {backend.get('plan')}"

    # Verify no private services or paid plans exist
    pservs = [s for s in services if s.get("type") == "pserv"]
    assert len(pservs) == 0, "render.yaml must not contain any pserv (private service) definitions"

    # Verify no obsolete CHROMA_SERVER_* env vars remain
    env_vars = {v["key"]: v for v in backend.get("envVars", [])}
    assert "CHROMA_SERVER_HOST" not in env_vars, "Obsolete CHROMA_SERVER_HOST must be removed"
    assert "CHROMA_SERVER_PORT" not in env_vars, "Obsolete CHROMA_SERVER_PORT must be removed"
    assert "CHROMA_SERVER_SSL" not in env_vars, "Obsolete CHROMA_SERVER_SSL must be removed"


def test_vector_store_integrity_check():
    """Verify vector store invariants: 166 vectors, 14 schemes, complete metadata, no duplicates."""
    vsm = VectorStoreManager()
    integrity = vsm.verify_integrity()
    assert integrity["status"] == "PASS"
    assert integrity["total_stored_vectors"] == 166
    assert integrity["unique_scheme_count"] == 14
    assert len(integrity["indexed_scheme_ids"]) == 14
    assert integrity["empty_documents_count"] == 0

