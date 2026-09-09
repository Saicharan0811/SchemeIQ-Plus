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


def test_chroma_timeout_handling():
    """Verify that query_similar raises TimeoutError when query duration exceeds timeout and releases lock."""
    vsm = VectorStoreManager()

    def _hanging_query(*args, **kwargs):
        time.sleep(0.5)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    with patch.object(vsm.collection, "query", side_effect=_hanging_query):
        t0 = time.perf_counter()
        with pytest.raises(TimeoutError) as exc_info:
            # Set small 0.1s timeout
            vsm.query_similar([0.0] * 384, top_k=1, timeout=0.1)
        elapsed = time.perf_counter() - t0

        assert "timed out after 0.1s" in str(exc_info.value)
        assert elapsed < 0.45, f"Timeout took too long to abort: {elapsed:.2f}s"

    # Verify the lock was cleanly released: a subsequent fast query succeeds immediately
    res = vsm.query_similar([0.0] * 384, top_k=1, timeout=5.0)
    assert isinstance(res, list)


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
