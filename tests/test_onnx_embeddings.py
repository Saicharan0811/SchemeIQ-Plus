# -*- coding: utf-8 -*-
"""
tests/test_onnx_embeddings.py — Comprehensive Validation for Phase 9D.1 ONNX Embedding Backend

Verifies:
1. Shape and dimensionality (384-dim, (1, 384))
2. Numerical validity (no NaN/Inf, unit L2 norm, clean zero for empty)
3. High cosine similarity parity (>0.9999) against PyTorch SentenceTransformer
4. Dense retrieval Top-K consistency against locked ChromaDB store
5. HybridRetriever end-to-end functionality
6. RAGService end-to-end functionality and citation preservation
7. Subprocess isolation: ONNX path does NOT import torch
8. Protected artifact integrity (hashes & ChromaDB vector count)
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.rag.embeddings import (
    ONNXEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.rag_service import RAGService
from src.rag.vector_store import VectorStoreManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIVE_QUERIES = [
    "Who is eligible for Rythu Bharosa?",
    "What health coverage is provided under Ayushman Bharat PM-JAY?",
    "What is the maximum project cost under PMEGP?",
    "What subsidy is available under PMEGP?",
    "What are the loan categories under MUDRA?",
    "Who can apply for Telangana ePASS scholarships?",
    "What benefits are provided under the MCH Kit Scheme?",
    "What pension is available under Aasara?",
]

# Protected artifact SHA-256 baseline hashes
PROTECTED_HASHES = {
    PROJECT_ROOT / "models" / "ranking" / "xgboost_ltr_model.json": "db4c166b0253b32b068776c09e468e1eacb2d4798dbf58dd6af633fd05b687f0",
    PROJECT_ROOT / "data" / "ranking" / "annotations_expert_01.json": "829e8da403a46b67543d9d077333b8bd8518f17427aeccc6952ff17011e68f18",
    PROJECT_ROOT / "data" / "eligibility" / "scheme_rules.json": "e16d3dedb485b49f19ee38d827b5f75c96e4120f5931b0b12e743e3f15813a1d",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class TestONNXEmbeddingBackend:
    """Test suite for ONNXEmbeddingProvider."""

    @pytest.fixture(scope="class")
    def onnx_provider(self) -> ONNXEmbeddingProvider:
        return ONNXEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    @pytest.fixture(scope="class")
    def pytorch_provider(self) -> SentenceTransformerEmbeddingProvider:
        return SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    def test_dimensions_and_shape(self, onnx_provider: ONNXEmbeddingProvider):
        """1. Verify dimensions are exactly 384 and single query output is shape (384,)."""
        assert onnx_provider.get_dimensions() == 384
        assert onnx_provider.model_name == "all-MiniLM-L6-v2"
        assert onnx_provider.provider_name == "onnx"

        emb = onnx_provider.embed_query("Who is eligible for Rythu Bharosa?")
        assert isinstance(emb, list)
        assert len(emb) == 384

        batch_embs = onnx_provider.embed_documents(["Query 1", "Query 2"])
        assert len(batch_embs) == 2
        assert len(batch_embs[0]) == 384
        assert len(batch_embs[1]) == 384

    def test_numerical_validity(self, onnx_provider: ONNXEmbeddingProvider):
        """2. Verify no NaN/Inf values, unit L2 norm, and clean empty handling."""
        emb = np.array(onnx_provider.embed_query("Sample citizen query for healthcare."))
        assert not np.isnan(emb).any(), "Embedding contains NaN values"
        assert not np.isinf(emb).any(), "Embedding contains Inf values"

        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-4), f"Embedding is not unit normalized: {norm}"

        # Empty / whitespace query handling
        empty_emb = onnx_provider.embed_query("")
        assert len(empty_emb) == 384
        assert all(v == 0.0 for v in empty_emb)

        space_emb = onnx_provider.embed_query("   ")
        assert len(space_emb) == 384
        assert all(v == 0.0 for v in space_emb)

    def test_cosine_similarity_parity_with_pytorch(
        self,
        onnx_provider: ONNXEmbeddingProvider,
        pytorch_provider: SentenceTransformerEmbeddingProvider,
    ):
        """3. Verify cosine similarity between PyTorch and ONNX is extremely high (>0.9999)."""
        for query in REPRESENTATIVE_QUERIES:
            pt_vec = np.array(pytorch_provider.embed_query(query), dtype=np.float32)
            ox_vec = np.array(onnx_provider.embed_query(query), dtype=np.float32)

            cos_sim = float(
                np.dot(pt_vec, ox_vec) / (np.linalg.norm(pt_vec) * np.linalg.norm(ox_vec))
            )
            max_diff = float(np.max(np.abs(pt_vec - ox_vec)))

            assert cos_sim >= 0.9999, (
                f"Cosine similarity too low for query '{query}': {cos_sim:.7f} < 0.9999"
            )
            assert max_diff <= 1e-4, (
                f"Max element difference too high for query '{query}': {max_diff:.7f} > 1e-4"
            )

    def test_dense_retrieval_top_k_consistency(
        self,
        onnx_provider: ONNXEmbeddingProvider,
        pytorch_provider: SentenceTransformerEmbeddingProvider,
    ):
        """4. Verify dense retrieval Top-K from locked ChromaDB is consistent across backends."""
        vstore = VectorStoreManager()

        for query in REPRESENTATIVE_QUERIES[:4]:
            pt_emb = pytorch_provider.embed_query(query)
            ox_emb = onnx_provider.embed_query(query)

            pt_results = vstore.query_similar(pt_emb, top_k=5)
            ox_results = vstore.query_similar(ox_emb, top_k=5)

            assert len(pt_results) == 5
            assert len(ox_results) == 5

            # Top-1 result MUST match
            assert pt_results[0].chunk_id == ox_results[0].chunk_id, (
                f"Top-1 chunk mismatch for '{query}': PyTorch={pt_results[0].chunk_id} vs ONNX={ox_results[0].chunk_id}"
            )
            assert pt_results[0].scheme_id == ox_results[0].scheme_id

            # Overlap in top 5 must be 100% or at least 4/5
            pt_ids = {r.chunk_id for r in pt_results}
            ox_ids = {r.chunk_id for r in ox_results}
            overlap = len(pt_ids & ox_ids)
            assert overlap >= 4, f"Top-5 overlap too low for '{query}': {overlap}/5 ({pt_ids} vs {ox_ids})"

            # Scores must be almost identical
            diff = abs(pt_results[0].similarity_score - ox_results[0].similarity_score)
            assert diff < 0.005, f"Top-1 similarity score difference too large: {diff}"

    def test_hybrid_retriever_with_onnx(self):
        """5. Verify HybridRetriever retrieves correct results with ONNX backend."""
        retriever = HybridRetriever()
        assert retriever._emb_provider.provider_name == "onnx"

        results = retriever.retrieve("Who is eligible for Rythu Bharosa?", final_top_k=5)
        assert len(results) == 5
        assert results[0].scheme_id == "TS001"
        assert results[0].rrf_score > 0.0

    def test_rag_service_with_onnx_and_citations(self):
        """6. Verify RAGService produces grounded answer with intact source citations."""
        rag = RAGService(allow_fallback=True)
        answer = rag.answer_query("Who is eligible for Rythu Bharosa?", top_k=5)

        assert answer.question == "Who is eligible for Rythu Bharosa?"
        assert answer.detected_scheme_id == "TS001"
        assert len(answer.sources) >= 1

        top_source = answer.sources[0]
        assert top_source.scheme_id == "TS001"
        assert top_source.official_url != ""
        assert top_source.source_filename.endswith(".html") or top_source.source_filename.endswith(".txt")

    def test_zero_torch_in_clean_subprocess(self):
        """7. Verify initializing ONNXEmbeddingProvider in clean process does NOT import torch."""
        code = (
            "import sys; "
            "from src.rag.embeddings import get_embedding_provider; "
            "p = get_embedding_provider('onnx'); "
            "v = p.embed_query('test query'); "
            "assert len(v) == 384; "
            "has_torch = 'torch' in sys.modules; "
            "print('HAS_TORCH=' + str(has_torch)); "
            "sys.exit(1 if has_torch else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Subprocess failed: stdout={result.stdout}, stderr={result.stderr}"
        assert "HAS_TORCH=False" in result.stdout

    def test_protected_artifact_integrity(self):
        """8. Verify all protected artifacts and ChromaDB collection remain completely untouched."""
        for path, expected_hash in PROTECTED_HASHES.items():
            assert path.exists(), f"Protected file missing: {path}"
            actual_hash = compute_sha256(path)
            assert actual_hash == expected_hash, (
                f"Protected artifact {path.name} was modified! "
                f"Expected: {expected_hash}, Got: {actual_hash}"
            )

        # Check ChromaDB collection integrity
        vstore = VectorStoreManager()
        integrity = vstore.verify_integrity()
        assert integrity["status"] == "PASS"
        assert integrity["total_stored_vectors"] == 166
