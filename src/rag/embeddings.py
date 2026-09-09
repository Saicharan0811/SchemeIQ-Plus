# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Embeddings Provider Abstraction
Provides extensible, pluggable embedding model implementations for dense semantic retrieval.
Supports:
1. ONNX Runtime (default, low-memory production inference, zero PyTorch dependency)
2. Local SentenceTransformers (PyTorch fallback / training / development)
3. OpenAI embeddings (external API option)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Global provider instance cache to prevent duplicate model sessions
_CACHED_PROVIDERS: Dict[str, EmbeddingProvider] = {}


class EmbeddingProvider(ABC):
    """Abstract base class for RAG embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the generated vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""
        pass


class ONNXEmbeddingProvider(EmbeddingProvider):
    """
    Low-memory ONNX Runtime embedding provider for all-MiniLM-L6-v2.
    Achieves identical numerical embeddings (~1.0 cosine similarity) without importing PyTorch.
    Uses tokenizers + onnxruntime for minimal memory footprint (~30 MB vs ~260 MB).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_dir: Optional[Union[str, Path]] = None,
        session_options: Optional[Any] = None,
    ) -> None:
        self._model_name = model_name
        self._provider_name = "onnx"
        self._dim = 384

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            logger.error(f"Missing ONNX dependencies: {e}. Install onnxruntime and tokenizers.")
            raise

        model_path, tokenizer_path = self._resolve_model_files(model_dir)
        logger.info(f"Loading ONNX embedding model: {self._model_name} from {model_path}")

        so = session_options
        if so is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        else:
            try:
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # all-MiniLM-L6-v2 standard maximum sequence length
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

    def _resolve_model_files(self, model_dir: Optional[Union[str, Path]]) -> tuple[Path, Path]:
        """Resolve local or cached ONNX model and tokenizer paths."""
        # 1. Check explicit model_dir
        if model_dir:
            p = Path(model_dir)
            m_path = p / "model.onnx"
            t_path = p / "tokenizer.json"
            if m_path.exists() and t_path.exists():
                return m_path, t_path

        # 2. Check environment variable ONNX_MODEL_DIR
        env_dir = os.environ.get("ONNX_MODEL_DIR")
        if env_dir:
            p = Path(env_dir)
            m_path = p / "model.onnx"
            t_path = p / "tokenizer.json"
            if m_path.exists() and t_path.exists():
                return m_path, t_path

        # 3. Check local repository model directory
        repo_root = Path(__file__).resolve().parents[2]
        local_dir = repo_root / "models" / "embeddings" / self._model_name
        if (local_dir / "model.onnx").exists() and (local_dir / "tokenizer.json").exists():
            return local_dir / "model.onnx", local_dir / "tokenizer.json"

        # 4. Check ChromaDB ONNX cache if present
        chroma_cache = Path.home() / ".cache" / "chroma" / "onnx_models" / self._model_name / "onnx"
        if (chroma_cache / "model.onnx").exists() and (chroma_cache / "tokenizer.json").exists():
            return chroma_cache / "model.onnx", chroma_cache / "tokenizer.json"

        # 5. Resolve via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            repo_id = f"sentence-transformers/{self._model_name}"
            m_file = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")
            t_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
            return Path(m_file), Path(t_file)
        except Exception as e:
            logger.error(f"Failed to locate or download ONNX model files: {e}")
            raise FileNotFoundError(
                f"Could not resolve ONNX model files for {self._model_name}. "
                "Ensure huggingface_hub is installed or files exist in local path."
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = [self.tokenizer.encode(t) for t in batch]

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, 384)

            # Mean pooling weighted by attention mask
            mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            # L2 normalization (cosine-ready)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norm, a_min=1e-12, a_max=None)

            all_embeddings.extend(normalized.astype(np.float32).tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dim
        return self.embed_documents([text])[0]

    def get_dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local SentenceTransformer embedding provider for fast, offline vector generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._provider_name = "sentence-transformers"

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self._model_name}")
            self.model = SentenceTransformer(self._model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {self._model_name}: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def get_dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider using text-embedding-3-small or text-embedding-ada-002."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self._model_name = model_name
        self._provider_name = "openai"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable or use local embeddings.")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self._dim = 1536 if "small" in self._model_name or "ada" in self._model_name else 3072
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Batch requests
        response = self.client.embeddings.create(
            input=texts,
            model=self._model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim
        response = self.client.embeddings.create(
            input=[text],
            model=self._model_name
        )
        return response.data[0].embedding

    def get_dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


def get_embedding_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    force_new: bool = False,
) -> EmbeddingProvider:
    """
    Factory function to initialize and return the configured embedding provider.
    Priority:
    1. Explicit arguments
    2. Environment variables (EMBEDDING_PROVIDER, EMBEDDING_MODEL)
    3. Default: ONNX ('all-MiniLM-L6-v2') — low memory, zero PyTorch dependency

    Features singleton caching to avoid redundant model sessions in production.
    """
    raw_provider = (
        provider_type
        or os.environ.get("EMBEDDING_PROVIDER")
        or "onnx"
    ).strip().lower()

    raw_model = (
        model_name
        or os.environ.get("EMBEDDING_MODEL")
        or "all-MiniLM-L6-v2"
    ).strip()

    cache_key = f"{raw_provider}::{raw_model}"
    if not force_new and cache_key in _CACHED_PROVIDERS:
        return _CACHED_PROVIDERS[cache_key]

    if raw_provider in ["openai"]:
        model = raw_model if raw_model else "text-embedding-3-small"
        provider: EmbeddingProvider = OpenAIEmbeddingProvider(model_name=model)
    elif raw_provider in ["sentence-transformers", "sentence_transformers", "pytorch", "torch"]:
        model = raw_model if raw_model else "all-MiniLM-L6-v2"
        provider = SentenceTransformerEmbeddingProvider(model_name=model)
    else:
        # Default to ONNX Runtime
        model = raw_model if raw_model else "all-MiniLM-L6-v2"
        provider = ONNXEmbeddingProvider(model_name=model)

    if not force_new:
        _CACHED_PROVIDERS[cache_key] = provider

    return provider
