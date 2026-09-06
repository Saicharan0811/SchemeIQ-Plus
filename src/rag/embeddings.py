# -*- coding: utf-8 -*-
"""
SchemeIQ+ — Embeddings Provider Abstraction
Provides extensible, pluggable embedding model implementations for dense semantic retrieval.
Supports local SentenceTransformers (default) and OpenAI embeddings with zero hardcoded credentials.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


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
) -> EmbeddingProvider:
    """
    Factory function to initialize and return the configured embedding provider.
    Priority:
    1. Explicit arguments
    2. Environment variables (EMBEDDING_PROVIDER, EMBEDDING_MODEL)
    3. Default: SentenceTransformer ('all-MiniLM-L6-v2')
    """
    provider = (provider_type or os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers")).lower()
    
    if provider in ["openai"]:
        model = model_name or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddingProvider(model_name=model)
    else:
        model = model_name or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingProvider(model_name=model)
