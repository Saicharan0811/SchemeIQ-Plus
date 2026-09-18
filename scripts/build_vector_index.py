# -*- coding: utf-8 -*-
"""
scripts/build_vector_index.py — Deterministic Vector Index Builder for SchemeIQ+

Builds a lightweight, self-contained, deterministic vector index from the
16 canonical processed scheme documents and corpus manifest:
  data/processed/documents/*.txt
  data/processed/corpus_manifest.json

Outputs:
  data/vector_store/dense_index.json

This replaces the heavy ChromaDB / SQLite runtime with a static JSON artifact
containing all 166 chunks, 384-dimensional ONNX embeddings, and full metadata.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.document_loader import DocumentLoader
from src.rag.chunker import StructureAwareChunker
from src.rag.embeddings import get_embedding_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("schemeiq.build_vector_index")

DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_manifest.json"
OUTPUT_INDEX_PATH = PROJECT_ROOT / "data" / "vector_store" / "dense_index.json"
EXPECTED_VECTOR_COUNT = 166
EXPECTED_SCHEME_COUNT = 14


def build_vector_index(
    docs_dir: Path = DOCS_DIR,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_INDEX_PATH,
) -> dict:
    """
    Deterministically chunk and embed canonical documents, saving to output_path.
    Returns the index dictionary.
    """
    logger.info("Starting deterministic vector index build...")
    t0 = time.perf_counter()

    # 1. Load canonical documents
    loader = DocumentLoader(docs_dir=docs_dir, manifest_path=manifest_path)
    documents = loader.load_documents()
    logger.info("Loaded %d documents from %s", len(documents), docs_dir)

    # 2. Chunk documents
    chunker = StructureAwareChunker(chunk_size=800, chunk_overlap=120)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))

    # 3. Deduplicate and filter empty chunks
    seen_ids: set[str] = set()
    valid_chunks = []
    for chk in all_chunks:
        if not chk.text or not chk.text.strip():
            continue
        if chk.chunk_id in seen_ids:
            continue
        seen_ids.add(chk.chunk_id)
        valid_chunks.append(chk)

    logger.info("Produced %d valid unique chunks", len(valid_chunks))

    if len(valid_chunks) != EXPECTED_VECTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_COUNT} chunks, got {len(valid_chunks)}"
        )

    # 4. Generate 384-dimensional embeddings via ONNX
    emb_provider = get_embedding_provider()
    logger.info("Using embedding provider: %s (%s)", emb_provider.provider_name, emb_provider.model_name)

    texts = [c.text for c in valid_chunks]
    embeddings = emb_provider.embed_documents(texts)
    logger.info("Generated %d embeddings (dim=%d)", len(embeddings), len(embeddings[0]))

    # 5. Assemble index records
    records = []
    unique_schemes = set()
    for chk, emb in zip(valid_chunks, embeddings):
        meta = chk.to_chroma_metadata()
        unique_schemes.add(meta.get("scheme_id", ""))
        records.append({
            "chunk_id": chk.chunk_id,
            "text": chk.text,
            "embedding": [round(float(v), 6) for v in emb],
            "metadata": meta,
        })

    if len(unique_schemes) != EXPECTED_SCHEME_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SCHEME_COUNT} unique schemes, found {len(unique_schemes)}: {unique_schemes}"
        )

    index_data = {
        "version": "1.0.0",
        "collection_name": "schemeiq_official_schemes",
        "embedding_model": getattr(emb_provider, "model_name", "all-MiniLM-L6-v2"),
        "embedding_dimension": len(embeddings[0]),
        "total_records": len(records),
        "unique_scheme_count": len(unique_schemes),
        "indexed_scheme_ids": sorted(list(unique_schemes)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }

    # 6. Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Vector index successfully built and saved to %s in %.2fs (%d records)",
        output_path, elapsed, len(records),
    )
    return index_data


if __name__ == "__main__":
    build_vector_index()
