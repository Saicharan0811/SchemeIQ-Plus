#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/start_chroma_service.py — SchemeIQ+ Chroma Private Service Startup

Runs inside the Render private service container (schemeiq-plus-chroma).

Startup sequence:
  1. Launch 'chroma run' as a subprocess on 0.0.0.0:8000 using /tmp/chroma_runtime.
  2. Poll local HttpClient.heartbeat() until Chroma is ready (up to 120 s).
  3. Inspect the 'schemeiq_official_schemes' collection count.
  4. If count == 0: seed the canonical corpus via existing ingest components.
  5. Verify collection.count() == 166. Exit non-zero if mismatch (Render restarts).
  6. Wait on the chroma subprocess indefinitely (serves queries from the backend).

No corpus data is hard-coded here. Reads from:
  data/processed/documents/  (16 canonical scheme text files)
  data/processed/corpus_manifest.json

CHROMA_PERSIST_PATH env var overrides the default /tmp/chroma_runtime path.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("schemeiq.chroma_service")

# Configuration
CHROMA_HOST = "0.0.0.0"
CHROMA_PORT = int(os.environ.get("CHROMA_SERVER_PORT", "8000"))
CHROMA_PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "/tmp/chroma_runtime")
COLLECTION_NAME = "schemeiq_official_schemes"
EXPECTED_VECTOR_COUNT = 166

HEARTBEAT_POLL_INTERVAL_S = 2.0
HEARTBEAT_TIMEOUT_S = 120.0

# Canonical corpus paths (relative to repo root)
DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_manifest.json"


def _launch_chroma() -> subprocess.Popen:
    """Start 'chroma run' as a subprocess and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "chromadb.cli.cli", "run",
        "--path", CHROMA_PERSIST_PATH,
        "--host", CHROMA_HOST,
        "--port", str(CHROMA_PORT),
    ]
    logger.info("Launching Chroma: %s", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))


def _wait_for_heartbeat(local_client) -> None:
    """Poll Chroma heartbeat until ready or timeout. Raises RuntimeError on timeout."""
    deadline = time.monotonic() + HEARTBEAT_TIMEOUT_S
    attempt = 0
    while True:
        attempt += 1
        try:
            hb = local_client.heartbeat()
            if isinstance(hb, (int, float)) and hb > 0:
                logger.info("Chroma heartbeat OK after %d attempt(s).", attempt)
                return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Chroma did not become ready within {HEARTBEAT_TIMEOUT_S}s. "
                    f"Last error: {exc}"
                ) from exc
            logger.debug(
                "Heartbeat attempt %d failed (%s); retrying in %.1fs...",
                attempt, exc, HEARTBEAT_POLL_INTERVAL_S,
            )
            time.sleep(HEARTBEAT_POLL_INTERVAL_S)


def _seed_corpus(collection) -> None:
    """Seed the collection from canonical corpus using existing pipeline components.

    Uses DocumentLoader, StructureAwareChunker, and the ONNX embedding provider
    configured by EMBEDDING_PROVIDER / EMBEDDING_MODEL env vars.  No data is
    duplicated or hard-coded -- only the Git-tracked processed documents are read.
    """
    from src.rag.document_loader import DocumentLoader
    from src.rag.chunker import StructureAwareChunker
    from src.rag.embeddings import get_embedding_provider

    logger.info("Collection is empty. Seeding canonical corpus from %s ...", DOCS_DIR)

    loader = DocumentLoader(docs_dir=DOCS_DIR, manifest_path=MANIFEST_PATH)
    chunker = StructureAwareChunker(chunk_size=800, chunk_overlap=120)
    emb_provider = get_embedding_provider()

    documents = loader.load_documents()
    logger.info("Loaded %d documents.", len(documents))

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))

    # Deduplicate and discard empty chunks (mirrors ingest.py logic)
    seen_ids: set[str] = set()
    valid_chunks = []
    for chk in all_chunks:
        if not chk.text or not chk.text.strip():
            continue
        if chk.chunk_id in seen_ids:
            continue
        seen_ids.add(chk.chunk_id)
        valid_chunks.append(chk)

    logger.info("Chunked corpus: %d valid chunks to embed.", len(valid_chunks))

    texts = [c.text for c in valid_chunks]
    embeddings = emb_provider.embed_documents(texts)
    logger.info("Generated %d embedding vectors.", len(embeddings))

    ids = [c.chunk_id for c in valid_chunks]
    documents_texts = [c.text for c in valid_chunks]
    metadatas = [c.to_chroma_metadata() for c in valid_chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents_texts,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into '%s'.", len(valid_chunks), COLLECTION_NAME)


def main() -> None:
    import chromadb
    from chromadb.config import Settings

    # 1. Launch Chroma
    chroma_proc = _launch_chroma()

    try:
        # 2. Wait for heartbeat
        local_client = chromadb.HttpClient(
            host="localhost",
            port=CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False),
        )
        _wait_for_heartbeat(local_client)

        # 3. Inspect / create collection
        collection = local_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "description": "SchemeIQ+ Official Government Scheme Corpus",
            },
        )
        count = collection.count()
        logger.info("Collection '%s' count: %d", COLLECTION_NAME, count)

        # 4. Seed if empty
        if count == 0:
            _seed_corpus(collection)
        else:
            logger.info("Collection already populated -- skipping seed.")

        # 5. Verify vector count
        final_count = collection.count()
        logger.info("Final vector count: %d (expected %d)", final_count, EXPECTED_VECTOR_COUNT)
        if final_count != EXPECTED_VECTOR_COUNT:
            logger.error(
                "CORPUS VERIFICATION FAILED: expected %d vectors, found %d. "
                "Render will restart the service.",
                EXPECTED_VECTOR_COUNT, final_count,
            )
            chroma_proc.terminate()
            sys.exit(1)

        logger.info(
            "Chroma service ready. Corpus verified (%d vectors). Serving on port %d.",
            final_count, CHROMA_PORT,
        )

        # 6. Stay alive -- Chroma subprocess handles all queries
        chroma_proc.wait()
        exit_code = chroma_proc.returncode
        logger.error("Chroma subprocess exited with code %d.", exit_code)
        sys.exit(exit_code if exit_code is not None else 1)

    except Exception as exc:
        logger.error("Startup failed: %s", exc, exc_info=True)
        chroma_proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
