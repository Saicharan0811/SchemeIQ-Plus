# -*- coding: utf-8 -*-
"""
wsgi.py — Production WSGI Entrypoint for SchemeIQ+ (Phase 9A)

Provides the standard WSGI callable 'app' for production application servers:
- Waitress (Windows / Cross-platform):
    waitress-serve --port=5000 wsgi:app
    or:
    python wsgi.py
- Gunicorn (Linux / Container):
    gunicorn --workers 2 --bind 0.0.0.0:5000 wsgi:app

Pre-warms heavy models (SentenceTransformer, ChromaDB, XGBoost) on the main process thread
prior to worker request serving to eliminate latency spikes and thread-local loading issues.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env configuration if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("schemeiq.wsgi")

# Pre-warm core services on main process thread
logger.info("Initializing SchemeIQ+ core services for production WSGI...")
from src.eligibility.service import EligibilityService
from src.rag.rag_service import RAGService
from src.api.app import create_app

try:
    _rag = RAGService(allow_fallback=True)
    logger.info("RAGService pre-warmed successfully.")
except Exception as e:
    logger.warning(f"RAGService pre-warming warning: {e}. Lazy loading will be used.")
    _rag = None

try:
    _elig = EligibilityService()
    logger.info("EligibilityService initialized successfully.")
except Exception as e:
    logger.warning(f"EligibilityService pre-warming warning: {e}.")
    _elig = None

# Create the WSGI application callable
app = create_app(eligibility_service=_elig, rag_service=_rag)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    threads = int(os.environ.get("WSGI_THREADS", "4"))

    try:
        import waitress
        logger.info(f"Starting SchemeIQ+ production WSGI server (Waitress) on http://{host}:{port} ({threads} threads)...")
        waitress.serve(app, host=host, port=port, threads=threads)
    except ImportError:
        logger.warning("Waitress not installed; falling back to Flask development server with debug=False.")
        app.run(host=host, port=port, debug=False)
