# -*- coding: utf-8 -*-
"""
scripts/verify_artifacts.py — Phase 9B: Runtime Artifacts Verification for Deployment

Checks that all 5 critical runtime artifacts are present, valid, and accessible:
1. Official RAG corpus (raw HTML files + metadata)
2. Processed documents (16 TXT files + corpus_manifest.json)
3. ChromaDB / vector store (166 vector embeddings)
4. Eligibility rules (14 statutory schemes in scheme_rules.json)
5. XGBoost model (xgboost_ltr_model.json artifact)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rag.vector_store import VectorStoreManager


def verify_all_runtime_artifacts():
    print("=" * 70)
    print("SchemeIQ+ Phase 9B — Deployment Artifacts Pre-Flight Verification")
    print("=" * 70)

    # 1. Official RAG corpus
    raw_dir = REPO_ROOT / "data" / "documents" / "raw"
    raw_html = list(raw_dir.glob("*.html"))
    meta_file = raw_dir / "metadata.json"
    assert len(raw_html) == 25, f"Expected 25 raw HTML files, found {len(raw_html)}"
    assert meta_file.exists(), "data/documents/raw/metadata.json missing"
    raw_meta = json.loads(meta_file.read_text("utf-8"))
    assert len(raw_meta) == 40, f"Expected 40 metadata entries, found {len(raw_meta)}"
    print(f"  [OK] 1. Official RAG Corpus: {len(raw_html)} raw HTML files, {len(raw_meta)} metadata entries verified.")

    # 2. Processed documents
    proc_dir = REPO_ROOT / "data" / "processed" / "documents"
    proc_txt = list(proc_dir.glob("*.txt"))
    manifest_file = REPO_ROOT / "data" / "processed" / "corpus_manifest.json"
    lock_file = REPO_ROOT / "data" / "processed" / "corpus_lock_report.json"
    assert len(proc_txt) == 16, f"Expected 16 processed TXT files, found {len(proc_txt)}"
    assert manifest_file.exists(), "corpus_manifest.json missing"
    manifest = json.loads(manifest_file.read_text("utf-8"))
    assert len(manifest) == 16, f"Expected 16 manifest entries, found {len(manifest)}"
    assert lock_file.exists(), "corpus_lock_report.json missing"
    print(f"  [OK] 2. Processed Documents: {len(proc_txt)} processed TXT files, 16 manifest entries verified.")

    # 3. ChromaDB / vector store
    vs = VectorStoreManager()
    v_count = vs.collection.count()
    assert v_count == 166, f"Expected 166 embeddings in ChromaDB, found {v_count}"
    print(f"  [OK] 3. ChromaDB Vector Store: {v_count} vector embeddings in collection '{vs.collection_name}' verified.")

    # 4. Eligibility rules
    rules_file = REPO_ROOT / "data" / "eligibility" / "scheme_rules.json"
    assert rules_file.exists(), "scheme_rules.json missing"
    rules_data = json.loads(rules_file.read_text("utf-8"))
    schemes_list = rules_data.get("schemes", rules_data)
    assert len(schemes_list) == 14, f"Expected 14 statutory schemes, found {len(schemes_list)}"
    print(f"  [OK] 4. Eligibility Rules: {len(schemes_list)} statutory schemes verified in scheme_rules.json ({rules_file.stat().st_size} bytes).")

    # 5. XGBoost model
    model_file = REPO_ROOT / "models" / "ranking" / "xgboost_ltr_model.json"
    assert model_file.exists(), "xgboost_ltr_model.json missing"
    assert model_file.stat().st_size == 38388, f"Unexpected model file size: {model_file.stat().st_size}"
    print(f"  [OK] 5. XGBoost Model: {model_file.name} ({model_file.stat().st_size} bytes) verified intact.")

    print("\n" + "=" * 70)
    print("ALL 5 REQUIRED RUNTIME ARTIFACTS VERIFIED SUCCESSFULLY (5/5)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_all_runtime_artifacts()
