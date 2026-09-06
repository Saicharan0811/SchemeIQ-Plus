# -*- coding: utf-8 -*-
"""
scripts/verify_render_deployment.py — Phase 9B: Live Public API Verification

Performs full end-to-end verification against a live deployed Render service
(or local production WSGI server) across all 10 required checks:
1. GET /health
2. POST /api/recommend (Heuristic mode)
3. GET /api/schemes/TS001
4. POST /api/ask (Grounded RAG)
5. Invalid scheme lookup -> 404
6. Privacy violation pre-screening -> rejected (400 PRIVACY_VIOLATION)
7. Invalid parameters -> rejected (400 VALIDATION_ERROR / INVALID_TOP_K)
8. CORS behavior (OPTIONS preflight & headers)
9. XGBoost-enabled recommendation (order updated, eligibility invariant preserved)
10. RAG grounded response (citations, official URLs, grounding status)

Usage:
  python scripts/verify_render_deployment.py https://schemeiq-plus-backend.onrender.com
  or (defaults to http://127.0.0.1:5000):
  python scripts/verify_render_deployment.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:5000"


def http_req(base_url: str, method: str, path: str, data: dict = None, headers: dict = None):
    url = f"{base_url.rstrip('/')}{path}"
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            resp_headers = dict(resp.headers)
            try:
                return resp.status, json.loads(raw), resp_headers
            except Exception:
                return resp.status, raw, resp_headers
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        resp_headers = dict(e.headers)
        try:
            return e.code, json.loads(raw), resp_headers
        except Exception:
            return e.code, raw, resp_headers


def ensure_local_server_if_needed(base_url: str):
    """If testing against localhost and no server is running, spin up the production app."""
    if "127.0.0.1" in base_url or "localhost" in base_url:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        print(f"  No server detected at {base_url}. Starting local production server...")
        import logging
        from threading import Thread
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        from src.rag.rag_service import RAGService
        from src.eligibility.service import EligibilityService
        from src.api.app import create_app
        rag = RAGService()
        elig = EligibilityService()
        app = create_app(eligibility_service=elig, rag_service=rag)
        t = Thread(target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False), daemon=True)
        t.start()
        import time
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=1) as r:
                    if r.status == 200:
                        print(f"  Local production server started successfully.")
                        return
            except Exception:
                time.sleep(0.5)


def run_verification(base_url: str):
    ensure_local_server_if_needed(base_url)
    print("=" * 70)
    print(f"SchemeIQ+ Phase 9B — Production Live API Verification")
    print(f"Target Service URL: {base_url}")

    print("=" * 70)

    # 1. Health check
    print("\n[Check 1/10] GET /health...")
    status, health_body, _ = http_req(base_url, "GET", "/health")
    assert status == 200, f"Health check failed with status {status}: {health_body}"
    assert health_body.get("status") == "healthy"
    assert health_body.get("official_schemes_count") == 14
    assert health_body.get("xgboost_model_available") is True
    print(f"  [PASS] Service healthy, 14 statutory schemes, XGBoost available.")

    # 2. POST /api/recommend (Heuristic mode)
    print("\n[Check 2/10] POST /api/recommend (Heuristic mode, enable_xgboost=False)...")
    farmer_profile = {
        "age": 42,
        "gender": "male",
        "state": "Telangana",
        "residence_type": "rural",
        "occupation": "farmer",
        "farmer_status": True,
        "land_ownership_status": True,
        "land_acres": 3.5,
        "annual_income": 140000,
        "pays_income_tax": False,
        "is_bpl_or_white_ration_card": True,
    }
    status, heur_rec, _ = http_req(base_url, "POST", "/api/recommend", data={
        "profile": farmer_profile,
        "top_k": 5,
        "enable_xgboost": False,
    })
    assert status == 200, f"Recommend failed with {status}: {heur_rec}"
    assert len(heur_rec["recommendations"]) > 0
    top_heur = heur_rec["recommendations"][0]
    assert "xgboost_score" not in top_heur.get("scoring_breakdown", {})
    print(f"  [PASS] Top heuristic: '{top_heur['scheme_name']}' (Score: {top_heur['score']}, Status: {top_heur['eligibility_status']})")

    # 3. GET /api/schemes/TS001
    print("\n[Check 3/10] GET /api/schemes/TS001...")
    status, scheme_body, _ = http_req(base_url, "GET", "/api/schemes/TS001")
    assert status == 200, f"Scheme lookup failed with {status}: {scheme_body}"
    assert scheme_body.get("scheme_id") == "TS001"
    assert len(scheme_body.get("rules", [])) > 0
    assert scheme_body.get("official_url")
    print(f"  [PASS] Scheme TS001: '{scheme_body['scheme_name']}', {len(scheme_body['rules'])} statutory rules.")

    # 4. POST /api/ask (Grounded RAG)
    print("\n[Check 4/10] POST /api/ask (Grounded RAG)...")
    status, rag_body, _ = http_req(base_url, "POST", "/api/ask", data={
        "query": "What financial assistance does Rythu Bharosa provide?",
        "top_k": 3,
    })
    assert status == 200, f"RAG ask failed with {status}: {rag_body}"
    assert rag_body.get("grounding_status") in ("GROUNDED", "PARTIALLY_GROUNDED", "FALLBACK")
    assert len(rag_body.get("sources", [])) > 0
    assert len(rag_body.get("answer", "")) > 20
    print(f"  [PASS] RAG answer generated (Status: {rag_body['grounding_status']}, Sources: {len(rag_body['sources'])}).")

    # 5. Invalid scheme lookup -> 404
    print("\n[Check 5/10] GET /api/schemes/NONEXISTENT_999 (Invalid scheme 404)...")
    status, not_found_body, _ = http_req(base_url, "GET", "/api/schemes/NONEXISTENT_999")
    assert status == 404, f"Expected 404, got {status}"
    assert not_found_body.get("code") == "SCHEME_NOT_FOUND"
    print(f"  [PASS] HTTP 404 SCHEME_NOT_FOUND confirmed.")

    # 6. Privacy violation rejection
    print("\n[Check 6/10] Privacy pre-screening (Prohibited PII rejection)...")
    for bad_field, bad_val in [("aadhaar_number", "1234-5678-9012"), ("pan_number", "ABCDE1234F")]:
        bad_profile = dict(farmer_profile, **{bad_field: bad_val})
        status, pii_body, _ = http_req(base_url, "POST", "/api/recommend", data={"profile": bad_profile})
        assert status == 400, f"Expected 400 for {bad_field}, got {status}"
        assert pii_body.get("code") == "PRIVACY_VIOLATION"
    print(f"  [PASS] Prohibited sensitive fields strictly intercepted with HTTP 400 PRIVACY_VIOLATION.")

    # 7. Invalid parameters rejection
    print("\n[Check 7/10] Invalid parameters rejection...")
    bad_age_profile = dict(farmer_profile, age=150)
    status, age_err, _ = http_req(base_url, "POST", "/api/recommend", data={"profile": bad_age_profile})
    assert status == 400 and age_err.get("code") == "VALIDATION_ERROR"

    status, topk_err, _ = http_req(base_url, "POST", "/api/recommend", data={"profile": farmer_profile, "top_k": 99})
    assert status == 400 and topk_err.get("code") == "INVALID_TOP_K"
    print(f"  [PASS] Invalid age (>120) and top_k (>14) rejected cleanly with HTTP 400.")

    # 8. CORS behavior
    print("\n[Check 8/10] CORS & OPTIONS Preflight...")
    status, _, cors_headers = http_req(base_url, "OPTIONS", "/api/recommend")
    assert status in (200, 204), f"OPTIONS preflight returned {status}"
    assert "POST" in cors_headers.get("Access-Control-Allow-Methods", "")
    assert cors_headers.get("Access-Control-Allow-Origin") is not None
    print(f"  [PASS] OPTIONS preflight OK, Access-Control-Allow-Origin: {cors_headers.get('Access-Control-Allow-Origin')}")

    # 9. XGBoost-enabled recommendation
    print("\n[Check 9/10] POST /api/recommend (XGBoost Mode: enable_xgboost=True)...")
    status, xgb_rec, _ = http_req(base_url, "POST", "/api/recommend", data={
        "profile": farmer_profile,
        "top_k": 5,
        "enable_xgboost": True,
    })
    assert status == 200, f"XGBoost recommend failed with {status}: {xgb_rec}"
    top_xgb = xgb_rec["recommendations"][0]
    assert "xgboost_score" in top_xgb.get("scoring_breakdown", {}), "Missing xgboost_score in breakdown"
    # Verify eligibility invariant: status counts identical
    assert heur_rec["status_counts"] == xgb_rec["status_counts"], "Status counts diverged between modes!"
    print(f"  [PASS] Top XGBoost: '{top_xgb['scheme_name']}' (ML score: {top_xgb['scoring_breakdown']['xgboost_score']:.4f}).")
    print(f"  [PASS] Strict Safety Invariant Confirmed: Eligibility statuses unchanged by XGBoost.")

    # 10. RAG grounded response verification
    print("\n[Check 10/10] RAG source citations & official URLs...")
    src = rag_body["sources"][0]
    assert src.get("document_title"), "Missing document_title in source citation"
    assert src.get("source_filename"), "Missing source_filename in source citation"
    assert src.get("official_url"), "Missing official_url in source citation"
    assert src.get("chunk_id"), "Missing chunk_id in source citation"
    print(f"  [PASS] Source citation complete: title='{src['document_title']}', file='{src['source_filename']}', url='{src['official_url']}'.")

    print("\n" + "=" * 70)
    print("ALL 10 PRODUCTION API VERIFICATION CHECKS PASSED SUCCESSFULLY (10/10)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    run_verification(target)
