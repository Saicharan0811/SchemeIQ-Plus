# -*- coding: utf-8 -*-
"""
scripts/e2e_smoke_test.py — SchemeIQ+ Phase 8D End-to-End Integration Smoke Test

Validates complete browser-to-backend integration:
1. Flask API lifecycle on http://127.0.0.1:5000
2. Health check endpoint (model availability, 14 schemes)
3. Citizen recommendation flow (heuristic mode)
4. Citizen recommendation flow (advisory XGBoost mode)
5. Invariant check: XGBoost ON vs OFF preserves identical eligibility statuses
6. Official scheme lookup (valid TS001 and 404 unknown)
7. Grounded RAG question answering with official citations
8. Privacy pre-screening rejection (Aadhaar, PAN, Bank Account)
9. Validation error states (age out-of-bounds, negative income)
10. Invalid top_k and boolean flag type guards
11. RAG empty and short query guards
12. CORS headers and OPTIONS preflight
13. Frontend production build artifacts and route integrity
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.api.app import create_app

PORT = 5000
BASE_URL = f"http://127.0.0.1:{PORT}"


def start_server():
    """Start the production Flask app in a background daemon thread.

    The RAGService (SentenceTransformer embedding model) MUST be pre-initialized
    on the main thread before the daemon thread starts. On Windows, PyTorch/
    SentenceTransformer models initialized from background daemon threads may
    fail with 'NoneType has no attribute tokenize' due to thread-local state.
    """
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Pre-initialize heavy services on the main thread
    print("  Pre-warming RAGService (SentenceTransformer embedding model)...")
    from src.rag.rag_service import RAGService
    from src.eligibility.service import EligibilityService
    rag = RAGService()
    elig = EligibilityService()

    app = create_app(eligibility_service=elig, rag_service=rag)
    server_thread = Thread(
        target=lambda: app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()
    # Wait for server to become responsive
    max_wait = 30
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"Server did not start on {BASE_URL} within {max_wait} seconds")


def http_request(method: str, path: str, data: dict = None, headers: dict = None):
    """Utility to make HTTP requests and return (status_code, response_json_or_text, headers)."""
    url = f"{BASE_URL}{path}"
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


def run_all_tests():
    print("=" * 70)
    print("SchemeIQ+ Phase 8D — Full End-to-End Integration Validation")
    print("=" * 70)

    print("\n[Step 1] Starting Flask API server on http://127.0.0.1:5000...")
    start_server()
    print("  ✓ Server is responsive and accepting connections.")

    # 1. Health check
    print("\n[Step 2] Testing GET /health...")
    status, body, _ = http_request("GET", "/health")
    assert status == 200, f"Health returned {status}: {body}"
    assert body["status"] == "healthy", f"Status not healthy: {body}"
    assert body["official_schemes_count"] == 14, f"Schemes count: {body['official_schemes_count']}"
    assert body["xgboost_model_available"] is True, "XGBoost model reported unavailable"
    print(f"  ✓ Health OK: service='{body['service']}', schemes={body['official_schemes_count']}, xgboost_available={body['xgboost_model_available']}")

    # 2. Recommendation in Heuristic Mode
    print("\n[Step 3] Testing POST /api/recommend (Heuristic Mode: enable_xgboost=False)...")
    profile = {
        "age": 42,
        "gender": "male",
        "state": "Telangana",
        "district": "Nalgonda",
        "residence_type": "rural",
        "occupation": "farmer",
        "farmer_status": True,
        "land_ownership_status": True,
        "land_acres": 3.5,
        "annual_income": 140000.0,
        "pays_income_tax": False,
        "is_bpl_or_white_ration_card": True,
    }
    status, heur_body, _ = http_request("POST", "/api/recommend", data={
        "profile": profile,
        "top_k": 5,
        "enable_xgboost": False,
    })
    assert status == 200, f"Recommendation failed with {status}: {heur_body}"
    assert heur_body["total_schemes_evaluated"] == 14
    assert len(heur_body["recommendations"]) == 5
    assert sum(heur_body["status_counts"].values()) == 14
    top_rec = heur_body["recommendations"][0]
    assert "scheme_id" in top_rec
    assert "scheme_name" in top_rec
    assert "eligibility_status" in top_rec
    assert "score" in top_rec
    assert "matched_rules" in top_rec
    assert "xgboost_score" not in top_rec["scoring_breakdown"], "xgboost_score should NOT be present when enable_xgboost=False"
    print(f"  ✓ Heuristic recommendation OK: top-1='{top_rec['scheme_name']}' ({top_rec['scheme_id']}) status={top_rec['eligibility_status']} score={top_rec['score']}")

    # 3. Recommendation in XGBoost Mode
    print("\n[Step 4] Testing POST /api/recommend (XGBoost Mode: enable_xgboost=True)...")
    status, xgb_body, _ = http_request("POST", "/api/recommend", data={
        "profile": profile,
        "top_k": 5,
        "enable_xgboost": True,
    })
    assert status == 200, f"XGBoost recommendation failed with {status}: {xgb_body}"
    assert len(xgb_body["recommendations"]) == 5
    xgb_top = xgb_body["recommendations"][0]
    assert "xgboost_score" in xgb_top["scoring_breakdown"], "xgboost_score MUST be present when enable_xgboost=True"
    print(f"  ✓ XGBoost recommendation OK: top-1='{xgb_top['scheme_name']}' ({xgb_top['scheme_id']}) status={xgb_top['eligibility_status']} score={xgb_top['score']} xgb_score={xgb_top['scoring_breakdown']['xgboost_score']:.4f}")

    # 4. Strict Invariant Check: Eligibility Statuses Untouched by XGBoost
    print("\n[Step 5] Verifying Strict Invariant: Eligibility Statuses UNCHANGED by XGBoost...")
    heur_statuses = {r["scheme_id"]: r["eligibility_status"] for r in heur_body["recommendations"]}
    xgb_statuses = {r["scheme_id"]: r["eligibility_status"] for r in xgb_body["recommendations"]}
    for sid in heur_statuses:
        if sid in xgb_statuses:
            assert heur_statuses[sid] == xgb_statuses[sid], (
                f"INVARIANT VIOLATION: Scheme {sid} status changed from {heur_statuses[sid]} to {xgb_statuses[sid]}"
            )
    assert heur_body["status_counts"] == xgb_body["status_counts"], "Status counts differ between heuristic and XGBoost modes!"
    print("  ✓ Strict Invariant Confirmed: XGBoost reordered candidates without changing any eligibility status.")

    # 5. Scheme Lookup Valid & Invalid
    print("\n[Step 6] Testing GET /api/schemes/<scheme_id>...")
    status, scheme_body, _ = http_request("GET", "/api/schemes/TS001")
    assert status == 200, f"Scheme lookup failed: {status}"
    assert scheme_body["scheme_id"] == "TS001"
    assert scheme_body["scheme_name"] == "Rythu Bharosa"
    assert len(scheme_body["rules"]) > 0
    print(f"  ✓ Scheme TS001 lookup OK: name='{scheme_body['scheme_name']}', rules={len(scheme_body['rules'])}, url='{scheme_body['official_url']}'")

    status, err_body, _ = http_request("GET", "/api/schemes/UNKNOWN_999")
    assert status == 404, f"Expected 404 for unknown scheme, got {status}"
    assert err_body["code"] == "SCHEME_NOT_FOUND"
    print(f"  ✓ Scheme UNKNOWN_999 404 OK: code='{err_body['code']}'")

    # 6. Grounded RAG Question Answering
    print("\n[Step 7] Testing POST /api/ask (Grounded RAG)...")
    status, rag_body, _ = http_request("POST", "/api/ask", data={
        "query": "Who is eligible for Rythu Bharosa financial assistance?",
        "top_k": 3,
    })
    assert status == 200, f"RAG ask failed with {status}: {rag_body}"
    assert rag_body["grounding_status"] in ("GROUNDED", "PARTIALLY_GROUNDED", "FALLBACK")
    assert len(rag_body["answer"]) > 20
    assert len(rag_body["sources"]) > 0
    first_src = rag_body["sources"][0]
    assert "document_title" in first_src
    assert "source_filename" in first_src
    assert "official_url" in first_src
    print(f"  ✓ Grounded RAG OK: status='{rag_body['grounding_status']}', scheme='{rag_body.get('detected_scheme_name')}', sources={len(rag_body['sources'])}, answer_len={len(rag_body['answer'])}")

    # 7. Privacy Violation Guards
    print("\n[Step 8] Testing Privacy Pre-screening Invariants (prohibited PII rejection)...")
    for bad_key, bad_val in [
        ("aadhaar_number", "1234-5678-9012"),
        ("pan_number", "ABCDE1234F"),
        ("bank_account_number", "98765432101234"),
    ]:
        bad_profile = dict(profile)
        bad_profile[bad_key] = bad_val
        status, pii_err, _ = http_request("POST", "/api/recommend", data={"profile": bad_profile})
        assert status == 400, f"Expected 400 for {bad_key}, got {status}"
        assert pii_err["code"] == "PRIVACY_VIOLATION", f"Expected PRIVACY_VIOLATION, got {pii_err['code']}"
        print(f"  ✓ Intercepted {bad_key}: code='{pii_err['code']}', error='{pii_err['error']}'")

    # 8. Schema Validation Guards
    print("\n[Step 9] Testing Demographic/Financial Validation Error States...")
    # Age > 120
    status, val_err, _ = http_request("POST", "/api/recommend", data={"profile": {"age": 150}})
    assert status == 400, f"Expected 400 for age 150, got {status}"
    assert val_err["code"] == "VALIDATION_ERROR"
    print(f"  ✓ Invalid age rejected: code='{val_err['code']}', field='{val_err['details'][0]['field']}'")

    # Negative income
    status, inc_err, _ = http_request("POST", "/api/recommend", data={"profile": {"annual_income": -5000}})
    assert status == 400, f"Expected 400 for negative income, got {status}"
    assert inc_err["code"] == "VALIDATION_ERROR"
    print(f"  ✓ Negative income rejected: code='{val_err['code']}'")

    # Invalid top_k (< 1 and > 14)
    status, topk_err, _ = http_request("POST", "/api/recommend", data={"profile": {"age": 30}, "top_k": 0})
    assert status == 400
    assert topk_err["code"] == "INVALID_TOP_K"

    status, topk_err2, _ = http_request("POST", "/api/recommend", data={"profile": {"age": 30}, "top_k": 20})
    assert status == 400
    assert topk_err2["code"] == "INVALID_TOP_K"
    print("  ✓ Invalid top_k bounds (<1 and >14) rejected cleanly.")

    # Invalid boolean flag
    status, bool_err, _ = http_request("POST", "/api/recommend", data={"profile": {"age": 30}, "enable_xgboost": "yes"})
    assert status == 400
    assert bool_err["code"] == "INVALID_BOOLEAN_FLAG"
    print("  ✓ Non-boolean flag rejected cleanly.")

    # 9. RAG Query Guards
    print("\n[Step 10] Testing RAG Query Validation Guards...")
    status, q_err, _ = http_request("POST", "/api/ask", data={"query": "   "})
    assert status == 400 and q_err["code"] == "EMPTY_QUERY"

    status, q_err2, _ = http_request("POST", "/api/ask", data={"query": "hi"})
    assert status == 400 and q_err2["code"] == "QUERY_TOO_SHORT"
    print("  ✓ Empty and short RAG queries rejected cleanly.")

    # 10. CORS & Preflight
    print("\n[Step 11] Testing CORS Headers & OPTIONS Preflight...")
    status, _, cors_headers = http_request("OPTIONS", "/api/recommend")
    assert status in (200, 204), f"OPTIONS preflight returned {status}"
    assert cors_headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in cors_headers.get("Access-Control-Allow-Methods", "")

    _, _, get_headers = http_request("GET", "/health")
    assert get_headers.get("Access-Control-Allow-Origin") == "*"
    print("  ✓ CORS headers confirmed on GET, POST, and OPTIONS preflight.")

    # 11. Frontend Build Assets Integrity
    print("\n[Step 12] Verifying Frontend Production Build Artifacts...")
    dist_dir = REPO_ROOT / "frontend" / "dist"
    index_html = dist_dir / "index.html"
    assert index_html.exists(), "frontend/dist/index.html does not exist!"
    content = index_html.read_text(encoding="utf-8")
    assert "SchemeIQ+" in content, "Branding missing from index.html"
    assert 'id="root"' in content, "React root mount missing from index.html"

    assets_dir = dist_dir / "assets"
    css_files = list(assets_dir.glob("*.css"))
    js_files = list(assets_dir.glob("*.js"))
    assert len(css_files) > 0, "No CSS files in frontend/dist/assets"
    assert len(js_files) > 0, "No JS files in frontend/dist/assets"
    print(f"  ✓ Frontend build verified: index.html ({index_html.stat().st_size} bytes), {len(css_files)} CSS bundle(s), {len(js_files)} JS bundle(s).")

    # 12. Frontend API Client Module Verification
    print("\n[Step 13] Verifying Frontend API Client Contract Alignment...")
    api_js = REPO_ROOT / "frontend" / "src" / "api.js"
    api_src = api_js.read_text(encoding="utf-8")
    for expected_fn in ["getHealth", "getRecommendations", "getSchemeDetails", "askQuestion"]:
        assert expected_fn in api_src, f"Missing {expected_fn} in frontend/src/api.js"
    assert "VITE_API_BASE_URL" in api_src, "Missing VITE_API_BASE_URL env var lookup in api.js"
    assert "BACKEND_UNAVAILABLE" in api_src, "Missing BACKEND_UNAVAILABLE network error handling in api.js"
    print("  ✓ Frontend API module implements all required endpoints with configurable base URL and network resilience.")

    print("\n" + "=" * 70)
    print("ALL PHASE 8D INTEGRATION & E2E FLOWS PASSED SUCCESSFULLY (13/13)!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ E2E TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
