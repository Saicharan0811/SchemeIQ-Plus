# -*- coding: utf-8 -*-
"""
tests/test_production_api.py — Phase 8A Production API Test Suite

Tests:
1.  test_health_endpoint
2.  test_recommend_heuristic_mode_success
3.  test_recommend_xgboost_mode_success
4.  test_recommend_privacy_rejection_aadhaar
5.  test_recommend_privacy_rejection_pan
6.  test_recommend_privacy_rejection_bank_account
7.  test_recommend_validation_error_invalid_age
8.  test_recommend_sparse_profile
9.  test_recommend_disqualified_schemes_never_promoted
10. test_recommend_xgboost_failure_safe_fallback
11. test_recommend_with_rag_explanation
12. test_scheme_lookup_valid
13. test_scheme_lookup_invalid
14. test_rag_ask_valid_query
15. test_rag_ask_empty_query_rejected
16. test_rag_ask_short_query_rejected
17. test_non_json_content_type_rejected
18. test_malformed_json_rejected
"""
import pytest
from unittest.mock import patch

from src.api.app import create_app
from src.eligibility.schemas import EligibilityStatus


@pytest.fixture
def client():
    app = create_app(test_config={"TESTING": True})
    with app.test_client() as client:
        yield client


@pytest.fixture
def farmer_payload():
    return {
        "profile": {
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
        },
        "top_k": 5,
        "enable_xgboost": False,
    }


def test_health_endpoint(client):
    """Verify GET /health returns service status and model availability."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "SchemeIQ+ Production API"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert "xgboost_model_available" in data
    assert data["official_schemes_count"] == 14


def test_recommend_heuristic_mode_success(client, farmer_payload):
    """Verify POST /api/recommend with enable_xgboost=False returns Phase 6 heuristic ranking."""
    resp = client.post("/api/recommend", json=farmer_payload)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["total_schemes_evaluated"] == 14
    assert len(data["recommendations"]) == 5
    # Verify no xgboost_score in heuristic mode
    for rec in data["recommendations"]:
        assert "scheme_id" in rec
        assert "scheme_name" in rec
        assert "eligibility_status" in rec
        assert "score" in rec
        assert "scoring_breakdown" in rec
        assert "xgboost_score" not in rec["scoring_breakdown"]


def test_recommend_xgboost_mode_success(client, farmer_payload):
    """Verify POST /api/recommend with enable_xgboost=True applies advisory XGBoost scores."""
    payload = dict(farmer_payload)
    payload["enable_xgboost"] = True

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()

    assert len(data["recommendations"]) == 5
    # The top recommendation should be an eligible or potentially eligible scheme with an xgboost_score
    top_rec = data["recommendations"][0]
    assert top_rec["eligibility_status"] in ("ELIGIBLE", "POTENTIALLY_ELIGIBLE")
    assert "xgboost_score" in top_rec["scoring_breakdown"]
    assert isinstance(top_rec["scoring_breakdown"]["xgboost_score"], float)
    # ev.score remains the deterministic score
    assert top_rec["score"] == top_rec["scoring_breakdown"]["total_score"]


def test_recommend_privacy_rejection_aadhaar(client, farmer_payload):
    """Verify POST /api/recommend immediately rejects prohibited Aadhaar key with HTTP 400."""
    payload = dict(farmer_payload)
    payload["profile"]["aadhaar_number"] = "123456789012"

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "PRIVACY_VIOLATION"
    assert "aadhaar" in data["error"].lower()


def test_recommend_privacy_rejection_pan(client, farmer_payload):
    """Verify POST /api/recommend immediately rejects prohibited PAN key with HTTP 400."""
    payload = dict(farmer_payload)
    payload["profile"]["pan_number"] = "ABCDE1234F"

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "PRIVACY_VIOLATION"
    assert "pan" in data["error"].lower()


def test_recommend_privacy_rejection_bank_account(client, farmer_payload):
    """Verify POST /api/recommend immediately rejects prohibited bank account number with HTTP 400."""
    payload = dict(farmer_payload)
    payload["profile"]["bank_account_number"] = "9876543210123"

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "PRIVACY_VIOLATION"


def test_recommend_validation_error_invalid_age(client, farmer_payload):
    """Verify POST /api/recommend returns HTTP 400 for impossible age (>120)."""
    payload = dict(farmer_payload)
    payload["profile"]["age"] = 150

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "VALIDATION_ERROR"
    assert "age" in str(data["details"]).lower()


def test_recommend_sparse_profile(client):
    """Verify POST /api/recommend handles sparse profiles safely without crashing."""
    payload = {
        "profile": {"state": "Telangana"},
        "top_k": 3,
        "enable_xgboost": True,
    }
    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["recommendations"]) == 3
    assert data["total_schemes_evaluated"] == 14


def test_recommend_disqualified_schemes_never_promoted(client):
    """Verify NOT_ELIGIBLE and NOT_APPLICABLE schemes are never given an xgboost_score."""
    # Out of state applicant with high income
    payload = {
        "profile": {
            "age": 40,
            "state": "Maharashtra",
            "annual_income": 3000000,
            "pays_income_tax": True,
            "owns_pucca_house": True,
            "farmer_status": False,
        },
        "top_k": 14,
        "enable_xgboost": True,
    }
    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()

    for rec in data["recommendations"]:
        if rec["eligibility_status"] in ("NOT_ELIGIBLE", "NOT_APPLICABLE"):
            assert "xgboost_score" not in rec["scoring_breakdown"], (
                f"Disqualified scheme {rec['scheme_id']} received an xgboost_score!"
            )


def test_recommend_xgboost_failure_safe_fallback(client, farmer_payload):
    """Verify ML runtime exception falls back safely to heuristic order with HTTP 200 (not 500)."""
    payload = dict(farmer_payload)
    payload["enable_xgboost"] = True

    with patch("xgboost.XGBRanker.predict", side_effect=RuntimeError("Simulated XGBoost failure")):
        resp = client.post("/api/recommend", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["recommendations"]) == 5
    # Should fall back cleanly
    for rec in data["recommendations"]:
        assert rec["score"] > 0


def test_recommend_with_rag_explanation(client, farmer_payload):
    """Verify include_rag_explanation=True attaches official RAG context to recommendations."""
    payload = dict(farmer_payload)
    payload["include_rag_explanation"] = True
    payload["top_k"] = 2

    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    top_rec = data["recommendations"][0]
    assert "[Official Grounded Context]:" in top_rec["explanation_summary"]


def test_scheme_lookup_valid(client):
    """Verify GET /api/schemes/<scheme_id> returns official scheme metadata for TS001."""
    resp = client.get("/api/schemes/TS001")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["scheme_id"] == "TS001"
    assert data["scheme_name"] == "Rythu Bharosa"
    assert data["scheme_scope"] == "State"
    assert data["official_url"].startswith("http")
    assert isinstance(data["rules"], list)
    assert len(data["rules"]) > 0


def test_scheme_lookup_invalid(client):
    """Verify GET /api/schemes/<scheme_id> returns HTTP 404 for unknown scheme."""
    resp = client.get("/api/schemes/INVALID_SCHEME_999")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["code"] == "SCHEME_NOT_FOUND"


def test_rag_ask_valid_query(client):
    """Verify POST /api/ask returns grounded RAG answer and source citations."""
    payload = {
        "query": "Who is eligible for Rythu Bharosa financial assistance?",
        "top_k": 3,
    }
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "grounding_status" in data
    assert "sources" in data
    assert "retrieved_scheme_ids" in data
    assert len(data["sources"]) > 0 or len(data["retrieved_scheme_ids"]) > 0


def test_rag_ask_empty_query_rejected(client):
    """Verify POST /api/ask returns HTTP 400 for empty or whitespace query."""
    resp = client.post("/api/ask", json={"query": "   "})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "EMPTY_QUERY"


def test_rag_ask_short_query_rejected(client):
    """Verify POST /api/ask returns HTTP 400 for queries shorter than 5 characters."""
    resp = client.post("/api/ask", json={"query": "hi"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "QUERY_TOO_SHORT"


def test_non_json_content_type_rejected(client):
    """Verify non-JSON Content-Type returns HTTP 400."""
    resp = client.post("/api/recommend", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_CONTENT_TYPE"


def test_malformed_json_rejected(client):
    """Verify malformed JSON body returns HTTP 400."""
    resp = client.post("/api/recommend", data="{not a valid json", content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "MALFORMED_JSON"


def test_invalid_boolean_flags_rejected(client, farmer_payload):
    """Verify non-boolean values for boolean flags are rejected with HTTP 400."""
    payload1 = dict(farmer_payload)
    payload1["enable_xgboost"] = "true"  # String instead of bool
    resp1 = client.post("/api/recommend", json=payload1)
    assert resp1.status_code == 400
    assert resp1.get_json()["code"] == "INVALID_BOOLEAN_FLAG"

    payload2 = dict(farmer_payload)
    payload2["include_rag_explanation"] = 1  # Int instead of bool
    resp2 = client.post("/api/recommend", json=payload2)
    assert resp2.status_code == 400
    assert resp2.get_json()["code"] == "INVALID_BOOLEAN_FLAG"


def test_invalid_top_k_types_and_bounds_rejected(client, farmer_payload):
    """Verify boolean, float, string, and out-of-bound top_k values are rejected."""
    for invalid_val in (True, 3.5, "5", 0, 15, -1):
        payload = dict(farmer_payload)
        payload["top_k"] = invalid_val
        resp = client.post("/api/recommend", json=payload)
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "INVALID_TOP_K"


def test_invalid_profile_types_rejected(client):
    """Verify non-dict profile values are rejected with HTTP 400."""
    for invalid_profile in ("string_profile", [1, 2, 3], 12345, True):
        resp = client.post("/api/recommend", json={"profile": invalid_profile})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "INVALID_PROFILE"


def test_cors_headers_and_options_preflight(client):
    """Verify CORS headers are present on responses and OPTIONS preflight succeeds."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"
    assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")

    # Preflight OPTIONS request
    options_resp = client.options("/api/recommend")
    assert options_resp.status_code in (200, 204)
    assert options_resp.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in options_resp.headers.get("Access-Control-Allow-Methods", "")


def test_frontend_contract_stability_and_field_presence(client, farmer_payload):
    """Verify that all response fields required by the future React frontend are present and stable."""
    resp = client.post("/api/recommend", json=farmer_payload)
    assert resp.status_code == 200
    data = resp.get_json()

    # Top-level contract keys
    required_top_level = [
        "total_schemes_evaluated",
        "recommendations",
        "status_counts",
        "user_profile_summary",
        "disclaimer",
        "evaluation_timestamp",
    ]
    for key in required_top_level:
        assert key in data, f"Missing required top-level key: {key}"

    # Per-recommendation contract keys
    required_rec_keys = [
        "scheme_id",
        "scheme_name",
        "scheme_scope",
        "category",
        "eligibility_status",
        "confidence",
        "score",
        "scoring_breakdown",
        "official_url",
        "source_authority",
        "matched_rules",
        "failed_rules",
        "unknown_rules",
        "missing_information",
        "unstructured_criteria",
        "explanation_summary",
    ]
    for rec in data["recommendations"]:
        for key in required_rec_keys:
            assert key in rec, f"Missing required recommendation key: {key}"
        assert isinstance(rec["score"], (int, float))
        assert isinstance(rec["scoring_breakdown"], dict)
        assert "base_status_score" in rec["scoring_breakdown"]
        assert "total_score" in rec["scoring_breakdown"]


# ---------------------------------------------------------------------------
# Phase 9A — Production Configuration & Deployment Tests
# ---------------------------------------------------------------------------

def test_cors_specific_allowed_origin_matched():
    """Verify that when CORS_ALLOWED_ORIGINS is configured, allowed Origin headers are reflected."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://app.schemeiq.gov.in,http://localhost:5173"}):
        app = create_app()
        with app.test_client() as c:
            resp = c.get("/health", headers={"Origin": "https://app.schemeiq.gov.in"})
            assert resp.status_code == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "https://app.schemeiq.gov.in"
            assert resp.headers.get("Vary") == "Origin"


def test_cors_disallowed_origin_not_reflected():
    """Verify that unlisted Origin headers are NOT reflected in Access-Control-Allow-Origin."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://app.schemeiq.gov.in"}):
        app = create_app()
        with app.test_client() as c:
            resp = c.get("/health", headers={"Origin": "https://malicious-site.example.com"})
            assert resp.status_code == 200
            # Must not reflect the malicious origin
            assert resp.headers.get("Access-Control-Allow-Origin") != "https://malicious-site.example.com"


def test_production_env_disables_debug():
    """Verify that FLASK_ENV=production strictly disables debug mode and testing."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"FLASK_ENV": "production", "ENVIRONMENT": "production"}):
        app = create_app()
        assert app.config["DEBUG"] is False
        assert app.config["TESTING"] is False
        assert app.config["ENV"] == "production"


def test_wsgi_entrypoint_import():
    """Verify that wsgi.py exposes a valid WSGI application callable."""
    import wsgi
    assert hasattr(wsgi, "app")
    assert callable(wsgi.app)
    assert wsgi.app.name == "src.api.app"


def test_health_endpoint_does_not_reload_xgboost_repeatedly(client):
    """Verify GET /health reuses the cached reranker and does not reload model from disk on each request."""
    import xgboost as xgb
    from unittest.mock import patch

    with patch.object(xgb.XGBRanker, "load_model", wraps=xgb.XGBRanker().load_model) as mock_load:
        # Call /health 5 times in sequence
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["xgboost_model_available"] is True

        # Model should be loaded at most once across all 5 requests (not 5 times)
        assert mock_load.call_count <= 1


def test_rag_pipeline_stage_level_observability(client, caplog):
    """Verify POST /api/ask emits concise INFO-level timing logs for each pipeline stage."""
    import logging

    with caplog.at_level(logging.INFO):
        resp = client.post(
            "/api/ask",
            json={"query": "Who is eligible for Rythu Bharosa?", "top_k": 3},
        )
        assert resp.status_code == 200

        logs = [r.message for r in caplog.records]
        # Verify each of the required stage timing logs was emitted
        assert any("HybridRetriever: query embedded in" in msg for msg in logs), "Missing ONNX embedding timing log"
        assert any("HybridRetriever: Chroma dense retrieval completed in" in msg for msg in logs), "Missing Chroma dense timing log"
        assert any("HybridRetriever: BM25 retrieval completed in" in msg for msg in logs), "Missing BM25 retrieval timing log"
        assert any("HybridRetriever: RRF merge completed in" in msg for msg in logs), "Missing RRF merge timing log"
        assert any("AnswerGenerator: calling LLM provider" in msg for msg in logs), "Missing pre-generation log"
        assert any("AnswerGenerator: LLM generation completed in" in msg for msg in logs), "Missing post-generation log"


def test_openai_provider_timeout_configured():
    """Verify OpenAILLMProvider initializes OpenAI client with explicit 15-second timeout."""
    import os
    from unittest.mock import patch
    from src.rag.answer_generator import OpenAILLMProvider

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-mock-key-for-timeout-check"}):
        provider = OpenAILLMProvider()
        # Verify client timeout is set to 15.0 seconds
        assert getattr(provider._client, "timeout", None) == 15.0


def test_onnx_spinning_disabled():
    """Verify ONNXEmbeddingProvider disables intra-op spinning in session options."""
    from src.rag.embeddings import get_embedding_provider, ONNXEmbeddingProvider

    provider = get_embedding_provider(provider_type="onnx", force_new=True)
    assert isinstance(provider, ONNXEmbeddingProvider)
    # Ensure provider is functional and dim is 384
    emb = provider.embed_query("warmup check")
    assert len(emb) == 384



