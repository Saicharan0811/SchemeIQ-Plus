# -*- coding: utf-8 -*-
"""
src/api/app.py — Production REST API Layer for SchemeIQ+ (Phase 8A)

Implements a clean, modular Flask REST API around the completed SchemeIQ+ backend:
1. GET  /health                  — System health, version, model availability
2. POST /api/recommend           — Citizen eligibility assessment & explainable scheme recommendation
3. GET  /api/schemes/<scheme_id> — Official scheme details from verified dataset
4. POST /api/ask                 — Grounded question answering via Hybrid RAG
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from pydantic import ValidationError

from src.eligibility.profile import validate_profile_privacy
from src.eligibility.schemas import UserProfile
from src.eligibility.service import EligibilityService
from src.rag.rag_service import RAGService
from src.recommendation.reranker import XGBoostReRanker

logger = logging.getLogger(__name__)


def create_app(
    test_config: Optional[Dict[str, Any]] = None,
    eligibility_service: Optional[EligibilityService] = None,
    rag_service: Optional[RAGService] = None,
) -> Flask:
    """
    Application factory for the SchemeIQ+ Production REST API.
    """
    app = Flask(__name__)

    # Environment detection & production hardening (Phase 9A)
    env = os.environ.get("FLASK_ENV", os.environ.get("ENVIRONMENT", "development")).lower()
    is_production = env in ("production", "prod")
    if is_production:
        app.config["DEBUG"] = False
        app.config["TESTING"] = False
        app.config["ENV"] = "production"
    else:
        app.config.setdefault("DEBUG", False)

    # Apply configuration (test_config can override for testing)
    if test_config:
        app.config.update(test_config)

    # Secret key configuration from environment
    secret_key = os.environ.get("SECRET_KEY")
    if secret_key:
        app.config["SECRET_KEY"] = secret_key
    elif is_production and not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.urandom(24).hex()

    # Dependency injection / lazy instantiation
    elig_service = eligibility_service or EligibilityService()
    _rag_instance: Optional[RAGService] = rag_service

    def get_rag_service() -> RAGService:
        nonlocal _rag_instance
        if _rag_instance is None:
            _rag_instance = RAGService(allow_fallback=True)
        return _rag_instance

    # Configurable CORS support (Phase 9A)
    # Allows specific frontend origins in production while preserving local development
    cors_allowed_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "*").strip()
    cors_allowed_origins = [
        origin.strip().rstrip("/")
        for origin in cors_allowed_raw.split(",")
        if origin.strip()
    ]

    @app.after_request
    def add_cors_headers(response):
        if "*" in cors_allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            req_origin = request.headers.get("Origin")
            if req_origin:
                norm_req = req_origin.rstrip("/")
                if norm_req in cors_allowed_origins:
                    response.headers["Access-Control-Allow-Origin"] = req_origin
                    response.headers["Vary"] = "Origin"
            else:
                # Non-browser client or direct API access
                response.headers["Access-Control-Allow-Origin"] = cors_allowed_origins[0] if cors_allowed_origins else "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/api/<path:path>", methods=["OPTIONS"])
    def handle_options(path):
        return "", 204


    # -----------------------------------------------------------------------
    # Endpoint 1: Health
    # -----------------------------------------------------------------------
    @app.route("/health", methods=["GET"])
    def health():
        """
        Service health status, version, and XGBoost LTR model availability.
        """
        reranker = elig_service.recommender._get_reranker()
        return jsonify({
            "status": "healthy",
            "service": "SchemeIQ+ Production API",
            "version": "1.0.0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "xgboost_model_available": reranker.is_available(),
            "official_schemes_count": len(elig_service.evaluator.schemes),
        }), 200

    # -----------------------------------------------------------------------
    # Endpoint 2: Scheme Recommendation
    # -----------------------------------------------------------------------
    @app.route("/api/recommend", methods=["POST"])
    def recommend_schemes():
        """
        Evaluate statutory eligibility and rank schemes for an applicant profile.
        Supports both Phase 6 deterministic heuristic ranking and advisory XGBoost LTR re-ranking.
        """
        if not request.is_json:
            return jsonify({
                "error": "Request Content-Type must be application/json",
                "code": "INVALID_CONTENT_TYPE",
            }), 400

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a valid JSON object",
                "code": "MALFORMED_JSON",
            }), 400

        # Validate 'profile' field structure
        if "profile" in data:
            profile_data = data["profile"]
            if not isinstance(profile_data, dict):
                return jsonify({
                    "error": "'profile' must be a JSON dictionary of citizen attributes",
                    "code": "INVALID_PROFILE",
                }), 400
        else:
            # Fall back to root dict minus API control parameters
            profile_data = {
                k: v for k, v in data.items()
                if k not in ("top_k", "query", "include_rag_explanation", "enable_xgboost")
            }

        if not isinstance(profile_data, dict):
            return jsonify({
                "error": "'profile' must be a JSON dictionary of citizen attributes",
                "code": "INVALID_PROFILE",
            }), 400

        # Strict Privacy Validation: Reject Aadhaar, PAN, bank accounts before evaluation
        try:
            validate_profile_privacy(profile_data)
        except ValueError as e:
            return jsonify({
                "error": str(e),
                "code": "PRIVACY_VIOLATION",
            }), 400

        # Pydantic schema validation
        try:
            user_profile = UserProfile(**profile_data)
        except ValidationError as e:
            # Sanitize validation error details (strip URLs)
            errors = [
                {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
                for err in e.errors()
            ]
            return jsonify({
                "error": "Invalid profile attributes",
                "details": errors,
                "code": "VALIDATION_ERROR",
            }), 400

        # Strict Top-k validation (must be true int, not bool or float or string)
        if "top_k" in data:
            top_k_raw = data["top_k"]
            if isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int):
                return jsonify({
                    "error": "'top_k' must be an integer between 1 and 14",
                    "code": "INVALID_TOP_K",
                }), 400
            if top_k_raw < 1 or top_k_raw > 14:
                return jsonify({
                    "error": "'top_k' must be an integer between 1 and 14",
                    "code": "INVALID_TOP_K",
                }), 400
            top_k = top_k_raw
        else:
            top_k = 5

        # Query parameter validation
        query = data.get("query")
        if query is not None and not isinstance(query, str):
            return jsonify({
                "error": "'query' must be a string",
                "code": "INVALID_QUERY",
            }), 400

        # Strict Boolean flag validation
        if "include_rag_explanation" in data:
            val = data["include_rag_explanation"]
            if not isinstance(val, bool):
                return jsonify({
                    "error": "'include_rag_explanation' must be a boolean (true/false)",
                    "code": "INVALID_BOOLEAN_FLAG",
                }), 400
            include_rag = val
        else:
            include_rag = False

        if "enable_xgboost" in data:
            val = data["enable_xgboost"]
            if not isinstance(val, bool):
                return jsonify({
                    "error": "'enable_xgboost' must be a boolean (true/false)",
                    "code": "INVALID_BOOLEAN_FLAG",
                }), 400
            enable_xgboost = val
        else:
            enable_xgboost = False

        # Execute recommendation pipeline
        try:
            result = elig_service.recommend_for_profile(
                profile=user_profile,
                query=query,
                top_k=top_k,
                include_rag_explanation=include_rag,
                enable_xgboost=enable_xgboost,
            )
            return jsonify(result.model_dump()), 200
        except Exception as e:
            logger.error(f"Recommendation processing error: {e}", exc_info=True)
            return jsonify({
                "error": "An internal error occurred while processing recommendations",
                "code": "PROCESSING_ERROR",
            }), 500

    # -----------------------------------------------------------------------
    # Endpoint 3: Scheme Lookup
    # -----------------------------------------------------------------------
    @app.route("/api/schemes/<scheme_id>", methods=["GET"])
    def get_scheme(scheme_id: str):
        """
        Retrieve verified scheme metadata and statutory rules from verified dataset.
        """
        clean_id = scheme_id.strip().upper()
        scheme = elig_service.evaluator.get_scheme_by_id(clean_id)
        if not scheme:
            return jsonify({
                "error": f"Scheme '{clean_id}' not found. Must be one of the 14 official schemes.",
                "code": "SCHEME_NOT_FOUND",
            }), 404

        return jsonify({
            "scheme_id": scheme.get("scheme_id"),
            "scheme_name": scheme.get("scheme_name"),
            "scheme_scope": scheme.get("scheme_scope"),
            "category": scheme.get("category"),
            "official_url": scheme.get("official_url"),
            "source_authority": scheme.get("source_authority"),
            "source_filename": scheme.get("source_filename"),
            "rules": scheme.get("rules", []),
            "unstructured_criteria": scheme.get("unstructured_criteria", []),
        }), 200

    # -----------------------------------------------------------------------
    # Endpoint 4: Grounded RAG Question Answering
    # -----------------------------------------------------------------------
    @app.route("/api/ask", methods=["POST"])
    def ask_rag():
        """
        Answer citizen queries grounded in verified government documents via Hybrid RAG.
        """
        if not request.is_json:
            return jsonify({
                "error": "Request Content-Type must be application/json",
                "code": "INVALID_CONTENT_TYPE",
            }), 400

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a valid JSON object",
                "code": "MALFORMED_JSON",
            }), 400

        query = data.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return jsonify({
                "error": "'query' is required and must be a non-empty string",
                "code": "EMPTY_QUERY",
            }), 400

        clean_query = query.strip()
        if len(clean_query) < 5:
            return jsonify({
                "error": "Query is too short (minimum 5 characters)",
                "code": "QUERY_TOO_SHORT",
            }), 400
        if len(clean_query) > 2000:
            return jsonify({
                "error": "Query is too long (maximum 2000 characters)",
                "code": "QUERY_TOO_LONG",
            }), 400

        if "top_k" in data:
            top_k_raw = data["top_k"]
            if isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int):
                return jsonify({
                    "error": "'top_k' must be an integer between 1 and 20",
                    "code": "INVALID_TOP_K",
                }), 400
            if top_k_raw < 1 or top_k_raw > 20:
                return jsonify({
                    "error": "'top_k' must be an integer between 1 and 20",
                    "code": "INVALID_TOP_K",
                }), 400
            top_k = top_k_raw
        else:
            top_k = 5

        try:
            rag = get_rag_service()
            grounded_answer = rag.answer_query(query=clean_query, top_k=top_k)
            return jsonify(grounded_answer.model_dump()), 200
        except Exception as e:
            logger.error(f"RAG query processing error: {e}", exc_info=True)
            return jsonify({
                "error": "An internal error occurred while answering the query",
                "code": "RAG_PROCESSING_ERROR",
            }), 500

    # -----------------------------------------------------------------------
    # Global Error Handlers (Sanitized JSON)
    # -----------------------------------------------------------------------
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Resource not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        return jsonify({"error": "Method not allowed for this endpoint", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(500)
    def handle_internal_server_error(e):
        return jsonify({"error": "Internal server error", "code": "INTERNAL_SERVER_ERROR"}), 500

    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    env = os.environ.get("FLASK_ENV", os.environ.get("ENVIRONMENT", "development")).lower()
    is_prod = env in ("production", "prod")

    app = create_app()
    debug_mode = False if is_prod else app.config.get("DEBUG", False)
    print(f"SchemeIQ+ API Server starting on http://{host}:{port} (env={env}, debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)


