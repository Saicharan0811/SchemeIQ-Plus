# -*- coding: utf-8 -*-
"""
src/eligibility/service.py — Production Eligibility & Recommendation Service
Orchestrates profile validation, deterministic rule evaluation, explainable ranking,
and optional grounded RAG explanation integration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.explanation import ExplanationFormatter
from src.eligibility.profile import normalize_profile, validate_profile_privacy
from src.eligibility.recommender import SchemeRecommender
from src.eligibility.schemas import RecommendationResult, SchemeEvaluation, UserProfile
from src.rag.rag_service import RAGService

logger = logging.getLogger(__name__)


class EligibilityService:
    """
    Production service for personalized eligibility assessment and scheme recommendation.

    Design Rule:
      - Deterministic rule engine is the SOLE authority for eligibility decisions and scores.
      - LLM / RAG is strictly used for natural language explanation grounded in retrieved official chunks.
    """

    def __init__(
        self,
        evaluator: Optional[SchemeEligibilityEvaluator] = None,
        recommender: Optional[SchemeRecommender] = None,
        rag_service: Optional[RAGService] = None,
        enable_xgboost: bool = False,
    ):
        self.evaluator = evaluator or SchemeEligibilityEvaluator()
        self.enable_xgboost = enable_xgboost
        self.recommender = recommender or SchemeRecommender(
            evaluator=self.evaluator,
            enable_xgboost=enable_xgboost,
        )
        self._rag_service = rag_service

    def _get_rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._rag_service = RAGService(allow_fallback=True)
        return self._rag_service

    def evaluate_profile(self, profile: UserProfile) -> List[SchemeEvaluation]:
        """
        Evaluate all 14 official schemes against the user profile.
        """
        validate_profile_privacy(profile.model_dump())
        normalized = normalize_profile(profile)
        return self.evaluator.evaluate_all(normalized)

    def recommend_for_profile(
        self,
        profile: UserProfile,
        query: Optional[str] = None,
        top_k: int = 5,
        include_rag_explanation: bool = False,
        enable_xgboost: Optional[bool] = None,
    ) -> RecommendationResult:
        """
        Full recommendation pipeline:
          1. Validate profile privacy (no sensitive IDs)
          2. Normalize profile
          3. Evaluate all 14 schemes deterministically
          4. Compute transparent scores & rank (with optional advisory XGBoost re-ranking)
          5. Optionally retrieve official RAG context for top scheme explanations
          6. Return structured RecommendationResult

        Args:
            profile: User profile instance.
            query: Optional natural language question (e.g. "Which agricultural scheme helps me buy equipment?").
            top_k: Number of top recommendations to return.
            include_rag_explanation: If True, uses RAGService to fetch official context.
            enable_xgboost: If True, applies advisory XGBoost LTR re-ranking to eligible candidates.

        Returns:
            RecommendationResult with ranked schemes and full explainability evidence.
        """
        validate_profile_privacy(profile.model_dump())
        normalized = normalize_profile(profile)

        # If user passed a query, we can use it to set category/preference if not already set
        if query:
            q_lower = query.lower()
            if "farm" in q_lower or "agri" in q_lower:
                normalized.category_preference = normalized.category_preference or "Agriculture"
            elif "health" in q_lower or "medical" in q_lower or "hospital" in q_lower:
                normalized.category_preference = normalized.category_preference or "Healthcare"
            elif "business" in q_lower or "loan" in q_lower or "startup" in q_lower:
                normalized.category_preference = normalized.category_preference or "Entrepreneurship"
            elif "student" in q_lower or "scholarship" in q_lower or "study" in q_lower:
                normalized.category_preference = normalized.category_preference or "Education"
            elif "house" in q_lower or "home" in q_lower or "housing" in q_lower:
                normalized.category_preference = normalized.category_preference or "Housing"

        # Deterministic evaluation and ranking (with optional advisory XGBoost re-ranking)
        use_xgboost = self.enable_xgboost if enable_xgboost is None else enable_xgboost
        result = self.recommender.recommend(normalized, top_k=top_k, enable_xgboost=use_xgboost)

        # Optional grounded RAG context enhancement
        if include_rag_explanation and result.recommendations:
            rag = self._get_rag_service()
            for rec in result.recommendations:
                # Query RAG service for official guidelines of this scheme
                try:
                    rag_query = f"What are the official eligibility conditions and benefits for {rec.scheme_name}?"
                    grounded_ans = rag.answer_query(rag_query, top_k=2)
                    if grounded_ans.answer:
                        # Append RAG official text to explanation summary without altering deterministic status/score
                        rec.explanation_summary += f"\n\n[Official Grounded Context]:\n{grounded_ans.answer[:400]}..."
                except Exception as e:
                    logger.warning(f"Failed to fetch RAG explanation for {rec.scheme_id}: {e}")

        return result
