# -*- coding: utf-8 -*-
"""
src/eligibility/recommender.py — Explainable Scheme Recommendation Engine
Ranks all 14 official schemes using a deterministic, transparent scoring formula with named constants.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.profile import create_safe_profile_summary, normalize_profile
from src.eligibility.schemas import (
    EligibilityStatus,
    RecommendationResult,
    SchemeEvaluation,
    UserProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transparent Scoring Constants (Step 6)
# ---------------------------------------------------------------------------
BASE_SCORE_ELIGIBLE: float = 100.0
BASE_SCORE_POTENTIALLY_ELIGIBLE: float = 70.0
BASE_SCORE_INSUFFICIENT_INFO: float = 40.0
BASE_SCORE_NOT_APPLICABLE: float = 10.0
BASE_SCORE_NOT_ELIGIBLE: float = 0.0

WEIGHT_RULE_PASS_RATIO: float = 15.0       # Max +15 based on proportion of rules passed
WEIGHT_STATE_SCOPE_MATCH: float = 10.0     # +10 if state residency matches scheme scope
WEIGHT_CATEGORY_PREFERENCE: float = 10.0   # +10 if user specified category/scope matches
WEIGHT_MISSING_INFO_PENALTY: float = -3.0  # -3 per unknown required rule
WEIGHT_FAILED_RULE_PENALTY: float = -15.0  # -15 per failed rule


class SchemeRecommender:
    """
    Evaluates and ranks all 14 official schemes for a given user profile.
    """

    def __init__(
        self,
        evaluator: Optional[SchemeEligibilityEvaluator] = None,
        enable_xgboost: bool = False,
        reranker: Optional[Any] = None,
    ):
        self.evaluator = evaluator or SchemeEligibilityEvaluator()
        self.enable_xgboost = enable_xgboost
        self._reranker = reranker

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from src.recommendation.reranker import XGBoostReRanker
            self._reranker = XGBoostReRanker()
        return self._reranker

    @staticmethod
    def calculate_score(evaluation: SchemeEvaluation, profile: UserProfile) -> tuple[float, Dict[str, float]]:
        """
        Calculate deterministic recommendation score and breakdown.
        """
        breakdown: Dict[str, float] = {}

        # 1. Base score by status
        if evaluation.eligibility_status == EligibilityStatus.ELIGIBLE:
            base = BASE_SCORE_ELIGIBLE
        elif evaluation.eligibility_status == EligibilityStatus.POTENTIALLY_ELIGIBLE:
            base = BASE_SCORE_POTENTIALLY_ELIGIBLE
        elif evaluation.eligibility_status == EligibilityStatus.INSUFFICIENT_INFORMATION:
            base = BASE_SCORE_INSUFFICIENT_INFO
        elif evaluation.eligibility_status == EligibilityStatus.NOT_APPLICABLE:
            base = BASE_SCORE_NOT_APPLICABLE
        else:
            base = BASE_SCORE_NOT_ELIGIBLE
        breakdown["base_status_score"] = base

        # 2. Rule pass ratio boost
        total_rules = len(evaluation.matched_rules) + len(evaluation.failed_rules) + len(evaluation.unknown_rules)
        pass_ratio = len(evaluation.matched_rules) / max(1, total_rules)
        rule_boost = round(WEIGHT_RULE_PASS_RATIO * pass_ratio, 2)
        breakdown["rule_pass_ratio_boost"] = rule_boost

        # 3. State scope match
        state_boost = 0.0
        if evaluation.scheme_scope == "State" and str(profile.state).strip().lower() == "telangana":
            state_boost = WEIGHT_STATE_SCOPE_MATCH
        elif evaluation.scheme_scope == "Central":
            state_boost = WEIGHT_STATE_SCOPE_MATCH * 0.8
        breakdown["state_scope_match_boost"] = state_boost

        # 4. User category / scope preference match
        pref_boost = 0.0
        if profile.category_preference and profile.category_preference.lower() in evaluation.category.lower():
            pref_boost += WEIGHT_CATEGORY_PREFERENCE
        if profile.scope_preference and profile.scope_preference.lower() == evaluation.scheme_scope.lower():
            pref_boost += 5.0
        breakdown["preference_boost"] = pref_boost

        # 5. Domain specificity boost (rewards matching specific criteria like student, farmer, entrepreneur, housing)
        domain_matched_count = sum(
            1 for r in evaluation.matched_rules
            if r.field in [
                "student_status", "farmer_status", "business_owner_status",
                "pensioner_category", "owns_pucca_house", "is_non_farm_enterprise",
                "land_ownership_status"
            ]
        )
        domain_boost = round(8.0 * domain_matched_count, 2)
        breakdown["domain_specificity_boost"] = domain_boost

        # 6. Missing info penalty
        missing_pen = round(WEIGHT_MISSING_INFO_PENALTY * len(evaluation.missing_information), 2)
        breakdown["missing_info_penalty"] = missing_pen

        # 7. Failed rule penalty
        fail_pen = round(WEIGHT_FAILED_RULE_PENALTY * len(evaluation.failed_rules), 2)
        breakdown["failed_rule_penalty"] = fail_pen

        # Total score
        total = base + rule_boost + state_boost + pref_boost + domain_boost + missing_pen + fail_pen
        total = max(0.0, round(total, 2))
        breakdown["total_score"] = total

        return total, breakdown

    def recommend(
        self,
        profile: UserProfile,
        top_k: int = 14,
        enable_xgboost: Optional[bool] = None,
    ) -> RecommendationResult:
        """
        Evaluate and rank all schemes for the given profile.
        """
        normalized_prof = normalize_profile(profile)
        evaluations = self.evaluator.evaluate_all(normalized_prof)

        # Compute score for each scheme
        for ev in evaluations:
            score, breakdown = self.calculate_score(ev, normalized_prof)
            ev.score = score
            ev.scoring_breakdown = breakdown

        # Rank order priority:
        # 1. ELIGIBLE (score desc)
        # 2. POTENTIALLY_ELIGIBLE (score desc)
        # 3. INSUFFICIENT_INFORMATION (score desc)
        # 4. NOT_APPLICABLE (score desc)
        # 5. NOT_ELIGIBLE (score desc)
        status_priority = {
            EligibilityStatus.ELIGIBLE: 1,
            EligibilityStatus.POTENTIALLY_ELIGIBLE: 2,
            EligibilityStatus.INSUFFICIENT_INFORMATION: 3,
            EligibilityStatus.NOT_APPLICABLE: 4,
            EligibilityStatus.NOT_ELIGIBLE: 5,
        }

        # Deterministic heuristic baseline ordering (Phase 6)
        evaluations.sort(
            key=lambda e: (status_priority.get(e.eligibility_status, 99), -e.score, e.scheme_id)
        )

        use_xgboost = self.enable_xgboost if enable_xgboost is None else enable_xgboost
        if use_xgboost:
            try:
                reranker = self._get_reranker()
                evaluations = reranker.rerank(normalized_prof, evaluations)
            except Exception as e:
                logger.warning(
                    "XGBoost reranking delegation failed (%s). Retaining heuristic ordering.",
                    e,
                )

        status_counts = {}
        for ev in evaluations:
            st = ev.eligibility_status.value
            status_counts[st] = status_counts.get(st, 0) + 1

        top_recs = evaluations[:top_k]

        return RecommendationResult(
            user_profile_summary=create_safe_profile_summary(normalized_prof),
            total_schemes_evaluated=len(evaluations),
            recommendations=top_recs,
            status_counts=status_counts,
            evaluation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
