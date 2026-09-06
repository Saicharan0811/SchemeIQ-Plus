# -*- coding: utf-8 -*-
"""
src/eligibility/ — SchemeIQ+ Personalized Eligibility and Scheme Recommendation Engine
"""

from src.eligibility.schemas import (
    EligibilityStatus,
    RuleStatus,
    RuleType,
    UserProfile,
    RuleResult,
    SchemeEvaluation,
    RecommendationResult,
    DATA_MINIMIZATION_NOTICE,
)
from src.eligibility.profile import (
    normalize_profile,
    validate_profile_privacy,
    create_safe_profile_summary,
)
from src.eligibility.rule_engine import RuleEngine
from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.recommender import SchemeRecommender
from src.eligibility.explanation import ExplanationFormatter
from src.eligibility.service import EligibilityService

__all__ = [
    "EligibilityStatus",
    "RuleStatus",
    "RuleType",
    "UserProfile",
    "RuleResult",
    "SchemeEvaluation",
    "RecommendationResult",
    "DATA_MINIMIZATION_NOTICE",
    "normalize_profile",
    "validate_profile_privacy",
    "create_safe_profile_summary",
    "RuleEngine",
    "SchemeEligibilityEvaluator",
    "SchemeRecommender",
    "ExplanationFormatter",
    "EligibilityService",
]
