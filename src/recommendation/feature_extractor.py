# -*- coding: utf-8 -*-
"""
src/recommendation/feature_extractor.py — Feature Extraction for SchemeIQ+ LTR (Phase 7B)

Converts (UserProfile, SchemeEvaluation) pairs into clean, normalized, tabular feature vectors
suitable for XGBoost Learning-to-Rank, preserving missingness and respecting privacy constraints.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.eligibility.profile import normalize_profile, validate_profile_privacy
from src.eligibility.schemas import EligibilityStatus, SchemeEvaluation, UserProfile
from src.recommendation.schemas import (
    RankingFeatureVector,
    SCHEME_ID_TO_INDEX,
)

logger = logging.getLogger(__name__)

# Categorical mapping constants
GENDER_MAP = {"female": 1, "male": 2, "transgender": 3, "other": 4}
RESIDENCE_MAP = {"urban": 1, "rural": 2}
CATEGORY_MAP = {"sc": 1, "st": 2, "bc": 3, "ebc": 4, "minority": 5, "general": 6, "obc": 3}
STATUS_TIER_MAP = {
    EligibilityStatus.NOT_ELIGIBLE: 0,
    EligibilityStatus.NOT_APPLICABLE: 1,
    EligibilityStatus.INSUFFICIENT_INFORMATION: 2,
    EligibilityStatus.POTENTIALLY_ELIGIBLE: 3,
    EligibilityStatus.ELIGIBLE: 4,
}


class RankingFeatureExtractor:
    """
    Extracts tabular features for (UserProfile, SchemeEvaluation) candidate pairs.
    """

    @staticmethod
    def _encode_binary_flag(val: Optional[bool]) -> int:
        if val is None:
            return -1
        return 1 if val else 0

    @classmethod
    def extract(
        cls,
        raw_profile: UserProfile,
        evaluation: SchemeEvaluation,
    ) -> RankingFeatureVector:
        """
        Extract numerical and encoded feature vector for a given profile and scheme evaluation.

        Args:
            raw_profile: The user profile.
            evaluation: The deterministic SchemeEvaluation output for the candidate scheme.

        Returns:
            RankingFeatureVector populated with features.
        """
        # 1. Enforce strict privacy validation
        validate_profile_privacy(raw_profile.model_dump())
        profile = normalize_profile(raw_profile)

        # 2. Extract continuous features (None is preserved for XGBoost missing split)
        age = float(profile.age) if profile.age is not None else None
        income = float(profile.annual_income) if profile.annual_income is not None else None
        land_acres = float(profile.land_acres) if profile.land_acres is not None else None
        loan_req = float(profile.loan_requirement) if profile.loan_requirement is not None else None
        project_cost = float(profile.project_cost) if profile.project_cost is not None else None
        pension = float(profile.monthly_pension) if profile.monthly_pension is not None else None

        # 3. Encode categorical features
        gender_code = GENDER_MAP.get(str(profile.gender).lower().strip(), 0) if profile.gender else 0
        residence_code = RESIDENCE_MAP.get(str(profile.residence_type).lower().strip(), 0) if profile.residence_type else 0
        cat_code = CATEGORY_MAP.get(str(profile.category).lower().strip(), 0) if profile.category else 0
        is_telangana = 1 if str(profile.state).strip().lower() == "telangana" else 0

        # 4. Encode binary flags (-1=unknown, 0=False, 1=True)
        is_farmer = cls._encode_binary_flag(profile.farmer_status)
        has_land = cls._encode_binary_flag(profile.land_ownership_status)
        is_student = cls._encode_binary_flag(profile.student_status)
        is_business = cls._encode_binary_flag(profile.business_owner_status)
        is_pensioner = cls._encode_binary_flag(profile.pensioner_status)
        owns_pucca = cls._encode_binary_flag(profile.owns_pucca_house)
        has_bpl = cls._encode_binary_flag(profile.is_bpl_or_white_ration_card)
        pays_tax = cls._encode_binary_flag(profile.pays_income_tax)
        is_pregnant = cls._encode_binary_flag(profile.is_pregnant_or_newborn)
        has_savings = cls._encode_binary_flag(profile.has_savings_bank_account)

        # 5. Scheme static features
        scheme_idx = SCHEME_ID_TO_INDEX.get(evaluation.scheme_id, 0)
        scheme_scope = 1 if evaluation.scheme_scope.lower() == "state" else 2

        # 6. Evaluation metrics
        matched_cnt = len(evaluation.matched_rules)
        failed_cnt = len(evaluation.failed_rules)
        unknown_cnt = len(evaluation.unknown_rules)
        total_rules = matched_cnt + failed_cnt + unknown_cnt
        pass_ratio = round(matched_cnt / max(1, total_rules), 4)
        missing_cnt = len(evaluation.missing_information)
        base_score = float(evaluation.score)
        tier_code = STATUS_TIER_MAP.get(evaluation.eligibility_status, 2)

        # 7. Match & preference features
        cat_pref_match = 0
        if profile.category_preference and profile.category_preference.lower() in evaluation.category.lower():
            cat_pref_match = 1

        scope_pref_match = 0
        if profile.scope_preference and profile.scope_preference.lower() == evaluation.scheme_scope.lower():
            scope_pref_match = 1

        domain_cnt = sum(
            1 for r in evaluation.matched_rules
            if r.field in [
                "student_status", "farmer_status", "business_owner_status",
                "pensioner_category", "owns_pucca_house", "is_non_farm_enterprise",
                "land_ownership_status"
            ]
        )

        return RankingFeatureVector(
            feat_user_age=age,
            feat_user_annual_income=income,
            feat_user_land_acres=land_acres,
            feat_user_loan_requirement=loan_req,
            feat_user_project_cost=project_cost,
            feat_user_monthly_pension=pension,
            feat_user_gender=gender_code,
            feat_user_residence_type=residence_code,
            feat_user_category=cat_code,
            feat_user_state_is_telangana=is_telangana,
            feat_user_is_farmer=is_farmer,
            feat_user_has_land_ownership=has_land,
            feat_user_is_student=is_student,
            feat_user_is_business_owner=is_business,
            feat_user_is_pensioner=is_pensioner,
            feat_user_owns_pucca_house=owns_pucca,
            feat_user_has_bpl_card=has_bpl,
            feat_user_pays_income_tax=pays_tax,
            feat_user_is_pregnant_or_newborn=is_pregnant,
            feat_user_has_savings_bank_account=has_savings,
            feat_scheme_id_index=scheme_idx,
            feat_scheme_scope=scheme_scope,
            feat_rule_pass_ratio=pass_ratio,
            feat_rules_matched_count=matched_cnt,
            feat_rules_failed_count=failed_cnt,
            feat_rules_unknown_count=unknown_cnt,
            feat_missing_fields_count=missing_cnt,
            feat_deterministic_base_score=base_score,
            feat_eligibility_tier=tier_code,
            feat_category_pref_match=cat_pref_match,
            feat_scope_pref_match=scope_pref_match,
            feat_domain_specificity_count=domain_cnt,
        )
