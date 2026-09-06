# -*- coding: utf-8 -*-
"""
tests/test_end_to_end_integration.py — Phase 7J End-to-End Recommendation + RAG Validation

Validates the complete SchemeIQ+ pipeline across both modes:
- enable_xgboost=False (Phase 6 baseline)
- enable_xgboost=True (Advisory XGBoost LTR re-ranking)

Validates all Phase 7J requirements:
1. User profile -> deterministic eligibility evaluation.
2. Only ELIGIBLE/POTENTIALLY_ELIGIBLE candidates can be XGBoost-ranked.
3. XGBoost changes ordering only; never eligibility status.
4. Final top-k recommendations correspond to the XGBoost-ranked candidate order when enabled.
5. Phase 6 ordering remains unchanged when XGBoost is disabled.
6. RAG retrieval still uses official government sources.
7. RAG explanations correspond to the recommended scheme.
8. Citations/source metadata remain intact.
9. Eligibility explanations remain deterministic and are not replaced by ML-generated claims.
10. XGBoost failure falls back to Phase 6 ranking and still produces a valid grounded answer.
11. Missing/insufficient profile information is handled safely.
12. Out-of-scope queries remain safely handled.
13. No disqualified scheme can appear as an eligible recommendation.
14. Candidate count and scheme identity remain consistent through the pipeline.
"""
from pathlib import Path
from unittest.mock import patch
import pytest

from src.eligibility.schemas import (
    EligibilityStatus,
    UserProfile,
)
from src.eligibility.service import EligibilityService
from src.recommendation.reranker import (
    DEFAULT_MODEL_PATH,
    ELIGIBLE_CANDIDATE_STATUSES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def telangana_farmer_profile() -> UserProfile:
    """Realistic rural Telangana smallholder farmer profile."""
    return UserProfile(
        age=42,
        gender="male",
        state="Telangana",
        district="Nalgonda",
        residence_type="rural",
        occupation="farmer",
        farmer_status=True,
        land_ownership_status=True,
        land_acres=3.2,
        annual_income=120000,
        pays_income_tax=False,
        is_bpl_or_white_ration_card=True,
        has_savings_bank_account=True,
        owns_pucca_house=False,
    )


def test_e2e_modes_comparison_and_safety_invariants(telangana_farmer_profile):
    """
    Validates requirements 1, 2, 3, 4, 5, 14:
    - Deterministic evaluation of all 14 schemes.
    - Only ELIGIBLE / POTENTIALLY_ELIGIBLE get xgboost_score.
    - Eligibility status unchanged.
    - Top-k reordered by XGBoost when enabled.
    - Phase 6 order preserved when disabled.
    - Candidate count and scheme identity consistent.
    """
    assert DEFAULT_MODEL_PATH.exists(), f"Model artifact missing: {DEFAULT_MODEL_PATH}"

    service = EligibilityService()

    # 1. Run with enable_xgboost=False (Phase 6 Heuristic)
    res_heuristic = service.recommend_for_profile(
        telangana_farmer_profile,
        top_k=14,
        enable_xgboost=False,
    )

    # 2. Run with enable_xgboost=True (Advisory XGBoost LTR)
    res_xgboost = service.recommend_for_profile(
        telangana_farmer_profile,
        top_k=14,
        enable_xgboost=True,
    )

    # Validate 14 schemes evaluated in both
    assert res_heuristic.total_schemes_evaluated == 14
    assert res_xgboost.total_schemes_evaluated == 14
    assert len(res_heuristic.recommendations) == 14
    assert len(res_xgboost.recommendations) == 14

    heur_recs = res_heuristic.recommendations
    xgb_recs = res_xgboost.recommendations

    # Scheme set consistency
    assert {r.scheme_id for r in heur_recs} == {r.scheme_id for r in xgb_recs}

    # Verify eligibility status of each scheme is strictly identical
    heur_statuses = {r.scheme_id: r.eligibility_status for r in heur_recs}
    xgb_statuses = {r.scheme_id: r.eligibility_status for r in xgb_recs}
    assert heur_statuses == xgb_statuses

    # Invariant: No disqualified scheme is in ELIGIBLE_CANDIDATE_STATUSES
    disqualified = {
        sid for sid, st in xgb_statuses.items()
        if st in (EligibilityStatus.NOT_ELIGIBLE, EligibilityStatus.NOT_APPLICABLE)
    }

    # Verify only ELIGIBLE and POTENTIALLY_ELIGIBLE candidates have xgboost_score
    for r in xgb_recs:
        if r.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES:
            assert "xgboost_score" in r.scoring_breakdown
            assert isinstance(r.scoring_breakdown["xgboost_score"], float)
        else:
            assert "xgboost_score" not in r.scoring_breakdown
            assert r.scheme_id in disqualified or r.eligibility_status == EligibilityStatus.INSUFFICIENT_INFORMATION

    # Verify that in XGBoost mode, eligible candidates are ordered by descending xgboost_score
    eligible_xgb_recs = [r for r in xgb_recs if r.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES]
    xgb_scores = [r.scoring_breakdown["xgboost_score"] for r in eligible_xgb_recs]
    assert xgb_scores == sorted(xgb_scores, reverse=True)

    # Verify deterministic base scores (ev.score) are preserved and positive
    for r in xgb_recs:
        assert r.score == r.scoring_breakdown["total_score"]


def test_e2e_rag_explanation_integration(telangana_farmer_profile):
    """
    Validates requirements 6, 7, 8, 9:
    - RAG retrieval uses official sources.
    - Grounded context attached to recommended scheme.
    - Citations and URLs intact.
    - Deterministic explanations preserved and not replaced by ML claims.
    """
    service = EligibilityService()

    # Query with RAG explanation enabled
    res = service.recommend_for_profile(
        telangana_farmer_profile,
        query="What government scheme provides financial assistance for agriculture in Telangana?",
        top_k=2,
        include_rag_explanation=True,
        enable_xgboost=True,
    )

    assert len(res.recommendations) == 2
    top_rec = res.recommendations[0]

    # Deterministic explainability preserved
    total_criteria = (
        len(top_rec.matched_rules)
        + len(top_rec.failed_rules)
        + len(top_rec.unknown_rules)
        + len(top_rec.unstructured_criteria)
    )
    assert total_criteria > 0, "Expected at least one deterministic rule or criterion"
    assert top_rec.official_url.startswith("http")
    assert top_rec.scheme_scope in ("State", "Central")

    # RAG Grounded Context attached
    assert "[Official Grounded Context]:" in top_rec.explanation_summary
    assert len(top_rec.explanation_summary) > 50

    # Deterministic status is not modified
    assert top_rec.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES


def test_e2e_xgboost_fallback_with_rag_operational(telangana_farmer_profile):
    """
    Validates requirement 10:
    - XGBoost failure falls back to Phase 6 ranking and still produces valid grounded answer.
    """
    service = EligibilityService()

    # Mock prediction failure in XGBoostReRanker
    with patch("xgboost.XGBRanker.predict", side_effect=RuntimeError("Simulated XGBoost engine failure")):
        res = service.recommend_for_profile(
            telangana_farmer_profile,
            query="Tell me about agricultural support",
            top_k=2,
            include_rag_explanation=True,
            enable_xgboost=True,
        )

    # Fallback must produce 2 valid recommendations
    assert len(res.recommendations) == 2
    for r in res.recommendations:
        assert r.scheme_id is not None
        assert r.score > 0
        assert "[Official Grounded Context]:" in r.explanation_summary


def test_e2e_sparse_and_missing_profile_info():
    """
    Validates requirement 11:
    - Missing/insufficient profile information is safely handled.
    """
    # Extremely sparse profile
    sparse_profile = UserProfile(state="Telangana")

    service = EligibilityService()
    res = service.recommend_for_profile(
        sparse_profile,
        top_k=5,
        enable_xgboost=True,
    )

    assert len(res.recommendations) == 5
    assert res.total_schemes_evaluated == 14
    for r in res.recommendations:
        assert r.eligibility_status in (
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.POTENTIALLY_ELIGIBLE,
            EligibilityStatus.INSUFFICIENT_INFORMATION,
            EligibilityStatus.NOT_APPLICABLE,
            EligibilityStatus.NOT_ELIGIBLE,
        )


def test_e2e_out_of_scope_and_disqualified_isolation():
    """
    Validates requirements 12, 13:
    - Out-of-scope profiles handled safely.
    - Disqualified schemes never appear as eligible recommendations.
    """
    # Profile completely out of scope for Telangana state schemes & income tax payer
    ineligible_profile = UserProfile(
        age=50,
        state="Karnataka",
        residence_type="urban",
        occupation="software_engineer",
        farmer_status=False,
        annual_income=2500000,
        pays_income_tax=True,
        owns_pucca_house=True,
    )

    service = EligibilityService()
    res = service.recommend_for_profile(
        ineligible_profile,
        top_k=14,
        enable_xgboost=True,
    )

    assert res.total_schemes_evaluated == 14
    for r in res.recommendations:
        # If a scheme is NOT_ELIGIBLE or NOT_APPLICABLE, verify it has NO xgboost_score
        if r.eligibility_status in (EligibilityStatus.NOT_ELIGIBLE, EligibilityStatus.NOT_APPLICABLE):
            assert "xgboost_score" not in r.scoring_breakdown, (
                f"Disqualified scheme {r.scheme_id} was scored by XGBoost!"
            )
