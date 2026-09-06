# -*- coding: utf-8 -*-
"""
tests/test_controlled_ranking.py — Phase 7I Controlled XGBoost Ranking & Safety Tests

Tests:
1. test_only_eligible_and_potential_reach_model
2. test_not_eligible_never_reaches_model
3. test_not_applicable_never_reaches_model
4. test_eligibility_status_unchanged_before_and_after
5. test_candidate_count_unchanged
6. test_scheme_details_preserved
7. test_deterministic_score_tie_breaking
8. test_missing_model_fallback
9. test_prediction_failure_fallback
10. test_feature_shape_failure_fallback
11. test_repeated_calls_produce_deterministic_results
12. test_enable_xgboost_false_exactly_preserves_phase6_behavior
13. test_real_model_integration_advisory_ranking
"""
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.recommender import SchemeRecommender
from src.eligibility.schemas import (
    EligibilityStatus,
    SchemeEvaluation,
    UserProfile,
)
from src.eligibility.service import EligibilityService
from src.recommendation.reranker import (
    DEFAULT_MODEL_PATH,
    ELIGIBLE_CANDIDATE_STATUSES,
    XGBoostReRanker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def farmer_profile() -> UserProfile:
    """Telangana farmer profile producing a mix of eligible, potential, and ineligible schemes."""
    return UserProfile(
        age=38,
        gender="male",
        state="Telangana",
        residence_type="rural",
        occupation="farmer",
        farmer_status=True,
        land_ownership_status=True,
        land_acres=4.5,
        annual_income=180000,
        pays_income_tax=False,
        is_institutional_landholder=False,
    )


@pytest.fixture
def senior_profile() -> UserProfile:
    """Senior citizen pensioner profile."""
    return UserProfile(
        age=68,
        gender="female",
        state="Telangana",
        residence_type="rural",
        occupation="unemployed",
        pensioner_status=True,
        pensioner_category="old_age",
        annual_income=40000,
        owns_pucca_house=False,
        is_bpl_or_white_ration_card=True,
    )


def test_only_eligible_and_potential_reach_model(farmer_profile):
    """Verify that ONLY ELIGIBLE and POTENTIALLY_ELIGIBLE candidates reach feature extraction."""
    evaluator = SchemeEligibilityEvaluator()
    recommender = SchemeRecommender(evaluator=evaluator)
    evaluations = evaluator.evaluate_all(farmer_profile)

    reranker = XGBoostReRanker()

    # Intercept RankingFeatureExtractor.extract to track which schemes are extracted
    extracted_schemes = []
    from src.recommendation.feature_extractor import RankingFeatureExtractor
    original_extract = RankingFeatureExtractor.extract

    def tracking_extract(prof, ev):
        extracted_schemes.append((ev.scheme_id, ev.eligibility_status))
        return original_extract(prof, ev)

    with patch.object(RankingFeatureExtractor, "extract", side_effect=tracking_extract):
        reranked = reranker.rerank(farmer_profile, evaluations)

    assert len(extracted_schemes) > 0, "Expected at least one eligible candidate to be extracted"
    for sid, status in extracted_schemes:
        assert status in ELIGIBLE_CANDIDATE_STATUSES, (
            f"Scheme {sid} with status {status} reached feature extraction!"
        )


def test_not_eligible_never_reaches_model(farmer_profile):
    """Verify NOT_ELIGIBLE schemes are strictly excluded from XGBoost prediction."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    not_eligible = [e for e in evaluations if e.eligibility_status == EligibilityStatus.NOT_ELIGIBLE]
    assert len(not_eligible) > 0, "Fixture should have at least one NOT_ELIGIBLE scheme"
    not_eligible_ids = {e.scheme_id for e in not_eligible}

    reranker = XGBoostReRanker()
    scored_schemes = set()

    original_predict = reranker._ensure_model_loaded().predict

    def tracking_predict(X):
        return original_predict(X)

    with patch.object(reranker._ensure_model_loaded(), "predict", side_effect=tracking_predict):
        reranked = reranker.rerank(farmer_profile, evaluations)

    # Verify none of the not_eligible candidates received an xgboost_score
    for r in reranked:
        if r.scheme_id in not_eligible_ids:
            assert "xgboost_score" not in r.scoring_breakdown, (
                f"NOT_ELIGIBLE scheme {r.scheme_id} was assigned an xgboost_score!"
            )


def test_not_applicable_never_reaches_model():
    """Verify NOT_APPLICABLE schemes are strictly excluded from XGBoost scoring."""
    # Profile from Andhra Pradesh -> State schemes from Telangana should be NOT_APPLICABLE
    out_of_state_profile = UserProfile(
        age=30,
        state="Andhra Pradesh",
        occupation="farmer",
        farmer_status=True,
    )
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(out_of_state_profile)

    not_applicable = [e for e in evaluations if e.eligibility_status == EligibilityStatus.NOT_APPLICABLE]
    assert len(not_applicable) > 0, "Out-of-state profile should have NOT_APPLICABLE schemes"
    not_applicable_ids = {e.scheme_id for e in not_applicable}

    reranker = XGBoostReRanker()
    reranked = reranker.rerank(out_of_state_profile, evaluations)

    for r in reranked:
        if r.scheme_id in not_applicable_ids:
            assert "xgboost_score" not in r.scoring_breakdown, (
                f"NOT_APPLICABLE scheme {r.scheme_id} was assigned an xgboost_score!"
            )


def test_eligibility_status_unchanged_before_and_after(farmer_profile):
    """Verify XGBoost cannot alter eligibility_status of any candidate."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    status_before = {e.scheme_id: e.eligibility_status for e in evaluations}

    reranker = XGBoostReRanker()
    reranked = reranker.rerank(farmer_profile, evaluations)

    status_after = {e.scheme_id: e.eligibility_status for e in reranked}

    assert status_before == status_after, "Eligibility statuses were modified by reranking!"


def test_candidate_count_unchanged(farmer_profile):
    """Verify total candidate count is preserved exactly before and after re-ranking."""
    recommender = SchemeRecommender()
    res_heuristic = recommender.recommend(farmer_profile, top_k=14, enable_xgboost=False)
    res_xgboost = recommender.recommend(farmer_profile, top_k=14, enable_xgboost=True)

    assert len(res_heuristic.recommendations) == 14
    assert len(res_xgboost.recommendations) == 14
    assert res_heuristic.total_schemes_evaluated == res_xgboost.total_schemes_evaluated == 14


def test_scheme_details_preserved(farmer_profile):
    """Verify matched rules, failed rules, URLs, and explanation details are strictly preserved."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    orig_details = {
        e.scheme_id: {
            "matched_rules": len(e.matched_rules),
            "failed_rules": len(e.failed_rules),
            "unknown_rules": len(e.unknown_rules),
            "url": e.official_url,
            "category": e.category,
            "scope": e.scheme_scope,
        }
        for e in evaluations
    }

    reranker = XGBoostReRanker()
    reranked = reranker.rerank(farmer_profile, evaluations)

    for r in reranked:
        expected = orig_details[r.scheme_id]
        assert len(r.matched_rules) == expected["matched_rules"]
        assert len(r.failed_rules) == expected["failed_rules"]
        assert len(r.unknown_rules) == expected["unknown_rules"]
        assert r.official_url == expected["url"]
        assert r.category == expected["category"]
        assert r.scheme_scope == expected["scope"]


def test_deterministic_score_tie_breaking(farmer_profile):
    """Verify deterministic scheme_id ascending tie-breaking when XGBoost scores are tied."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    reranker = XGBoostReRanker()

    # Mock predict to return identical scores (0.0) for all candidates
    with patch.object(reranker._ensure_model_loaded(), "predict", side_effect=lambda X: np.zeros(len(X))):
        reranked = reranker.rerank(farmer_profile, evaluations)

    eligible_reranked = [
        r.scheme_id for r in reranked if r.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES
    ]
    # When scores are identical (0.0), tie-breaking must sort strictly by scheme_id ascending
    assert eligible_reranked == sorted(eligible_reranked), (
        f"Tie-breaking failed: {eligible_reranked} is not sorted alphabetically by scheme_id"
    )


def test_missing_model_fallback(farmer_profile):
    """Verify missing model file triggers safe fallback to heuristic ordering without raising."""
    non_existent_path = Path("models/ranking/non_existent_model_9999.json")
    reranker = XGBoostReRanker(model_path=non_existent_path)

    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    # Rerank should log warning and return original candidates safely
    result = reranker.rerank(farmer_profile, evaluations)
    assert len(result) == len(evaluations)
    # Check that no error was raised and all scheme IDs match
    assert [r.scheme_id for r in result] == [e.scheme_id for e in evaluations]


def test_prediction_failure_fallback(farmer_profile):
    """Verify runtime exception during model.predict triggers safe heuristic fallback."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    reranker = XGBoostReRanker()
    with patch.object(reranker._ensure_model_loaded(), "predict", side_effect=RuntimeError("CUDA OOM mock")):
        result = reranker.rerank(farmer_profile, evaluations)

    assert len(result) == len(evaluations)
    assert [r.scheme_id for r in result] == [e.scheme_id for e in evaluations]


def test_feature_shape_failure_fallback(farmer_profile):
    """Verify matrix shape irregularity triggers safe heuristic fallback."""
    evaluator = SchemeEligibilityEvaluator()
    evaluations = evaluator.evaluate_all(farmer_profile)

    reranker = XGBoostReRanker()
    # Mock RankingFeatureExtractor.extract to return an invalid vector length
    mock_vec = MagicMock()
    mock_vec.to_dense_vector.return_value = [1.0, 2.0]  # Only 2 features instead of 32

    from src.recommendation.feature_extractor import RankingFeatureExtractor
    with patch.object(RankingFeatureExtractor, "extract", return_value=mock_vec):
        result = reranker.rerank(farmer_profile, evaluations)

    assert len(result) == len(evaluations)
    assert [r.scheme_id for r in result] == [e.scheme_id for e in evaluations]


def test_repeated_calls_produce_deterministic_results(farmer_profile):
    """Verify that multiple executions on the same profile produce identical rankings and scores."""
    recommender = SchemeRecommender(enable_xgboost=True)

    res1 = recommender.recommend(farmer_profile, top_k=14)
    res2 = recommender.recommend(farmer_profile, top_k=14)

    sids1 = [r.scheme_id for r in res1.recommendations]
    sids2 = [r.scheme_id for r in res2.recommendations]
    assert sids1 == sids2

    scores1 = [r.scoring_breakdown.get("xgboost_score") for r in res1.recommendations]
    scores2 = [r.scoring_breakdown.get("xgboost_score") for r in res2.recommendations]
    assert scores1 == scores2


def test_enable_xgboost_false_exactly_preserves_phase6_behavior(farmer_profile):
    """Verify default enable_xgboost=False produces identical output to pure Phase 6 recommender."""
    recommender_default = SchemeRecommender(enable_xgboost=False)
    recommender_explicit_false = SchemeRecommender()

    res_default = recommender_default.recommend(farmer_profile, top_k=14)
    res_explicit = recommender_explicit_false.recommend(farmer_profile, top_k=14, enable_xgboost=False)

    sids_default = [r.scheme_id for r in res_default.recommendations]
    sids_explicit = [r.scheme_id for r in res_explicit.recommendations]
    assert sids_default == sids_explicit

    # Ensure no xgboost_score was added
    for r in res_default.recommendations:
        assert "xgboost_score" not in r.scoring_breakdown


def test_real_model_integration_advisory_ranking(senior_profile):
    """Integration test using the real trained XGBoost model artifact."""
    assert DEFAULT_MODEL_PATH.exists(), f"Missing real model artifact: {DEFAULT_MODEL_PATH}"

    service = EligibilityService(enable_xgboost=True)
    res = service.recommend_for_profile(senior_profile, top_k=5)

    assert len(res.recommendations) == 5
    # The top recommendation should be an eligible or potentially eligible scheme
    top = res.recommendations[0]
    assert top.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES
    assert "xgboost_score" in top.scoring_breakdown
    assert isinstance(top.scoring_breakdown["xgboost_score"], float)

    # ev.score is still the deterministic score
    assert top.score > 0.0
    assert top.score == top.scoring_breakdown["total_score"]
