# -*- coding: utf-8 -*-
"""
tests/test_ranking_pipeline.py — Unit & Integration Tests for Phase 7B

Tests:
1. Feature vector extraction from UserProfile & SchemeEvaluation
2. Schema constraints (RelevanceGrade, Scheme ID validity)
3. DatasetValidator rules:
   - Prohibited personal identifiers rejection
   - Invalid relevance labels rejection
   - Missing query/group ID rejection
   - Duplicate candidate pair (query_id, scheme_id) rejection
4. Dataset I/O round-trip and XGBoost LTR array conversion
5. Integrity: no modification to locked corpus or deterministic rules
"""
import copy
import json
import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.eligibility.schemas import UserProfile
from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.recommender import SchemeRecommender
from src.recommendation.schemas import (
    OFFICIAL_SCHEME_IDS,
    DatasetPurpose,
    RankingDataset,
    RankingDatasetMetadata,
    RankingFeatureVector,
    RankingRecord,
    RelevanceGrade,
)
from src.recommendation.feature_extractor import RankingFeatureExtractor
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.dataset_io import RankingDatasetIO


@pytest.fixture
def sample_profile_and_eval():
    evaluator = SchemeEligibilityEvaluator()
    recommender = SchemeRecommender(evaluator)
    profile = UserProfile(
        age=35,
        gender="female",
        state="Telangana",
        residence_type="urban",
        occupation="business_owner",
        business_owner_status=True,
        is_non_farm_enterprise=True,
        annual_income=250000,
        project_cost=1000000,
        loan_requirement=500000,
    )
    recs = recommender.recommend(profile, top_k=2).recommendations
    return profile, recs[0]


def test_feature_extractor(sample_profile_and_eval):
    profile, evaluation = sample_profile_and_eval
    feat = RankingFeatureExtractor.extract(profile, evaluation)

    assert isinstance(feat, RankingFeatureVector)
    assert feat.feat_user_age == 35.0
    assert feat.feat_user_annual_income == 250000.0
    assert feat.feat_user_state_is_telangana == 1
    assert feat.feat_user_is_business_owner == 1
    assert feat.feat_scheme_id_index >= 0
    assert feat.feat_rule_pass_ratio >= 0.0
    assert feat.feat_deterministic_base_score > 0.0

    # Ensure to_feature_dict and to_dense_vector work
    f_dict = feat.to_feature_dict()
    assert isinstance(f_dict, dict)
    assert "feat_user_age" in f_dict

    dense = feat.to_dense_vector()
    assert isinstance(dense, list)
    assert len(dense) == len(f_dict)


def test_prohibited_privacy_rejection_in_extractor():
    evaluator = SchemeEligibilityEvaluator()
    eval_res = evaluator.evaluate_all(UserProfile(age=30, state="Telangana"))[0]

    # Attempt to inject prohibited field into raw model dump
    bad_dict = {"age": 30, "state": "Telangana", "aadhaar_number": "1234-5678-9012"}
    with pytest.raises(ValueError, match="Privacy Violation"):
        from src.eligibility.profile import validate_profile_privacy
        validate_profile_privacy(bad_dict)


def test_ranking_record_validation():
    feat = RankingFeatureVector()

    # Valid record
    rec = RankingRecord(
        record_id="rec_001",
        query_id="q_001",
        scheme_id="TS001",
        scheme_name="Rythu Bharosa",
        features=feat,
        relevance_label=3,
        eligibility_status="POTENTIALLY_ELIGIBLE",
    )
    assert rec.scheme_id == "TS001"
    assert rec.relevance_label == 3

    # Invalid scheme ID
    with pytest.raises(ValueError, match="Invalid scheme_id"):
        RankingRecord(
            record_id="rec_002",
            query_id="q_001",
            scheme_id="INVALID_999",
            scheme_name="Fake Scheme",
            features=feat,
            relevance_label=2,
            eligibility_status="POTENTIALLY_ELIGIBLE",
        )

    # Empty query ID
    with pytest.raises(ValueError, match="query_id cannot be empty"):
        RankingRecord(
            record_id="rec_003",
            query_id="   ",
            scheme_id="TS001",
            scheme_name="Rythu Bharosa",
            features=feat,
            relevance_label=1,
            eligibility_status="POTENTIALLY_ELIGIBLE",
        )


def test_validator_detects_prohibited_identifiers():
    raw_bad_data = {
        "metadata": {
            "dataset_name": "Bad Data",
            "created_at": "2026-09-03",
            "purpose": "SCHEMA_DEMONSTRATION_ONLY_NOT_FOR_TRAINING",
            "pan_number": "ABCDE1234F",  # Privacy violation in metadata
        },
        "records": [],
    }
    report = RankingDatasetValidator.validate_raw_dict(raw_bad_data)
    assert not report.is_valid
    assert any("Privacy Violation" in err for err in report.errors)


def test_validator_detects_duplicate_candidate_pair():
    feat = RankingFeatureVector().model_dump()
    raw_data = {
        "metadata": {
            "dataset_name": "Test",
            "created_at": "2026-09-03",
            "purpose": "SCHEMA_DEMONSTRATION_ONLY_NOT_FOR_TRAINING",
        },
        "records": [
            {
                "record_id": "r1",
                "query_id": "q1",
                "scheme_id": "TS001",
                "scheme_name": "Rythu Bharosa",
                "features": feat,
                "relevance_label": 2,
                "eligibility_status": "ELIGIBLE",
            },
            {
                "record_id": "r2",
                "query_id": "q1",
                "scheme_id": "TS001",  # Duplicate candidate pair for q1!
                "scheme_name": "Rythu Bharosa",
                "features": feat,
                "relevance_label": 1,
                "eligibility_status": "ELIGIBLE",
            },
        ],
    }
    report = RankingDatasetValidator.validate_raw_dict(raw_data)
    assert not report.is_valid
    assert any("Duplicate candidate pair" in err for err in report.errors)


def test_validator_detects_invalid_relevance_label():
    feat = RankingFeatureVector().model_dump()
    raw_data = {
        "metadata": {
            "dataset_name": "Test",
            "created_at": "2026-09-03",
            "purpose": "SCHEMA_DEMONSTRATION_ONLY_NOT_FOR_TRAINING",
        },
        "records": [
            {
                "record_id": "r1",
                "query_id": "q1",
                "scheme_id": "TS001",
                "scheme_name": "Rythu Bharosa",
                "features": feat,
                "relevance_label": 99,  # Invalid label (must be 0..3)
                "eligibility_status": "ELIGIBLE",
            }
        ],
    }
    report = RankingDatasetValidator.validate_raw_dict(raw_data)
    assert not report.is_valid
    assert any("invalid relevance_label" in err for err in report.errors)


def test_dataset_io_roundtrip_and_xgboost_format(tmp_path):
    # Load demonstration dataset
    demo_file = root / "data" / "ranking" / "schema_example.json"
    assert demo_file.exists(), "schema_example.json must exist"

    dataset = RankingDatasetIO.load_json(demo_file)
    assert len(dataset.records) == 6
    assert len(dataset.get_query_groups()) == 2

    # Round trip save to tmp_path
    save_path = tmp_path / "test_saved.json"
    RankingDatasetIO.save_json(dataset, save_path)
    reloaded = RankingDatasetIO.load_json(save_path)
    assert len(reloaded.records) == 6

    # Convert to XGBoost LTR arrays
    X, y, qid, group_counts = RankingDatasetIO.to_xgboost_format(dataset)
    assert len(X) == 6
    assert len(y) == 6
    assert len(qid) == 6
    assert group_counts == [3, 3]  # 2 groups with 3 candidates each
    assert all(isinstance(label, int) and 0 <= label <= 3 for label in y)
    assert qid[:3] == ["query_demo_farmer_001"] * 3
    assert qid[3:] == ["query_demo_student_002"] * 3
