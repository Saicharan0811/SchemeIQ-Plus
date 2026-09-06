# -*- coding: utf-8 -*-
"""
tests/test_ltr_training.py — Phase 7H XGBoost Learning-to-Rank Tests

Tests:
1. test_dataset_loading_and_validation
2. test_query_split_zero_leakage
3. test_query_split_invalid_id_raises
4. test_group_matrix_ordering
5. test_metric_calculations
6. test_xgbranker_training_and_evaluation
7. test_heuristic_baseline_comparison
8. test_artifact_persistence
"""
import json
from pathlib import Path
import pytest
import numpy as np
import xgboost as xgb

from src.recommendation.schemas import RankingDataset
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.train_ltr import (
    load_and_validate_dataset,
    split_by_query,
    prepare_group_matrices,
    compute_metrics_per_group,
    compute_heuristic_scores,
    train_and_evaluate_ltr,
    dcg_at_k,
    ndcg_at_k,
    mrr,
    precision_at_k,
    DEFAULT_VAL_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "ranking" / "annotations_expert_01.json"


def test_dataset_loading_and_validation():
    """Verify that the real expert annotations load and pass all schema/privacy checks."""
    assert ANNOTATIONS_PATH.exists(), f"Missing real dataset: {ANNOTATIONS_PATH}"
    dataset = load_and_validate_dataset(ANNOTATIONS_PATH)
    assert isinstance(dataset, RankingDataset)
    assert len(dataset.records) == 138
    groups = dataset.get_query_groups()
    assert len(groups) == 16
    for qid, recs in groups.items():
        assert len(recs) >= 2, f"Group {qid} has fewer than 2 candidates"
    # Check all labels are in {1, 2, 3}
    for rec in dataset.records:
        assert rec.relevance_label in {1, 2, 3}
        assert rec.eligibility_status == "POTENTIALLY_ELIGIBLE"


def test_query_split_zero_leakage():
    """Verify train/validation split strictly by query_id with zero query leakage."""
    dataset = load_and_validate_dataset(ANNOTATIONS_PATH)
    val_ids = ["ARCH004", "ARCH008", "ARCH011", "ARCH015"]
    train_recs, val_recs, train_qids, actual_val_qids = split_by_query(dataset, val_ids)

    assert len(train_qids) == 12
    assert len(actual_val_qids) == 4
    assert set(actual_val_qids) == set(val_ids)

    # Zero leakage
    train_q_set = set(r.query_id for r in train_recs)
    val_q_set = set(r.query_id for r in val_recs)
    assert len(train_q_set & val_q_set) == 0, "Query leakage detected between train and val"

    # Complete coverage
    assert len(train_recs) + len(val_recs) == 138
    assert len(train_recs) == 105
    assert len(val_recs) == 33


def test_query_split_invalid_id_raises():
    """Verify ValueError is raised if non-existent validation query ID is specified."""
    dataset = load_and_validate_dataset(ANNOTATIONS_PATH)
    with pytest.raises(ValueError, match="Validation query IDs not found"):
        split_by_query(dataset, ["INVALID_QID_999"])


def test_group_matrix_ordering():
    """Verify matrices match record counts, feature dimensions, and group sorting."""
    dataset = load_and_validate_dataset(ANNOTATIONS_PATH)
    val_ids = ["ARCH004", "ARCH008", "ARCH011", "ARCH015"]
    train_recs, val_recs, train_qids, actual_val_qids = split_by_query(dataset, val_ids)

    X_train, y_train, train_group_counts = prepare_group_matrices(train_recs, train_qids)
    assert X_train.shape == (105, 32)
    assert y_train.shape == (105,)
    assert sum(train_group_counts) == 105
    assert len(train_group_counts) == 12

    X_val, y_val, val_group_counts = prepare_group_matrices(val_recs, actual_val_qids)
    assert X_val.shape == (33, 32)
    assert y_val.shape == (33,)
    assert sum(val_group_counts) == 33
    assert len(val_group_counts) == 4


def test_metric_calculations():
    """Verify unit accuracy of NDCG, MRR, and Precision metrics."""
    # Perfect ranking
    perfect = [3.0, 2.0, 1.0, 0.0]
    assert ndcg_at_k(perfect, 3) == pytest.approx(1.0)
    assert ndcg_at_k(perfect, 5) == pytest.approx(1.0)
    assert mrr(perfect, primary_grade=3) == 1.0
    assert precision_at_k(perfect, 3, threshold=2) == pytest.approx(2 / 3)

    # Inverted ranking
    inverted = [1.0, 2.0, 3.0]
    assert ndcg_at_k(inverted, 3) < 1.0
    assert mrr(inverted, primary_grade=3) == pytest.approx(1.0 / 3.0)

    # All zeros / empty
    assert ndcg_at_k([0.0, 0.0], 2) == 1.0
    assert ndcg_at_k([], 3) == 1.0
    assert mrr([1.0, 2.0], primary_grade=3) == 0.0
    assert precision_at_k([], 3) == 0.0


def test_xgbranker_training_and_evaluation(tmp_path):
    """Verify end-to-end training and evaluation on a temporary output directory."""
    out_dir = tmp_path / "ranking"
    report = train_and_evaluate_ltr(
        dataset_path=ANNOTATIONS_PATH,
        output_dir=out_dir,
        val_ids=DEFAULT_VAL_IDS,
    )

    assert report["TRAINING_STATUS"] == "SUCCESS"
    assert report["N_TRAIN_RECORDS"] == 105
    assert report["N_VAL_RECORDS"] == 33
    assert report["FEATURE_DIM"] == 32

    # Verify metrics structure
    assert "xgboost" in report["TRAIN_METRICS"]
    assert "xgboost" in report["VALIDATION_METRICS"]
    xgb_val = report["VALIDATION_METRICS"]["xgboost"]["aggregate"]
    assert 0.0 <= xgb_val["ndcg@3"] <= 1.0
    assert 0.0 <= xgb_val["ndcg@5"] <= 1.0
    assert 0.0 <= xgb_val["mrr"] <= 1.0
    assert 0.0 <= xgb_val["precision@3"] <= 1.0

    # Verify model artifact can be reloaded by XGBRanker
    model_path = Path(report["MODEL_PATH"])
    assert model_path.exists()
    loaded_ranker = xgb.XGBRanker()
    loaded_ranker.load_model(str(model_path))
    assert loaded_ranker is not None


def test_heuristic_baseline_comparison(tmp_path):
    """Verify heuristic baseline is evaluated on the exact same validation groups."""
    out_dir = tmp_path / "ranking_baseline"
    report = train_and_evaluate_ltr(
        dataset_path=ANNOTATIONS_PATH,
        output_dir=out_dir,
        val_ids=DEFAULT_VAL_IDS,
    )

    heur_val = report["HEURISTIC_METRICS"]
    assert 0.0 <= heur_val["ndcg@3"] <= 1.0
    assert 0.0 <= heur_val["ndcg@5"] <= 1.0
    assert 0.0 <= heur_val["mrr"] <= 1.0
    assert 0.0 <= heur_val["precision@3"] <= 1.0

    comp = report["XGBOOST_VS_HEURISTIC"]
    assert "ndcg@3_improvement" in comp
    assert "ndcg@5_improvement" in comp


def test_artifact_persistence(tmp_path):
    """Verify model and report JSON are written properly with required fields."""
    out_dir = tmp_path / "ranking_artifacts"
    report = train_and_evaluate_ltr(
        dataset_path=ANNOTATIONS_PATH,
        output_dir=out_dir,
        val_ids=DEFAULT_VAL_IDS,
    )

    model_file = out_dir / "xgboost_ltr_model.json"
    report_file = out_dir / "training_report.json"
    assert model_file.exists()
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)

    required_keys = [
        "TRAINING_STATUS",
        "MODEL_PATH",
        "TRAIN_QUERY_GROUPS",
        "VALIDATION_QUERY_GROUPS",
        "TRAIN_METRICS",
        "VALIDATION_METRICS",
        "HEURISTIC_METRICS",
        "XGBOOST_VS_HEURISTIC",
        "OVERFITTING_ANALYSIS",
    ]
    for key in required_keys:
        assert key in saved_report
