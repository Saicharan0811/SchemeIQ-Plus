# -*- coding: utf-8 -*-
"""
src/recommendation/train_ltr.py — XGBoost Learning-to-Rank Training & Evaluation (Phase 7H)

Implements:
1. Load + validate real expert-labeled dataset via existing RankingDatasetIO / RankingDatasetValidator.
2. Split strictly by query_id — zero query leakage between train and validation.
3. Build numpy matrices sorted by query group order for XGBRanker.
4. Train XGBRanker(objective="rank:ndcg") with conservative hyper-parameters.
5. Evaluate NDCG@3, NDCG@5, MRR (primary=3), Precision@3 (relevant>=2) on train + validation.
6. Compare XGBoost against Phase 6 heuristic baseline (feat_deterministic_base_score).
7. Overfitting analysis.
8. Save model JSON + training report JSON to models/ranking/.

DO NOT modify eligibility, RAG, retrieval, corpus, ChromaDB, or live recommendation logic.
DO NOT fabricate or modify any relevance labels.
"""
from __future__ import annotations

import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

# Fix Windows cp1252 encoding issues for print output
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import RankingDataset, RankingRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "ranking" / "annotations_expert_01.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "ranking"

DEFAULT_VAL_IDS: List[str] = ["ARCH004", "ARCH008", "ARCH011", "ARCH015"]

XGBOOST_PARAMS: Dict[str, Any] = {
    "objective": "rank:ndcg",
    "n_estimators": 50,
    "learning_rate": 0.08,
    "max_depth": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "tree_method": "hist",
}

# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    r = np.asarray(relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return float(np.sum((2.0 ** r - 1.0) / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Normalized DCG at rank k. Returns 1.0 if ideal DCG is 0."""
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0.0:
        return 1.0
    return dcg_at_k(relevances, k) / ideal_dcg


def mrr(relevances: List[float], primary_grade: int = 3) -> float:
    """Mean Reciprocal Rank: reciprocal of rank of first primary-grade candidate."""
    for rank, rel in enumerate(relevances, start=1):
        if rel >= primary_grade:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevances: List[float], k: int, threshold: int = 2) -> float:
    """Precision@k: fraction of top-k results that are relevant (>= threshold)."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return float(sum(1 for r in top_k if r >= threshold)) / len(top_k)


def compute_metrics_per_group(
    records: List[RankingRecord],
    scores: List[float],
) -> Dict[str, Any]:
    """
    Compute NDCG@3, NDCG@5, MRR, Precision@3 per query group.
    """
    if len(records) != len(scores):
        raise ValueError(
            f"records ({len(records)}) and scores ({len(scores)}) must have equal length"
        )

    group_data: Dict[str, List[Tuple[float, float]]] = {}
    for rec, score in zip(records, scores):
        group_data.setdefault(rec.query_id, []).append((rec.relevance_label, score))

    per_query: Dict[str, Dict[str, float]] = {}
    for qid, pairs in group_data.items():
        sorted_pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
        ranked_rels = [p[0] for p in sorted_pairs]
        per_query[qid] = {
            "ndcg@3": round(ndcg_at_k(ranked_rels, 3), 4),
            "ndcg@5": round(ndcg_at_k(ranked_rels, 5), 4),
            "mrr": round(mrr(ranked_rels, primary_grade=3), 4),
            "precision@3": round(precision_at_k(ranked_rels, 3, threshold=2), 4),
            "n_candidates": len(ranked_rels),
        }

    aggregate = {
        "ndcg@3": round(float(np.mean([v["ndcg@3"] for v in per_query.values()])), 4),
        "ndcg@5": round(float(np.mean([v["ndcg@5"] for v in per_query.values()])), 4),
        "mrr": round(float(np.mean([v["mrr"] for v in per_query.values()])), 4),
        "precision@3": round(
            float(np.mean([v["precision@3"] for v in per_query.values()])), 4
        ),
        "n_groups": len(per_query),
    }

    return {"per_query": per_query, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def load_and_validate_dataset(dataset_path: Path) -> RankingDataset:
    """Load and validate via existing RankingDatasetIO. Raises on failure."""
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = RankingDatasetIO.load_json(dataset_path)
    report = RankingDatasetValidator.validate_dataset(dataset)
    if not report.is_valid:
        raise ValueError(f"Dataset validation failed: {report.errors}")
    logger.info(
        f"Dataset loaded: {len(dataset.records)} records, "
        f"{report.total_query_groups} query groups"
    )
    return dataset


# ---------------------------------------------------------------------------
# Train / Validation split — strictly by query_id, zero leakage
# ---------------------------------------------------------------------------

def split_by_query(
    dataset: RankingDataset,
    val_ids: List[str],
) -> Tuple[List[RankingRecord], List[RankingRecord], List[str], List[str]]:
    """Split records strictly by query_id with zero leakage guarantee."""
    groups = dataset.get_query_groups()
    all_qids = list(groups.keys())

    missing = [vid for vid in val_ids if vid not in groups]
    if missing:
        raise ValueError(
            f"Validation query IDs not found in dataset: {missing}. Available: {all_qids}"
        )

    val_set = set(val_ids)
    train_qids = [qid for qid in all_qids if qid not in val_set]
    actual_val_qids = [qid for qid in all_qids if qid in val_set]

    train_records: List[RankingRecord] = []
    for qid in train_qids:
        train_records.extend(groups[qid])

    val_records: List[RankingRecord] = []
    for qid in actual_val_qids:
        val_records.extend(groups[qid])

    # Zero-leakage sanity check
    train_q_set = set(r.query_id for r in train_records)
    val_q_set = set(r.query_id for r in val_records)
    overlap = train_q_set & val_q_set
    if overlap:
        raise RuntimeError(f"Query leakage detected: {overlap}")

    logger.info(
        f"Split: {len(train_records)} train records ({len(train_qids)} groups) | "
        f"{len(val_records)} val records ({len(actual_val_qids)} groups)"
    )
    return train_records, val_records, train_qids, actual_val_qids


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

def prepare_group_matrices(
    records: List[RankingRecord],
    group_order: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build XGBoost-compatible (X, y, group_counts) sorted by group_order."""
    grouped: Dict[str, List[RankingRecord]] = {}
    for r in records:
        grouped.setdefault(r.query_id, []).append(r)

    X_rows: List[List[Optional[float]]] = []
    y_vals: List[int] = []
    group_counts: List[int] = []

    for qid in group_order:
        group_records = grouped.get(qid, [])
        if not group_records:
            raise ValueError(f"Query group '{qid}' not found in records.")
        group_counts.append(len(group_records))
        for rec in group_records:
            X_rows.append(rec.features.to_dense_vector())
            y_vals.append(rec.relevance_label)

    X = np.array(
        [[v if v is not None else np.nan for v in row] for row in X_rows],
        dtype=float,
    )
    y = np.asarray(y_vals, dtype=float)
    return X, y, group_counts


# ---------------------------------------------------------------------------
# Heuristic baseline scoring
# ---------------------------------------------------------------------------

def compute_heuristic_scores(records: List[RankingRecord]) -> List[float]:
    """
    Phase 6 heuristic baseline: feat_deterministic_base_score with
    scheme_id alphabetical tie-breaker for determinism.
    """
    # Collect scheme_ids per query group for tie-breaking
    query_schemes: Dict[str, List[str]] = {}
    for rec in records:
        query_schemes.setdefault(rec.query_id, []).append(rec.scheme_id)
    sorted_query_schemes: Dict[str, List[str]] = {
        qid: sorted(sids) for qid, sids in query_schemes.items()
    }

    scores = []
    for rec in records:
        base = rec.features.feat_deterministic_base_score
        q_sids = sorted_query_schemes[rec.query_id]
        n = max(1, len(q_sids))
        scheme_rank = q_sids.index(rec.scheme_id) if rec.scheme_id in q_sids else 0
        tie_break = (n - scheme_rank) * 1e-5
        scores.append(base + tie_break)
    return scores


# ---------------------------------------------------------------------------
# Main training and evaluation pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate_ltr(
    dataset_path: Path,
    output_dir: Path,
    val_ids: Optional[List[str]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full Phase 7H LTR training and evaluation pipeline."""
    if val_ids is None:
        val_ids = DEFAULT_VAL_IDS
    if xgb_params is None:
        xgb_params = XGBOOST_PARAMS

    # Step 1: Load and validate
    dataset = load_and_validate_dataset(dataset_path)

    # Step 2: Split by query_id
    train_records, val_records, train_qids, val_qids = split_by_query(dataset, val_ids)

    # Step 3: Prepare matrices
    X_train, y_train, train_group_counts = prepare_group_matrices(train_records, train_qids)
    X_val, y_val, val_group_counts = prepare_group_matrices(val_records, val_qids)

    logger.info(f"Train matrix: {X_train.shape} | Val matrix: {X_val.shape}")

    # Step 4: Train XGBRanker
    logger.info(f"Training XGBRanker with params: {xgb_params}")
    ranker = xgb.XGBRanker(**xgb_params)
    ranker.fit(X_train, y_train, group=train_group_counts)
    logger.info("XGBRanker training complete.")

    # Step 5: Predict
    train_preds = ranker.predict(X_train).tolist()
    val_preds = ranker.predict(X_val).tolist()

    # Step 6: XGBoost metrics
    xgb_train_metrics = compute_metrics_per_group(train_records, train_preds)
    xgb_val_metrics = compute_metrics_per_group(val_records, val_preds)

    # Step 7: Heuristic baseline
    heuristic_train_scores = compute_heuristic_scores(train_records)
    heuristic_val_scores = compute_heuristic_scores(val_records)
    heuristic_train_metrics = compute_metrics_per_group(train_records, heuristic_train_scores)
    heuristic_val_metrics = compute_metrics_per_group(val_records, heuristic_val_scores)

    # Step 8: Overfitting analysis
    xgb_train_agg = xgb_train_metrics["aggregate"]
    xgb_val_agg = xgb_val_metrics["aggregate"]
    train_val_gap = {
        "ndcg@3_gap": round(xgb_train_agg["ndcg@3"] - xgb_val_agg["ndcg@3"], 4),
        "ndcg@5_gap": round(xgb_train_agg["ndcg@5"] - xgb_val_agg["ndcg@5"], 4),
        "mrr_gap": round(xgb_train_agg["mrr"] - xgb_val_agg["mrr"], 4),
        "precision@3_gap": round(
            xgb_train_agg["precision@3"] - xgb_val_agg["precision@3"], 4
        ),
    }
    overfitting_flag = any(abs(v) > 0.15 for v in train_val_gap.values())
    overfitting_observation = (
        "WARNING: Large train/validation gap detected -- possible overfitting."
        if overfitting_flag
        else (
            "No strong overfitting detected (train/val gap within 0.15). "
            "Note: 16 query groups is a small sample -- results should be interpreted cautiously."
        )
    )

    # Step 9: XGBoost vs Heuristic comparison
    heuristic_val_agg = heuristic_val_metrics["aggregate"]
    xgb_vs_heuristic = {
        "ndcg@3_improvement": round(xgb_val_agg["ndcg@3"] - heuristic_val_agg["ndcg@3"], 4),
        "ndcg@5_improvement": round(xgb_val_agg["ndcg@5"] - heuristic_val_agg["ndcg@5"], 4),
        "mrr_improvement": round(xgb_val_agg["mrr"] - heuristic_val_agg["mrr"], 4),
        "precision@3_improvement": round(
            xgb_val_agg["precision@3"] - heuristic_val_agg["precision@3"], 4
        ),
    }

    # Step 10: Persist model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_ltr_model.json"
    ranker.save_model(str(model_path))
    logger.info(f"Model saved to: {model_path}")

    # Step 11: Build and save report
    report = {
        "TRAINING_STATUS": "SUCCESS",
        "MODEL_PATH": str(model_path),
        "DATASET_PATH": str(dataset_path),
        "TIMESTAMP": datetime.now(timezone.utc).isoformat(),
        "TRAIN_QUERY_GROUPS": train_qids,
        "VALIDATION_QUERY_GROUPS": val_qids,
        "N_TRAIN_RECORDS": len(train_records),
        "N_VAL_RECORDS": len(val_records),
        "FEATURE_DIM": X_train.shape[1],
        "XGBOOST_PARAMS": xgb_params,
        "TRAIN_METRICS": {
            "xgboost": xgb_train_metrics,
            "heuristic_baseline": heuristic_train_metrics,
        },
        "VALIDATION_METRICS": {
            "xgboost": xgb_val_metrics,
            "heuristic_baseline": heuristic_val_metrics,
        },
        "HEURISTIC_METRICS": heuristic_val_agg,
        "XGBOOST_VS_HEURISTIC": xgb_vs_heuristic,
        "OVERFITTING_ANALYSIS": {
            "train_val_gap": train_val_gap,
            "overfitting_flag": overfitting_flag,
            "overfitting_observation": overfitting_observation,
        },
        "CAVEATS": [
            "Only 16 query groups total -- results have high variance.",
            "All 138 records annotated as POTENTIALLY_ELIGIBLE (no ELIGIBLE records).",
            "Model must NOT override deterministic eligibility in live recommendation.",
            "XGBoost model is NOT integrated into live recommendation yet.",
        ],
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Training report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_report_summary(report: Dict[str, Any]) -> None:
    """Print concise human-readable training summary."""
    sep = "-" * 70
    print(sep)
    print("Phase 7H -- XGBoost LTR Training Summary")
    print(sep)
    print(f"TRAINING_STATUS        : {report['TRAINING_STATUS']}")
    print(f"MODEL_PATH             : {report['MODEL_PATH']}")
    print(f"TRAIN_QUERY_GROUPS     : {report['TRAIN_QUERY_GROUPS']}")
    print(f"VALIDATION_QUERY_GROUPS: {report['VALIDATION_QUERY_GROUPS']}")
    print(f"N_TRAIN_RECORDS        : {report['N_TRAIN_RECORDS']}")
    print(f"N_VAL_RECORDS          : {report['N_VAL_RECORDS']}")
    print(f"FEATURE_DIM            : {report['FEATURE_DIM']}")
    print()

    xgb_train = report["TRAIN_METRICS"]["xgboost"]["aggregate"]
    xgb_val = report["VALIDATION_METRICS"]["xgboost"]["aggregate"]
    heur_val = report["HEURISTIC_METRICS"]

    print("TRAIN_METRICS (XGBoost):")
    print(f"  NDCG@3={xgb_train['ndcg@3']}  NDCG@5={xgb_train['ndcg@5']}  MRR={xgb_train['mrr']}  P@3={xgb_train['precision@3']}")
    print()
    print("VALIDATION_METRICS:")
    print(f"  XGBoost  : NDCG@3={xgb_val['ndcg@3']}  NDCG@5={xgb_val['ndcg@5']}  MRR={xgb_val['mrr']}  P@3={xgb_val['precision@3']}")
    print(f"  Heuristic: NDCG@3={heur_val['ndcg@3']}  NDCG@5={heur_val['ndcg@5']}  MRR={heur_val['mrr']}  P@3={heur_val['precision@3']}")
    print()

    xvh = report["XGBOOST_VS_HEURISTIC"]
    print("XGBOOST_VS_HEURISTIC (val improvement over heuristic):")
    print(f"  dNDCG@3={xvh['ndcg@3_improvement']:+.4f}  dNDCG@5={xvh['ndcg@5_improvement']:+.4f}  dMRR={xvh['mrr_improvement']:+.4f}  dP@3={xvh['precision@3_improvement']:+.4f}")
    print()

    oa = report["OVERFITTING_ANALYSIS"]
    print(f"OVERFITTING_OBSERVATION: {oa['overfitting_observation']}")
    gap = oa["train_val_gap"]
    print(f"  Train-Val gap: NDCG@3={gap['ndcg@3_gap']:+.4f}  NDCG@5={gap['ndcg@5_gap']:+.4f}  MRR={gap['mrr_gap']:+.4f}  P@3={gap['precision@3_gap']:+.4f}")
    print()

    print("Per-query VALIDATION metrics (XGBoost):")
    for qid, m in report["VALIDATION_METRICS"]["xgboost"]["per_query"].items():
        print(f"  {qid}: NDCG@3={m['ndcg@3']}  NDCG@5={m['ndcg@5']}  MRR={m['mrr']}  P@3={m['precision@3']}  n={m['n_candidates']}")
    print()
    print("Per-query VALIDATION metrics (Heuristic):")
    for qid, m in report["VALIDATION_METRICS"]["heuristic_baseline"]["per_query"].items():
        print(f"  {qid}: NDCG@3={m['ndcg@3']}  NDCG@5={m['ndcg@5']}  MRR={m['mrr']}  P@3={m['precision@3']}  n={m['n_candidates']}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7H -- Train XGBoost Learning-to-Rank model"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ids", nargs="+", default=DEFAULT_VAL_IDS)
    args = parser.parse_args()

    report = train_and_evaluate_ltr(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        val_ids=args.val_ids,
    )
    _print_report_summary(report)


if __name__ == "__main__":
    main()
