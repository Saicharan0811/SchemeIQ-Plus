# -*- coding: utf-8 -*-
"""
src/recommendation/dataset_io.py — Dataset Serialization & XGBoost LTR Export (Phase 7B)

Handles loading, saving, and format conversion of Learning-to-Rank datasets:
1. JSON / JSONL serialization with metadata and privacy validation.
2. Conversion to XGBoost LTR structures: (X, y, qid, group_counts).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import RankingDataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANKING_DIR = PROJECT_ROOT / "data" / "ranking"


class RankingDatasetIO:
    """
    I/O and format conversion utility for Learning-to-Rank datasets.
    """

    @classmethod
    def save_json(cls, dataset: RankingDataset, output_path: Path) -> Path:
        """
        Validate and save dataset to JSON.
        """
        report = RankingDatasetValidator.validate_dataset(dataset)
        if not report.is_valid:
            raise ValueError(f"Dataset validation failed before saving: {report.errors}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved ranking dataset to {output_path} ({len(dataset.records)} records)")
        return output_path

    @classmethod
    def load_json(cls, input_path: Path) -> RankingDataset:
        """
        Load and validate dataset from JSON file.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Ranking dataset not found at: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        report = RankingDatasetValidator.validate_raw_dict(data)
        if not report.is_valid:
            raise ValueError(f"Loaded dataset failed validation: {report.errors}")

        return RankingDataset(**data)

    @classmethod
    def to_xgboost_format(
        cls,
        dataset: RankingDataset,
    ) -> Tuple[List[Dict[str, Any]], List[int], List[str], List[int]]:
        """
        Convert RankingDataset to standard XGBoost Learning-to-Rank input arrays.

        Returns:
            X: List of feature dictionaries (one per candidate record).
            y: List of relevance labels (integers 0..3).
            qid: List of query/group IDs matching each record row.
            group_counts: List of group sizes (records per query_id) in sequence for DMatrix group sizing.
        """
        report = RankingDatasetValidator.validate_dataset(dataset)
        if not report.is_valid:
            raise ValueError(f"Cannot export invalid dataset to XGBoost format: {report.errors}")

        # Ensure records are strictly grouped by query_id
        groups = dataset.get_query_groups()

        X: List[Dict[str, Any]] = []
        y: List[int] = []
        qid: List[str] = []
        group_counts: List[int] = []

        for group_id, records in groups.items():
            group_counts.append(len(records))
            for rec in records:
                X.append(rec.features.to_feature_dict())
                y.append(rec.relevance_label)
                qid.append(group_id)

        return X, y, qid, group_counts
