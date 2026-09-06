# -*- coding: utf-8 -*-
"""
src/recommendation/dataset_validator.py — XGBoost LTR Dataset Validator (Phase 7B)

Validates ranking datasets against strict data-quality and privacy rules:
1. Prohibited sensitive personal identifiers (Aadhaar, PAN, bank accounts, etc.)
2. Missing or malformed required fields
3. Invalid relevance labels (must be integers 0..3)
4. Invalid, empty, or missing query/group IDs
5. Duplicate record IDs or duplicate (query_id, scheme_id) candidate pairs within a group
6. Scheme ID validity against the 14 official corpus schemes
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field

from src.eligibility.schemas import PROHIBITED_SENSITIVE_KEYWORDS
from src.recommendation.schemas import (
    OFFICIAL_SCHEME_IDS,
    RankingDataset,
    RankingRecord,
)

logger = logging.getLogger(__name__)


class DatasetValidationReport(BaseModel):
    """
    Detailed audit report produced by RankingDatasetValidator.
    """
    is_valid: bool = True
    total_records: int = 0
    total_query_groups: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary_stats: Dict[str, Any] = Field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class RankingDatasetValidator:
    """
    Validation engine for Learning-to-Rank training datasets.
    """

    @classmethod
    def validate_raw_dict(cls, data: Dict[str, Any]) -> DatasetValidationReport:
        """
        Validate a raw dictionary representation before parsing into RankingDataset.
        Checks for top-level privacy violations, metadata, and record structures.
        """
        report = DatasetValidationReport()

        # 1. Check raw dictionary for prohibited keywords
        cls._scan_dict_for_privacy(data, report)

        # 2. Check metadata
        meta = data.get("metadata")
        if not meta or not isinstance(meta, dict):
            report.add_error("Missing or malformed 'metadata' block.")

        # 3. Check records
        records_raw = data.get("records")
        if not isinstance(records_raw, list):
            report.add_error("Field 'records' must be a list of record objects.")
            return report

        report.total_records = len(records_raw)
        if len(records_raw) == 0:
            report.add_warning("Dataset contains 0 records.")
            return report

        # Parse and validate individual records
        cls._validate_records_list(records_raw, report)
        return report

    @classmethod
    def validate_dataset(cls, dataset: RankingDataset) -> DatasetValidationReport:
        """
        Validate an instantiated RankingDataset model.
        """
        report = DatasetValidationReport()
        report.total_records = len(dataset.records)

        # Check records
        dict_records = [r.model_dump() for r in dataset.records]
        cls._validate_records_list(dict_records, report)
        return report

    @classmethod
    def _scan_dict_for_privacy(cls, data: Any, report: DatasetValidationReport, path: str = "") -> None:
        """Recursively scan keys and string values for prohibited sensitive terms."""
        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = str(k).lower().strip()
                for kw in PROHIBITED_SENSITIVE_KEYWORDS:
                    if kw in k_lower:
                        report.add_error(f"Privacy Violation: Prohibited sensitive key '{k}' found at path '{path}'.")
                cls._scan_dict_for_privacy(v, report, f"{path}.{k}" if path else k)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                cls._scan_dict_for_privacy(item, report, f"{path}[{i}]")
        elif isinstance(data, str):
            # Check string contents for explicit raw PAN/Aadhaar patterns
            d_lower = data.lower()
            if "aadhaar" in d_lower and any(c.isdigit() for c in data):
                report.add_error(f"Privacy Violation: Potential sensitive Aadhaar number string at '{path}'.")

    @classmethod
    def _validate_records_list(
        cls,
        records: List[Dict[str, Any]],
        report: DatasetValidationReport,
    ) -> None:
        """Validate list of record dicts for all Phase 7B requirements."""
        seen_record_ids: Set[str] = set()
        seen_query_scheme_pairs: Set[Tuple[str, str]] = set()
        query_groups: Dict[str, int] = {}
        label_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        for idx, rec in enumerate(records):
            rec_id = rec.get("record_id")
            qid = rec.get("query_id")
            sid = rec.get("scheme_id")
            label = rec.get("relevance_label")
            features = rec.get("features")

            # Missing required fields
            if not rec_id:
                report.add_error(f"Record at index {idx} is missing 'record_id'.")
            elif rec_id in seen_record_ids:
                report.add_error(f"Duplicate 'record_id' '{rec_id}' at index {idx}.")
            else:
                seen_record_ids.add(str(rec_id))

            # Query/Group ID check
            if not qid or not str(qid).strip():
                report.add_error(f"Record '{rec_id}' (index {idx}) has missing or empty 'query_id'.")
            else:
                qid_str = str(qid).strip()
                query_groups[qid_str] = query_groups.get(qid_str, 0) + 1

            # Scheme ID check
            if not sid:
                report.add_error(f"Record '{rec_id}' (index {idx}) is missing 'scheme_id'.")
            elif str(sid).strip().upper() not in OFFICIAL_SCHEME_IDS:
                report.add_error(
                    f"Record '{rec_id}' has invalid scheme_id '{sid}'. Must be one of {sorted(OFFICIAL_SCHEME_IDS)}."
                )

            # Duplicate (query_id, scheme_id) check
            if qid and sid:
                pair = (str(qid).strip(), str(sid).strip().upper())
                if pair in seen_query_scheme_pairs:
                    report.add_error(
                        f"Duplicate candidate pair: scheme '{pair[1]}' is assigned multiple labels for query_id '{pair[0]}'."
                    )
                else:
                    seen_query_scheme_pairs.add(pair)

            # Relevance label check
            if label is None or not isinstance(label, int) or label < 0 or label > 3:
                report.add_error(
                    f"Record '{rec_id}' has invalid relevance_label '{label}'. Must be an integer in [0, 1, 2, 3]."
                )
            else:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Feature vector check
            if not features or not isinstance(features, dict):
                report.add_error(f"Record '{rec_id}' is missing or has malformed 'features' dictionary.")

        report.total_query_groups = len(query_groups)

        # Check for single-record groups (XGBoost ranking algorithms need pairwise comparisons)
        for qid, count in query_groups.items():
            if count < 2:
                report.add_warning(
                    f"Query group '{qid}' contains only {count} candidate. "
                    "Learning-to-rank models typically benefit from ≥ 2 candidates per group for pairwise ranking."
                )

        report.summary_stats = {
            "total_records": len(records),
            "total_groups": len(query_groups),
            "label_distribution": label_counts,
            "average_candidates_per_group": round(len(records) / max(1, len(query_groups)), 2),
        }
