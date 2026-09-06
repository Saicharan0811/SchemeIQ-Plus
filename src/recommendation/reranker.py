# -*- coding: utf-8 -*-
"""
src/recommendation/reranker.py — Controlled XGBoost LTR Re-Ranking Layer (Phase 7I)

Integrates the trained XGBoost model strictly as an advisory re-ranking layer.

Core Invariants:
1. Deterministic eligibility remains the sole authority.
2. ONLY ELIGIBLE and POTENTIALLY_ELIGIBLE candidates reach feature extraction and XGBoost prediction.
3. NOT_ELIGIBLE, NOT_APPLICABLE, and other non-eligible candidates (protected_pool) are NEVER scored by XGBoost.
4. XGBoost cannot alter eligibility_status, create eligibility decisions, or modify candidate objects.
5. ML scores are stored in scoring_breakdown["xgboost_score"] without overwriting ev.score.
6. Deterministic tie-breaking by scheme_id ascending.
7. Safe fallback to Phase 6 deterministic heuristic ordering if model loading, feature extraction,
   prediction, or shape validation fails.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import xgboost as xgb

from src.eligibility.schemas import EligibilityStatus, SchemeEvaluation, UserProfile
from src.recommendation.feature_extractor import RankingFeatureExtractor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ranking" / "xgboost_ltr_model.json"

ELIGIBLE_CANDIDATE_STATUSES: Set[EligibilityStatus] = {
    EligibilityStatus.ELIGIBLE,
    EligibilityStatus.POTENTIALLY_ELIGIBLE,
}

EXPECTED_FEATURE_DIM: int = 32


class XGBoostReRanker:
    """
    Advisory re-ranker utilizing the trained XGBoost Learning-to-Rank model.
    Operates strictly within the pre-screened eligible candidate pool.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        lazy_load: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model: Optional[xgb.XGBRanker] = None
        self._load_error: Optional[str] = None

        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> xgb.XGBRanker:
        """
        Load or return cached XGBRanker model.
        Raises FileNotFoundError or RuntimeError if loading fails.
        """
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            msg = f"XGBoost model file not found at: {self.model_path}"
            self._load_error = msg
            raise FileNotFoundError(msg)

        try:
            ranker = xgb.XGBRanker()
            ranker.load_model(str(self.model_path))
            self._model = ranker
            self._load_error = None
            logger.info("Loaded XGBoost LTR model from %s", self.model_path)
            return self._model
        except Exception as e:
            self._load_error = str(e)
            raise RuntimeError(f"Failed to load XGBoost model from {self.model_path}: {e}") from e

    def is_available(self) -> bool:
        """Return True if model file exists and is successfully loaded/loadable."""
        try:
            self._ensure_model_loaded()
            return True
        except Exception:
            return False

    def rerank(
        self,
        profile: UserProfile,
        candidates: List[SchemeEvaluation],
    ) -> List[SchemeEvaluation]:
        """
        Advisory re-ranking of evaluated scheme candidates.

        Args:
            profile: User profile used for feature extraction.
            candidates: Full list of SchemeEvaluation objects (typically all 14 evaluated schemes).

        Returns:
            List of SchemeEvaluation objects with eligible candidates reordered by descending
            XGBoost score (tie-break by scheme_id), and protected candidates preserved in
            their original deterministic order.
        """
        if not candidates:
            return []

        # -------------------------------------------------------------------
        # Step 1: Partition into eligible_pool and protected_pool
        # -------------------------------------------------------------------
        eligible_pool: List[SchemeEvaluation] = []
        protected_pool: List[SchemeEvaluation] = []

        for c in candidates:
            if c.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES:
                eligible_pool.append(c)
            else:
                protected_pool.append(c)

        # Invariant checks: NOT_ELIGIBLE and NOT_APPLICABLE must NEVER enter eligible_pool
        for ep in eligible_pool:
            if ep.eligibility_status in (
                EligibilityStatus.NOT_ELIGIBLE,
                EligibilityStatus.NOT_APPLICABLE,
            ):
                raise AssertionError(
                    f"Invariant violated: {ep.scheme_id} ({ep.eligibility_status}) entered eligible_pool"
                )

        # If eligible pool has 0 or 1 candidate, no reordering is required
        if len(eligible_pool) <= 1:
            return eligible_pool + protected_pool

        # -------------------------------------------------------------------
        # Step 2: Extract features, predict XGBoost scores, and reorder
        # -------------------------------------------------------------------
        try:
            model = self._ensure_model_loaded()

            rows: List[List[Optional[float]]] = []
            for ev in eligible_pool:
                feat_vec = RankingFeatureExtractor.extract(profile, ev)
                dense_row = feat_vec.to_dense_vector()
                if len(dense_row) != EXPECTED_FEATURE_DIM:
                    raise ValueError(
                        f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM}, got {len(dense_row)}"
                    )
                rows.append([v if v is not None else np.nan for v in dense_row])

            X = np.array(rows, dtype=float)
            if X.ndim != 2 or X.shape[1] != EXPECTED_FEATURE_DIM:
                raise ValueError(
                    f"Invalid feature matrix shape {X.shape}, expected (*, {EXPECTED_FEATURE_DIM})"
                )

            preds = model.predict(X)
            if len(preds) != len(eligible_pool):
                raise RuntimeError(
                    f"Prediction length mismatch: got {len(preds)} predictions for {len(eligible_pool)} candidates"
                )

            # Record ML scores in scoring_breakdown WITHOUT overwriting deterministic ev.score
            for ev, pred in zip(eligible_pool, preds):
                ev.scoring_breakdown["xgboost_score"] = round(float(pred), 4)

            # Reorder only eligible_pool: descending XGBoost score, tie-break by scheme_id ascending
            eligible_pool.sort(
                key=lambda ev: (-ev.scoring_breakdown.get("xgboost_score", 0.0), ev.scheme_id)
            )

            # Total candidate count preservation verification
            result = eligible_pool + protected_pool
            if len(result) != len(candidates):
                raise RuntimeError(
                    f"Candidate count changed after rerank: before={len(candidates)}, after={len(result)}"
                )

            return result

        except Exception as e:
            logger.warning(
                "XGBoost reranking failed (%s). Falling back safely to Phase 6 deterministic heuristic order.",
                e,
            )
            # Safe Fallback: return candidates in original deterministic heuristic order
            return candidates
