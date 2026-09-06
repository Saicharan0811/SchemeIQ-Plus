# -*- coding: utf-8 -*-
"""
src/recommendation/ — SchemeIQ+ Learning-to-Rank Data & Recommendation Module (Phase 7B/7E)
"""

from src.recommendation.schemas import (
    RelevanceGrade,
    OFFICIAL_SCHEME_IDS,
    SCHEME_ID_TO_INDEX,
    RankingFeatureVector,
    RankingRecord,
    DatasetPurpose,
    RankingDatasetMetadata,
    RankingDataset,
    CitizenArchetype,
)
from src.recommendation.feature_extractor import (
    RankingFeatureExtractor,
)
from src.recommendation.dataset_validator import (
    DatasetValidationReport,
    RankingDatasetValidator,
)
from src.recommendation.dataset_io import (
    RankingDatasetIO,
)
from src.recommendation.annotation_tool import (
    VALID_EXPERT_RELEVANCE_GRADES,
    ELIGIBLE_CANDIDATE_STATUSES,
    validate_annotator_id,
    load_archetypes_from_file,
    ExpertAnnotationSession,
)
from src.recommendation.web_annotation import (
    create_app as create_web_annotation_app,
    launch_web_annotation,
)
from src.recommendation.train_ltr import (
    train_and_evaluate_ltr,
    split_by_query,
    prepare_group_matrices,
    compute_metrics_per_group,
)
from src.recommendation.reranker import (
    XGBoostReRanker,
)

__all__ = [
    "RelevanceGrade",
    "OFFICIAL_SCHEME_IDS",
    "SCHEME_ID_TO_INDEX",
    "RankingFeatureVector",
    "RankingRecord",
    "DatasetPurpose",
    "RankingDatasetMetadata",
    "RankingDataset",
    "CitizenArchetype",
    "RankingFeatureExtractor",
    "DatasetValidationReport",
    "RankingDatasetValidator",
    "RankingDatasetIO",
    "VALID_EXPERT_RELEVANCE_GRADES",
    "ELIGIBLE_CANDIDATE_STATUSES",
    "validate_annotator_id",
    "load_archetypes_from_file",
    "ExpertAnnotationSession",
    "create_web_annotation_app",
    "launch_web_annotation",
    "train_and_evaluate_ltr",
    "split_by_query",
    "prepare_group_matrices",
    "compute_metrics_per_group",
    "XGBoostReRanker",
]
