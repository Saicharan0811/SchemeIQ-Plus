# -*- coding: utf-8 -*-
"""
src/recommendation/schemas.py — XGBoost Learning-to-Rank Data Schemas (Phase 7B)

Defines data models for ranking records, feature vectors, query/session groupings,
and dataset validation containers for XGBoost Learning-to-Rank.
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.eligibility.schemas import UserProfile


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class RelevanceGrade(IntEnum):
    """
    Standard 4-tier relevance grading for Learning-to-Rank (LTR).
    0 = Irrelevant / Disqualified / Unhelpful
    1 = Broadly Relevant / General Information
    2 = High Interest / Targeted Match
    3 = Primary Choice / Applied / Explicit Positive Ground Truth
    """
    NOT_RELEVANT = 0
    BROADLY_RELEVANT = 1
    HIGH_INTEREST = 2
    PRIMARY_CHOICE = 3


OFFICIAL_SCHEME_IDS = {
    "CT001", "CT002", "CT003", "CT004", "CT005", "CT006",
    "TS001", "TS002", "TS003", "TS004", "TS005", "TS006", "TS007", "TS008",
}

SCHEME_ID_TO_INDEX: Dict[str, int] = {
    sid: idx for idx, sid in enumerate(sorted(OFFICIAL_SCHEME_IDS))
}


# ---------------------------------------------------------------------------
# Feature Vector Schema
# ---------------------------------------------------------------------------

class RankingFeatureVector(BaseModel):
    """
    Tabular numerical and encoded feature representation for a (user, scheme) pair.
    Maps exclusively to verified fields from UserProfile and SchemeEvaluation.
    Missing values remain None to allow XGBoost native missing-value handling.
    """
    # 1. Demographic & Continuous User Profile Features
    feat_user_age: Optional[float] = Field(default=None, description="Applicant age in years")
    feat_user_annual_income: Optional[float] = Field(default=None, description="Annual family income in INR")
    feat_user_land_acres: Optional[float] = Field(default=None, description="Total cultivable land in acres")
    feat_user_loan_requirement: Optional[float] = Field(default=None, description="Loan requirement in INR")
    feat_user_project_cost: Optional[float] = Field(default=None, description="Proposed project cost in INR")
    feat_user_monthly_pension: Optional[float] = Field(default=None, description="Current monthly pension in INR")

    # 2. Encoded Categorical Attributes (0 = unknown/unspecified)
    feat_user_gender: int = Field(default=0, description="0=unknown, 1=female, 2=male, 3=transgender, 4=other")
    feat_user_residence_type: int = Field(default=0, description="0=unknown, 1=urban, 2=rural")
    feat_user_category: int = Field(default=0, description="0=unknown, 1=SC, 2=ST, 3=BC, 4=EBC, 5=Minority, 6=General")
    feat_user_state_is_telangana: int = Field(default=0, description="1 if state is Telangana, 0 otherwise")

    # 3. Encoded Binary Profile Flags (-1 = unknown, 0 = False, 1 = True)
    feat_user_is_farmer: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_has_land_ownership: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_is_student: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_is_business_owner: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_is_pensioner: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_owns_pucca_house: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_has_bpl_card: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_pays_income_tax: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_is_pregnant_or_newborn: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")
    feat_user_has_savings_bank_account: int = Field(default=-1, description="-1=unknown, 0=False, 1=True")

    # 4. Scheme Static Features
    feat_scheme_id_index: int = Field(default=0, description="Integer index (0..13) of official scheme")
    feat_scheme_scope: int = Field(default=1, description="1=State, 2=Central")

    # 5. Deterministic Rule & Evaluation Signals
    feat_rule_pass_ratio: float = Field(default=0.0, description="Passed rules / total evaluated rules")
    feat_rules_matched_count: int = Field(default=0, description="Count of matched rules")
    feat_rules_failed_count: int = Field(default=0, description="Count of failed rules")
    feat_rules_unknown_count: int = Field(default=0, description="Count of unknown rules")
    feat_missing_fields_count: int = Field(default=0, description="Count of unprovided profile fields")
    feat_deterministic_base_score: float = Field(default=0.0, description="Base score from SchemeRecommender")
    feat_eligibility_tier: int = Field(
        default=0,
        description="0=NOT_ELIGIBLE, 1=NOT_APPLICABLE, 2=INSUFFICIENT_INFO, 3=POTENTIALLY_ELIGIBLE, 4=ELIGIBLE",
    )

    # 6. Alignment & Preference Features
    feat_category_pref_match: int = Field(default=0, description="1 if user category preference matches scheme")
    feat_scope_pref_match: int = Field(default=0, description="1 if user scope preference matches scheme")
    feat_domain_specificity_count: int = Field(default=0, description="Number of matched domain criteria")

    def to_feature_dict(self) -> Dict[str, Any]:
        """Return feature dictionary for model tabular training matrices."""
        return self.model_dump()

    def to_dense_vector(self) -> List[Optional[float]]:
        """Return values as an ordered list of floats or None for XGBoost DMatrix."""
        vals = []
        for k, v in self.model_dump().items():
            vals.append(float(v) if v is not None else None)
        return vals


# ---------------------------------------------------------------------------
# Ranking Training Record
# ---------------------------------------------------------------------------

class RankingRecord(BaseModel):
    """
    Atomic unit of observation for Learning-to-Rank: (user/session, scheme, features, label, group_id).
    """
    record_id: str = Field(description="Unique record identifier, e.g. 'rec_q001_TS001'")
    query_id: str = Field(description="Group/Query ID grouping candidates compared against the same user session")
    session_id: Optional[str] = Field(default=None, description="Optional anonymous session identifier")
    scheme_id: str = Field(description="Official scheme identifier (e.g. 'TS001')")
    scheme_name: str = Field(description="Official scheme name")
    features: RankingFeatureVector = Field(description="Extracted feature vector for this (user, scheme) pair")
    relevance_label: int = Field(
        ge=0,
        le=3,
        description="Relevance grade: 0=Not Relevant, 1=Broadly Relevant, 2=High Interest, 3=Primary Choice",
    )
    eligibility_status: str = Field(description="Deterministic status from SchemeEligibilityEvaluator")
    provenance_source: str = Field(
        default="expert_curation",
        description="Source of the label: 'expert_curation', 'user_interaction', 'synthetic_validation_example'",
    )
    notes: Optional[str] = Field(default=None, description="Optional contextual rationale for relevance grade")

    @field_validator("scheme_id")
    @classmethod
    def validate_scheme_id(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in OFFICIAL_SCHEME_IDS:
            raise ValueError(f"Invalid scheme_id '{v}'. Must be one of {sorted(OFFICIAL_SCHEME_IDS)}")
        return clean

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query_id cannot be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Ranking Dataset Container & Metadata
# ---------------------------------------------------------------------------

class DatasetPurpose(str, Enum):
    SCHEMA_DEMONSTRATION = "SCHEMA_DEMONSTRATION_ONLY_NOT_FOR_TRAINING"
    EXPERT_CURATED_BENCHMARK = "EXPERT_CURATED_BENCHMARK"
    ANONYMIZED_PILOT_LOGS = "ANONYMIZED_PILOT_LOGS"
    PRODUCTION_TRAINING = "PRODUCTION_TRAINING"


class RankingDatasetMetadata(BaseModel):
    """Metadata describing a ranking dataset package."""
    dataset_name: str = Field(default="SchemeIQ_Plus_LTR_Dataset")
    schema_version: str = Field(default="1.0.0")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    purpose: DatasetPurpose = Field(
        default=DatasetPurpose.SCHEMA_DEMONSTRATION,
        description="Explicit flag ensuring demonstration data is never accidentally used for real training",
    )
    total_records: int = 0
    total_groups: int = 0
    label_distribution: Dict[str, int] = Field(default_factory=dict)
    author_or_curator: str = "SchemeIQ+ Project"
    disclaimer: str = (
        "Notice: XGBoost LTR datasets contain feature representations for personalized scheme ranking. "
        "They do NOT alter or replace the deterministic statutory eligibility rules."
    )


class RankingDataset(BaseModel):
    """
    Top-level container for a complete Learning-to-Rank dataset.
    """
    metadata: RankingDatasetMetadata
    records: List[RankingRecord] = Field(default_factory=list)

    def get_query_groups(self) -> Dict[str, List[RankingRecord]]:
        """Group records by query_id in preserved sequence."""
        groups: Dict[str, List[RankingRecord]] = {}
        for r in self.records:
            groups.setdefault(r.query_id, []).append(r)
        return groups

    def get_group_counts(self) -> List[int]:
        """Return list of group sizes in query_id sequence, as required by XGBoost .group / group parameter."""
        groups = self.get_query_groups()
        return [len(records) for records in groups.values()]


# ---------------------------------------------------------------------------
# Citizen Archetype Schema (Phase 7E)
# ---------------------------------------------------------------------------

class CitizenArchetype(BaseModel):
    """
    Standardized citizen archetype definition for expert relevance annotation.
    Represents a realistic, non-PII demographic and socio-economic persona.
    """
    archetype_id: str = Field(description="Unique archetype identifier, e.g. 'arch_farmer_001'")
    archetype_name: str = Field(description="Descriptive archetype title, e.g. 'Telangana Landholding Small Farmer'")
    narrative: Optional[str] = Field(default=None, description="Contextual background of the citizen's situation")
    profile: UserProfile = Field(description="Validated user profile attributes conforming to Phase 6 schema")

    @field_validator("archetype_id")
    @classmethod
    def validate_archetype_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("archetype_id cannot be empty")
        return v.strip()

