# -*- coding: utf-8 -*-
"""
src/eligibility/schemas.py — Pydantic Schemas for Personalized Eligibility Engine
Defines data structures for user profiles, rule evaluation results, scheme assessments, and recommendations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Privacy Notice
# ---------------------------------------------------------------------------
DATA_MINIMIZATION_NOTICE = (
    "Privacy Notice: SchemeIQ+ adheres to strict data minimization. "
    "Only provide attributes necessary for eligibility screening. "
    "Do NOT submit Aadhaar numbers, PAN numbers, bank account numbers, passwords, OTPs, "
    "or exact street addresses. All evaluation is performed in-memory without persistent profile storage."
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class EligibilityStatus(str, Enum):
    """
    Deterministic eligibility assessment statuses.
    """
    ELIGIBLE = "ELIGIBLE"
    POTENTIALLY_ELIGIBLE = "POTENTIALLY_ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleStatus(str, Enum):
    """
    Status of an individual rule evaluation.
    """
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleType(str, Enum):
    """
    Type of rule constraint.
    """
    HARD_FILTER = "hard_filter"
    PREFERENCE = "preference"
    DOMAIN_MATCH = "domain_match"


# ---------------------------------------------------------------------------
# User Profile Schema (Step 3 & Step 9 Data Minimization)
# ---------------------------------------------------------------------------
PROHIBITED_SENSITIVE_KEYWORDS = {
    "aadhaar_number", "aadhaar_no", "pan_number", "pan_no",
    "bank_account_number", "bank_acc_no", "account_number", "account_no",
    "password", "otp", "pin", "cvv", "credit_card_number"
}


class UserProfile(BaseModel):
    """
    Structured user profile for deterministic eligibility evaluation.
    Supports partial information — no optional field is strictly required.
    """
    # Baseline Demographics
    age: Optional[int] = Field(default=None, description="Age of the applicant in years (0-120)")
    state: Optional[str] = Field(default="Telangana", description="State of residence (e.g. Telangana, Andhra Pradesh, etc.)")
    district: Optional[str] = Field(default=None, description="District of residence")
    residence_type: Optional[str] = Field(default=None, description="Residence location type: 'urban' or 'rural'")
    gender: Optional[str] = Field(default=None, description="Gender: 'female', 'male', 'transgender', 'other'")
    category: Optional[str] = Field(default=None, description="Social category: 'SC', 'ST', 'BC', 'EBC', 'Minority', 'General', 'OBC'")
    marital_status: Optional[str] = Field(default=None, description="Marital status: 'single', 'married', 'widowed', 'divorced'")
    annual_income: Optional[float] = Field(default=None, description="Gross annual family income in INR (e.g. 150000)")

    # Occupation & Sector Flags
    occupation: Optional[str] = Field(default=None, description="Primary occupation (e.g. 'farmer', 'student', 'artisan', 'business_owner', 'unemployed')")
    employment_status: Optional[str] = Field(default=None, description="Employment status: 'employed', 'self_employed', 'unemployed', 'student', 'retired'")
    farmer_status: Optional[bool] = Field(default=None, description="True if the applicant/family is actively engaged in farming")
    land_ownership_status: Optional[bool] = Field(default=None, description="True if the applicant/family owns agricultural land in official land records")
    land_acres: Optional[float] = Field(default=None, description="Total cultivable landholding in acres")

    # Education & Student
    student_status: Optional[bool] = Field(default=None, description="True if applicant is currently an enrolled student")
    education_level: Optional[str] = Field(default=None, description="Highest education completed: 'below_8th', '8th_pass', '10th_pass', '12th_pass', 'diploma', 'graduate', 'post_graduate'")

    # Vulnerability & Social Security
    disability_status: Optional[bool] = Field(default=None, description="True if applicant has a certified disability (e.g. SADAREM)")
    pensioner_status: Optional[bool] = Field(default=None, description="True if applicant is currently drawing or seeking a pension")
    pensioner_category: Optional[str] = Field(default=None, description="Specific category: 'old_age', 'widow', 'disabled', 'weaver', 'toddy_tapper', 'single_women', 'filaria'")
    monthly_pension: Optional[float] = Field(default=None, description="Current monthly pension in INR (if applicable)")

    # Business & Micro-Enterprise
    business_owner_status: Optional[bool] = Field(default=None, description="True if applicant owns or proposes to start a micro/small business")
    business_type: Optional[str] = Field(default=None, description="Enterprise type: 'manufacturing', 'service', 'trading', 'agro_processing'")
    is_non_farm_enterprise: Optional[bool] = Field(default=None, description="True if enterprise is in non-farm sector")
    project_cost: Optional[float] = Field(default=None, description="Estimated total project cost in INR for business/subsidy")
    loan_requirement: Optional[float] = Field(default=None, description="Total loan amount required in INR")

    # Housing & Assets
    owns_pucca_house: Optional[bool] = Field(default=None, description="True if applicant/family already owns a pucca house anywhere in India")
    is_bpl_or_white_ration_card: Optional[bool] = Field(default=None, description="True if family holds a Food Security Card / BPL / White Ration Card")
    is_pregnant_or_newborn: Optional[bool] = Field(default=None, description="True if beneficiary is a pregnant woman or mother with newborn")
    has_savings_bank_account: Optional[bool] = Field(default=None, description="True if applicant holds an active savings bank account")
    pays_income_tax: Optional[bool] = Field(default=None, description="True if applicant or family member is an active income tax payer")
    is_institutional_landholder: Optional[bool] = Field(default=None, description="True if land is held by an institution/trust rather than individual family")

    # Preferences & Filter Overrides
    scope_preference: Optional[str] = Field(default=None, description="Preference: 'State', 'Central', or 'All'")
    category_preference: Optional[str] = Field(default=None, description="Preferred category: 'Agriculture', 'Healthcare', 'Women', 'Education', 'Social Security', 'Housing', 'Entrepreneurship'")

    @field_validator("age")
    @classmethod
    def validate_age_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 120):
            raise ValueError("Age must be between 0 and 120 years.")
        return v

    @field_validator("annual_income", "project_cost", "loan_requirement", "monthly_pension", "land_acres")
    @classmethod
    def validate_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Financial/numerical attributes cannot be negative.")
        return v


# ---------------------------------------------------------------------------
# Rule-Level Result Schema (Step 4)
# ---------------------------------------------------------------------------
class RuleResult(BaseModel):
    """
    Structured outcome of a single rule evaluation.
    Preserves exact evidence and source provenance.
    """
    rule_id: str
    field: str
    operator: str
    status: RuleStatus
    user_value: Any = None
    expected_value: Any = None
    required: bool = True
    rule_type: str = "hard_filter"
    reason: str = ""
    source_filename: str = ""
    official_url: str = ""
    source_excerpt: str = ""
    confidence: str = "HIGH"


# ---------------------------------------------------------------------------
# Scheme-Level Evaluation Schema (Step 5)
# ---------------------------------------------------------------------------
class SchemeEvaluation(BaseModel):
    """
    Complete evaluation outcome for a single scheme against a user profile.
    """
    scheme_id: str
    scheme_name: str
    scheme_scope: str
    category: str
    official_url: str
    source_authority: str = ""
    eligibility_status: EligibilityStatus
    confidence: str = "HIGH"
    score: float = 0.0
    scoring_breakdown: Dict[str, float] = Field(default_factory=dict)
    matched_rules: List[RuleResult] = Field(default_factory=list)
    failed_rules: List[RuleResult] = Field(default_factory=list)
    unknown_rules: List[RuleResult] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    unstructured_criteria: List[Dict[str, str]] = Field(default_factory=list)
    explanation_summary: str = ""


# ---------------------------------------------------------------------------
# Recommendation Result Schema (Step 6)
# ---------------------------------------------------------------------------
class RecommendationResult(BaseModel):
    """
    Full recommendation response containing ranked schemes and explainability metadata.
    """
    user_profile_summary: Dict[str, Any]
    total_schemes_evaluated: int
    recommendations: List[SchemeEvaluation]
    status_counts: Dict[str, int]
    evaluation_timestamp: str
    disclaimer: str = (
        "Preliminary Assessment Disclaimer: This eligibility assessment is strictly derived from "
        "the SchemeIQ+ verified official source dataset. Final eligibility, entitlement sanctioning, "
        "and benefit disbursement are exclusively determined by the competent government authority "
        "following official scrutiny of original documents."
    )
