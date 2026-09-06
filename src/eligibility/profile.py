# -*- coding: utf-8 -*-
"""
src/eligibility/profile.py — User Profile Normalization and Privacy Validator
Ensures clean categorical parsing, non-persistence, and privacy data minimization.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.eligibility.schemas import PROHIBITED_SENSITIVE_KEYWORDS, UserProfile

logger = logging.getLogger(__name__)


def validate_profile_privacy(raw_dict: Dict[str, Any]) -> None:
    """
    Ensure no prohibited sensitive identifiers are passed into the profile.
    Raises ValueError if prohibited keys are detected.
    """
    found_prohibited = []
    for k in raw_dict.keys():
        k_lower = str(k).lower().strip()
        for p in PROHIBITED_SENSITIVE_KEYWORDS:
            if p == k_lower or p in k_lower:
                found_prohibited.append(k)

    if found_prohibited:
        raise ValueError(
            f"Privacy Violation: Prohibited sensitive fields detected: {found_prohibited}. "
            "SchemeIQ+ enforces strict data minimization. Do NOT submit Aadhaar, PAN, bank account numbers, "
            "passwords, or OTP details."
        )


def normalize_profile(profile: UserProfile) -> UserProfile:
    """
    Normalize string fields for consistent deterministic matching.
    """
    data = profile.model_dump()

    # Normalize state
    if data.get("state"):
        s = data["state"].strip()
        if s.lower() in ["ts", "telangana state", "tg"]:
            data["state"] = "Telangana"
        elif s.lower() in ["ap", "andhra pradesh", "andhra"]:
            data["state"] = "Andhra Pradesh"
        else:
            data["state"] = s.title()

    # Normalize gender
    if data.get("gender"):
        g = data["gender"].strip().lower()
        if g in ["f", "female", "woman", "women", "girl"]:
            data["gender"] = "female"
        elif g in ["m", "male", "man", "men", "boy"]:
            data["gender"] = "male"
        elif g in ["trans", "transgender", "tg"]:
            data["gender"] = "transgender"
        else:
            data["gender"] = g

    # Normalize residence_type
    if data.get("residence_type"):
        r = data["residence_type"].strip().lower()
        if r in ["urban", "city", "town", "municipality"]:
            data["residence_type"] = "urban"
        elif r in ["rural", "village", "gram", "panchayat"]:
            data["residence_type"] = "rural"

    # Normalize occupation
    if data.get("occupation"):
        occ = data["occupation"].strip().lower()
        data["occupation"] = occ
        if "farm" in occ or "agri" in occ:
            data["farmer_status"] = True if data.get("farmer_status") is None else data["farmer_status"]
        if "student" in occ:
            data["student_status"] = True if data.get("student_status") is None else data["student_status"]
        if "business" in occ or "entrepreneur" in occ or "shop" in occ:
            data["business_owner_status"] = True if data.get("business_owner_status") is None else data["business_owner_status"]

    return UserProfile(**data)


def create_safe_profile_summary(profile: UserProfile) -> Dict[str, Any]:
    """
    Create a compact dictionary summarizing non-null profile fields for explanation and reporting.
    """
    raw = profile.model_dump(exclude_none=True)
    return raw
