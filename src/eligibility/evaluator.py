# -*- coding: utf-8 -*-
"""
src/eligibility/evaluator.py — Scheme Eligibility Evaluator
Evaluates all structured rules for each of the 14 official schemes against a user profile.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eligibility.profile import normalize_profile
from src.eligibility.rule_engine import RuleEngine
from src.eligibility.schemas import (
    EligibilityStatus,
    RuleResult,
    RuleStatus,
    SchemeEvaluation,
    UserProfile,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "data" / "eligibility" / "scheme_rules.json"


class SchemeEligibilityEvaluator:
    """
    Loads derived scheme rules and evaluates scheme-level eligibility.
    """

    def __init__(self, rules_path: Path = DEFAULT_RULES_PATH):
        self.rules_path = rules_path
        self._rules_data = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules database not found at: {self.rules_path}")
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def schemes(self) -> List[Dict[str, Any]]:
        return self._rules_data.get("schemes", [])

    def get_scheme_by_id(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        for s in self.schemes:
            if s.get("scheme_id") == scheme_id:
                return s
        return None

    def evaluate_scheme(self, scheme_def: Dict[str, Any], raw_profile: UserProfile) -> SchemeEvaluation:
        """
        Evaluate a single scheme against a user profile.
        """
        profile = normalize_profile(raw_profile)
        scheme_id = scheme_def.get("scheme_id", "UNKNOWN")
        scheme_name = scheme_def.get("scheme_name", "Unknown Scheme")
        scheme_scope = scheme_def.get("scheme_scope", "Central")
        category = scheme_def.get("category", "General")
        official_url = scheme_def.get("official_url", "")
        source_authority = scheme_def.get("source_authority", "")
        rules = scheme_def.get("rules", [])
        unstructured = scheme_def.get("unstructured_criteria", [])

        matched_rules: List[RuleResult] = []
        failed_rules: List[RuleResult] = []
        unknown_rules: List[RuleResult] = []
        missing_fields: List[str] = []

        # Evaluate each rule
        for r_def in rules:
            res = RuleEngine.evaluate_rule(r_def, profile)
            if res.status == RuleStatus.PASS:
                matched_rules.append(res)
            elif res.status == RuleStatus.FAIL:
                failed_rules.append(res)
            elif res.status == RuleStatus.UNKNOWN:
                unknown_rules.append(res)
                if res.field not in missing_fields:
                    missing_fields.append(res.field)

        # ---------------------------------------------------------
        # Determine Eligibility Status
        # ---------------------------------------------------------
        req_failed = [r for r in failed_rules if r.required]
        req_unknown = [r for r in unknown_rules if r.required]
        req_passed = [r for r in matched_rules if r.required]
        total_req = len([r for r in rules if r.get("required", True)])

        # Check for clear state or gender hard filter failures -> NOT_APPLICABLE vs NOT_ELIGIBLE
        state_fail = any(r.field == "state" for r in req_failed)
        gender_fail = any(r.field == "gender" for r in req_failed)

        if state_fail and scheme_scope == "State":
            eligibility_status = EligibilityStatus.NOT_APPLICABLE
            summary = f"Scheme is restricted to residents of Telangana. User resides in '{profile.state}'."
        elif gender_fail:
            eligibility_status = EligibilityStatus.NOT_APPLICABLE
            summary = f"Scheme is gender-specific. User specified gender '{profile.gender}'."
        elif len(req_failed) > 0:
            eligibility_status = EligibilityStatus.NOT_ELIGIBLE
            failed_reasons = "; ".join([r.reason for r in req_failed])
            summary = f"Failed required criteria: {failed_reasons}."
        elif len(matched_rules) == 0 and len(unknown_rules) == len(rules):
            # Profile has no matching attributes at all
            eligibility_status = EligibilityStatus.INSUFFICIENT_INFORMATION
            summary = "Insufficient profile information provided to assess this scheme."
        elif len(req_unknown) > 0 or len(unstructured) > 0 or len(req_passed) < total_req:
            # Baseline is satisfied, but some criteria are unverified
            eligibility_status = EligibilityStatus.POTENTIALLY_ELIGIBLE
            passed_count = len(matched_rules)
            summary = f"Satisfies {passed_count} verified criteria. Additional verification required for: {', '.join(missing_fields) if missing_fields else 'official departmental records'}."
        else:
            # All required rules passed, 0 failed, 0 required unknown
            eligibility_status = EligibilityStatus.ELIGIBLE
            summary = f"All {len(req_passed)} structured required criteria are satisfied based on provided profile."

        return SchemeEvaluation(
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            scheme_scope=scheme_scope,
            category=category,
            official_url=official_url,
            source_authority=source_authority,
            eligibility_status=eligibility_status,
            confidence="HIGH" if len(matched_rules) > 0 else "MEDIUM",
            matched_rules=matched_rules,
            failed_rules=failed_rules,
            unknown_rules=unknown_rules,
            missing_information=missing_fields,
            unstructured_criteria=unstructured,
            explanation_summary=summary,
        )

    def evaluate_all(self, profile: UserProfile) -> List[SchemeEvaluation]:
        """
        Evaluate all 14 official schemes against the user profile.
        """
        results = []
        for scheme_def in self.schemes:
            eval_res = self.evaluate_scheme(scheme_def, profile)
            results.append(eval_res)
        return results
