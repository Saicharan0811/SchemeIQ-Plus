# -*- coding: utf-8 -*-
"""
src/eligibility/rule_engine.py — Deterministic Rule Evaluation Engine
Evaluates individual criteria against structured user profiles with complete evidence capture.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.eligibility.schemas import RuleResult, RuleStatus, UserProfile

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Evaluates rule definitions against a user profile using deterministic operators.
    Preserves rule-level evidence and provenance.
    """

    @staticmethod
    def evaluate_rule(rule_def: Dict[str, Any], profile: UserProfile) -> RuleResult:
        """
        Evaluate a single rule definition against the user profile.

        Returns RuleResult with status PASS, FAIL, UNKNOWN, or NOT_APPLICABLE.
        """
        rule_id = rule_def.get("rule_id", "UNKNOWN_RULE")
        field = rule_def.get("field", "")
        operator = rule_def.get("operator", "eq").lower().strip()
        expected = rule_def.get("expected_value")
        required = rule_def.get("required", True)
        rule_type = rule_def.get("rule_type", "hard_filter")
        description = rule_def.get("description", "")
        source_filename = rule_def.get("source_filename", "")
        official_url = rule_def.get("official_url", "")
        source_excerpt = rule_def.get("source_excerpt", "")
        confidence = rule_def.get("confidence", "HIGH")

        # Get actual value from profile
        user_value = getattr(profile, field, None)

        # Base result template
        res_kwargs = {
            "rule_id": rule_id,
            "field": field,
            "operator": operator,
            "user_value": user_value,
            "expected_value": expected,
            "required": required,
            "rule_type": rule_type,
            "source_filename": source_filename,
            "official_url": official_url,
            "source_excerpt": source_excerpt,
            "confidence": confidence,
        }

        # ---------------------------------------------------------
        # Case 1: Missing user information (UNKNOWN)
        # Note: In accordance with Step 4 & 5, missing profile info is UNKNOWN, never automatic FAIL.
        # Exception: For presence operators 'exists' / 'missing', None is evaluated directly.
        # ---------------------------------------------------------
        if user_value is None and operator not in ["exists", "missing"]:
            return RuleResult(
                status=RuleStatus.UNKNOWN,
                reason=f"User profile does not specify '{field}'. Criterion cannot be evaluated.",
                **res_kwargs,
            )

        # ---------------------------------------------------------
        # Case 2: Operator Evaluations
        # ---------------------------------------------------------
        try:
            # Numeric operators
            if operator == "eq":
                passed = user_value == expected
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' equals '{expected}'" if passed else f"Value '{user_value}' does not equal '{expected}'"

            elif operator == "neq":
                passed = user_value != expected
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is not equal to '{expected}'" if passed else f"Value '{user_value}' equals '{expected}'"

            elif operator == "gt":
                passed = float(user_value) > float(expected)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is strictly greater than '{expected}'" if passed else f"Value '{user_value}' is not greater than '{expected}'"

            elif operator == "gte":
                passed = float(user_value) >= float(expected)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' meets minimum requirement of '{expected}'" if passed else f"Value '{user_value}' is below required minimum of '{expected}'"

            elif operator == "lt":
                passed = float(user_value) < float(expected)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is strictly less than '{expected}'" if passed else f"Value '{user_value}' is not less than '{expected}'"

            elif operator == "lte":
                passed = float(user_value) <= float(expected)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is within upper limit of '{expected}'" if passed else f"Value '{user_value}' exceeds upper limit of '{expected}'"

            elif operator == "between":
                # expected is [min, max]
                min_v, max_v = expected[0], expected[1]
                passed = float(min_v) <= float(user_value) <= float(max_v)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' falls within range [{min_v}, {max_v}]" if passed else f"Value '{user_value}' falls outside range [{min_v}, {max_v}]"

            # Categorical / String operators
            elif operator == "equals_ignore_case":
                passed = str(user_value).strip().lower() == str(expected).strip().lower()
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' matches required '{expected}'" if passed else f"Value '{user_value}' does not match required '{expected}'"

            elif operator == "in":
                # expected is a list of acceptable values
                exp_list = [str(x).strip().lower() for x in (expected if isinstance(expected, list) else [expected])]
                passed = str(user_value).strip().lower() in exp_list
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is in allowed list {expected}" if passed else f"Value '{user_value}' is not in allowed list {expected}"

            elif operator == "not_in":
                exp_list = [str(x).strip().lower() for x in (expected if isinstance(expected, list) else [expected])]
                passed = str(user_value).strip().lower() not in exp_list
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' is correctly excluded from {expected}" if passed else f"Value '{user_value}' is in excluded list {expected}"

            elif operator == "contains_any":
                user_str = str(user_value).lower()
                exp_list = [str(x).lower() for x in (expected if isinstance(expected, list) else [expected])]
                passed = any(item in user_str for item in exp_list)
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Value '{user_value}' matches one of {expected}" if passed else f"Value '{user_value}' does not contain any of {expected}"

            # Boolean operators
            elif operator == "is_true":
                passed = bool(user_value) is True
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Condition '{field}' is satisfied (True)" if passed else f"Condition '{field}' is not satisfied (False)"

            elif operator == "is_false":
                passed = bool(user_value) is False
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Exclusion condition '{field}' is satisfied (False)" if passed else f"Exclusion condition '{field}' failed (True)"

            # Presence operators
            elif operator == "exists":
                passed = user_value is not None
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Field '{field}' is provided" if passed else f"Field '{field}' is missing"

            elif operator == "missing":
                passed = user_value is None
                status = RuleStatus.PASS if passed else RuleStatus.FAIL
                reason = f"Field '{field}' is absent as expected" if passed else f"Field '{field}' is present"

            else:
                status = RuleStatus.UNKNOWN
                reason = f"Unsupported operator '{operator}'."

        except (ValueError, TypeError) as e:
            status = RuleStatus.UNKNOWN
            reason = f"Type conversion error evaluating operator '{operator}' on value '{user_value}': {e}"

        return RuleResult(status=status, reason=reason, **res_kwargs)
