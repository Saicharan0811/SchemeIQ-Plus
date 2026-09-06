# -*- coding: utf-8 -*-
"""
src/eligibility/explanation.py — Explainability & Evidence Formatting Layer
Formats transparent, human-readable explanations detailing why schemes were recommended,
what criteria passed/failed, what is missing, and attaching official source provenance.
"""
from __future__ import annotations

from typing import List
from src.eligibility.schemas import EligibilityStatus, SchemeEvaluation, RecommendationResult


class ExplanationFormatter:
    """
    Constructs explainable natural-language and bulleted summaries of eligibility assessments.
    """

    @staticmethod
    def format_scheme_explanation(eval_res: SchemeEvaluation) -> str:
        """
        Format a detailed explainability block for a single scheme evaluation.
        """
        lines: List[str] = []
        lines.append(f"Scheme: [{eval_res.scheme_id}] {eval_res.scheme_name} ({eval_res.scheme_scope})")
        lines.append(f"Category: {eval_res.category}")
        lines.append(f"Eligibility Assessment: {eval_res.eligibility_status.value}")
        lines.append(f"Recommendation Score: {eval_res.score:.1f} / 150")
        lines.append(f"Official Portal: {eval_res.official_url}")
        lines.append("")

        # 1. Why recommended / Status Summary
        lines.append("Assessment Summary:")
        lines.append(f"  • {eval_res.explanation_summary}")
        lines.append("")

        # 2. Matched criteria
        if eval_res.matched_rules:
            lines.append("✓ Satisfied Official Criteria:")
            for r in eval_res.matched_rules:
                req_label = "(Required)" if r.required else "(Optional/Preference)"
                lines.append(f"  • [{r.field}] {r.reason} {req_label}")
                if r.official_url:
                    lines.append(f"    Source: {r.source_filename} ({r.official_url})")
            lines.append("")

        # 3. Failed criteria
        if eval_res.failed_rules:
            lines.append("✗ Unmet / Disqualifying Criteria:")
            for r in eval_res.failed_rules:
                req_label = "(Mandatory Condition)" if r.required else "(Preference)"
                lines.append(f"  • [{r.field}] {r.reason} {req_label}")
                if r.official_url:
                    lines.append(f"    Source: {r.source_filename} ({r.official_url})")
            lines.append("")

        # 4. Missing information needed
        if eval_res.missing_information:
            lines.append("? Information Still Needed for Verification:")
            for f in eval_res.missing_information:
                lines.append(f"  • Please provide '{f}' to definitively evaluate remaining criteria.")
            lines.append("")

        # 5. Unstructured/Departmental criteria
        if eval_res.unstructured_criteria:
            lines.append("📋 Official Verification & Ground-Truthing Requirements:")
            for unc in eval_res.unstructured_criteria:
                lines.append(f"  • {unc.get('description', '')}")
                lines.append(f"    Verification mode: {unc.get('reason_unstructured', '')}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_recommendation_report(result: RecommendationResult, max_schemes: int = 5) -> str:
        """
        Format the top-K scheme recommendations into a user-friendly report.
        """
        lines: List[str] = []
        lines.append("=" * 75)
        lines.append("  SchemeIQ+ Personalized Scheme Discovery & Eligibility Report")
        lines.append("=" * 75)
        lines.append("")
        lines.append(f"Timestamp: {result.evaluation_timestamp}")
        lines.append(f"Total Official Schemes Assessed: {result.total_schemes_evaluated}")
        lines.append(f"Status Breakdown: {result.status_counts}")
        lines.append("")

        lines.append("--- USER PROFILE SUMMARY ---")
        for k, v in result.user_profile_summary.items():
            lines.append(f"  • {k}: {v}")
        lines.append("")

        lines.append(f"--- TOP {min(max_schemes, len(result.recommendations))} RECOMMENDED SCHEMES ---")
        lines.append("")

        for i, rec in enumerate(result.recommendations[:max_schemes], start=1):
            lines.append(f"[{i}] {rec.scheme_name} ({rec.scheme_id})")
            lines.append("-" * 50)
            lines.append(ExplanationFormatter.format_scheme_explanation(rec))
            lines.append("-" * 50)
            lines.append("")

        lines.append("=" * 75)
        lines.append("LEGAL & ADMINISTRATIVE DISCLAIMER:")
        lines.append(f"  {result.disclaimer}")
        lines.append("=" * 75)

        return "\n".join(lines)
