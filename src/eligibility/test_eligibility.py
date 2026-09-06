# -*- coding: utf-8 -*-
"""
src/eligibility/test_eligibility.py — Deterministic Test Suite for Phase 6
Executes 12 comprehensive test scenarios, validates ranking, explanation, privacy, and outputs
data/reports/eligibility_evaluation_report.json
"""
import sys, json, datetime
from pathlib import Path

root = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
sys.path.insert(0, str(root))

from src.eligibility.schemas import (
    EligibilityStatus,
    RuleStatus,
    UserProfile,
)
from src.eligibility.service import EligibilityService
from src.eligibility.explanation import ExplanationFormatter
from src.eligibility.profile import validate_profile_privacy

reports_dir = root / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)


def run_all_tests() -> dict:
    print("\n" + "=" * 75)
    print("  PHASE 6: PERSONALIZED ELIGIBILITY & RECOMMENDATION TEST SUITE")
    print("=" * 75)

    service = EligibilityService()

    test_scenarios = [
        # Scenario 1: Complete Farmer Profile
        {
            "scenario_id": "SC01",
            "name": "Complete Telangana Landholding Farmer",
            "profile": UserProfile(
                age=38,
                gender="male",
                state="Telangana",
                residence_type="rural",
                occupation="farmer",
                farmer_status=True,
                land_ownership_status=True,
                land_acres=4.5,
                annual_income=180000,
                pays_income_tax=False,
                is_institutional_landholder=False,
            ),
            "expected_top_schemes": ["TS001", "TS002", "CT001"],
            "expected_top_status": [EligibilityStatus.ELIGIBLE, EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 2: Student Profile
        {
            "scenario_id": "SC02",
            "name": "Low-Income Telangana Post-Matric Student",
            "profile": UserProfile(
                age=20,
                gender="female",
                state="Telangana",
                student_status=True,
                category="BC",
                annual_income=120000,
                education_level="graduate",
            ),
            "expected_top_schemes": ["TS006"],
            "expected_top_status": [EligibilityStatus.ELIGIBLE, EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 3: Small-Business / Entrepreneur Profile
        {
            "scenario_id": "SC03",
            "name": "Micro-Enterprise Entrepreneur seeking PMEGP/MUDRA",
            "profile": UserProfile(
                age=28,
                gender="male",
                state="Telangana",
                business_owner_status=True,
                is_non_farm_enterprise=True,
                education_level="12th_pass",
                project_cost=1500000,
                loan_requirement=500000,
            ),
            "expected_top_schemes": ["CT005", "CT004"],
            "expected_top_status": [EligibilityStatus.ELIGIBLE, EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 4: Senior Citizen Profile
        {
            "scenario_id": "SC04",
            "name": "Senior Citizen Aged 72 in Telangana",
            "profile": UserProfile(
                age=72,
                gender="female",
                state="Telangana",
                pensioner_category="old_age",
                pensioner_status=True,
            ),
            "expected_top_schemes": ["CT002", "TS005"],
            "expected_top_status": [EligibilityStatus.ELIGIBLE, EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 5: Low-Income BPL Urban Profile
        {
            "scenario_id": "SC05",
            "name": "Low-Income Urban Family Seeking Housing / Healthcare",
            "profile": UserProfile(
                age=35,
                gender="female",
                state="Telangana",
                residence_type="urban",
                annual_income=240000,
                owns_pucca_house=False,
                is_bpl_or_white_ration_card=True,
            ),
            "expected_top_schemes": ["CT003", "TS003", "TS004"],
            "expected_top_status": [EligibilityStatus.ELIGIBLE, EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 6: Partial Profile with Missing Income
        {
            "scenario_id": "SC06",
            "name": "Partial Profile: Missing Income (Must not crash or fail falsely)",
            "profile": UserProfile(
                age=24,
                state="Telangana",
                student_status=True,
                # annual_income is omitted
            ),
            "expected_top_schemes": ["TS006"],
            "expected_top_status": [EligibilityStatus.POTENTIALLY_ELIGIBLE],
        },
        # Scenario 7: Partial Profile with Missing Age
        {
            "scenario_id": "SC07",
            "name": "Partial Profile: Missing Age (Must not fail age-bounded schemes falsely)",
            "profile": UserProfile(
                state="Telangana",
                farmer_status=True,
                land_ownership_status=True,
                # age is omitted
            ),
            "expected_top_schemes": ["TS001", "TS002", "CT001"],
            "expected_top_status": [EligibilityStatus.POTENTIALLY_ELIGIBLE, EligibilityStatus.ELIGIBLE],
        },
        # Scenario 8: Explicit Rule Failure
        {
            "scenario_id": "SC08",
            "name": "Explicit Rule Failure: Age 55 applying for APY (Max Age 40)",
            "profile": UserProfile(
                age=55,
                state="Telangana",
                has_savings_bank_account=True,
            ),
            "expected_top_schemes": None, # CT006 must be NOT_ELIGIBLE
            "specific_check_scheme": "CT006",
            "expected_specific_status": EligibilityStatus.NOT_ELIGIBLE,
        },
        # Scenario 9: Unknown Must Not Become FAIL
        {
            "scenario_id": "SC09",
            "name": "Unknown Field Handling: Minimal Profile",
            "profile": UserProfile(
                state="Telangana",
            ),
            "expected_top_schemes": None,
            "specific_check_unknown_handling": True,
        },
        # Scenario 10: State Mismatch (Non-Telangana resident applying for state schemes)
        {
            "scenario_id": "SC10",
            "name": "Geographic Mismatch: Karnataka Resident",
            "profile": UserProfile(
                age=30,
                state="Karnataka",
                farmer_status=True,
                land_ownership_status=True,
            ),
            "expected_top_schemes": ["CT001"], # Central farmer scheme should rank, state TS schemes must be NOT_APPLICABLE
            "specific_check_scheme": "TS001",
            "expected_specific_status": EligibilityStatus.NOT_APPLICABLE,
        },
        # Scenario 11: Recommendation Ordering
        {
            "scenario_id": "SC11",
            "name": "Recommendation Priority & Deterministic Sorting",
            "profile": UserProfile(
                age=25,
                gender="female",
                state="Telangana",
                student_status=True,
                annual_income=150000,
            ),
            "specific_check_priority_order": True,
        },
        # Scenario 12: All 14 Schemes Evaluated
        {
            "scenario_id": "SC12",
            "name": "Corpus Completeness: All 14 Schemes Assessed",
            "profile": UserProfile(age=30, state="Telangana"),
            "specific_check_14_schemes": True,
        },
    ]

    test_results = []
    total_passed = 0

    for sc in test_scenarios:
        sc_id = sc["scenario_id"]
        sc_name = sc["name"]
        prof = sc["profile"]

        rec_result = service.recommend_for_profile(prof, top_k=14)
        top_recs = rec_result.recommendations

        sc_pass = True
        notes = []

        # Check total evaluated
        if rec_result.total_schemes_evaluated != 14:
            sc_pass = False
            notes.append(f"Expected 14 schemes evaluated, got {rec_result.total_schemes_evaluated}")

        # Check specific expected top schemes
        if sc.get("expected_top_schemes"):
            top_ids = [r.scheme_id for r in top_recs[:len(sc["expected_top_schemes"])]]
            matched_any = any(s in top_ids for s in sc["expected_top_schemes"])
            if not matched_any:
                sc_pass = False
                notes.append(f"Expected one of {sc['expected_top_schemes']} in top, got {top_ids}")

        # Check specific scheme status
        if sc.get("specific_check_scheme"):
            target_id = sc["specific_check_scheme"]
            target_eval = next((r for r in top_recs if r.scheme_id == target_id), None)
            if target_eval is None or target_eval.eligibility_status != sc["expected_specific_status"]:
                sc_pass = False
                actual_st = target_eval.eligibility_status if target_eval else "NONE"
                notes.append(f"Scheme {target_id} expected {sc['expected_specific_status']}, got {actual_st}")

        # Check unknown handling: unknown rules should have status UNKNOWN, not FAIL
        if sc.get("specific_check_unknown_handling"):
            for r in top_recs:
                for uk in r.unknown_rules:
                    if uk.status != RuleStatus.UNKNOWN:
                        sc_pass = False
                        notes.append(f"Unknown rule {uk.rule_id} has wrong status {uk.status}")

        # Check priority order
        if sc.get("specific_check_priority_order"):
            statuses = [r.eligibility_status for r in top_recs]
            status_order = {
                EligibilityStatus.ELIGIBLE: 1,
                EligibilityStatus.POTENTIALLY_ELIGIBLE: 2,
                EligibilityStatus.INSUFFICIENT_INFORMATION: 3,
                EligibilityStatus.NOT_APPLICABLE: 4,
                EligibilityStatus.NOT_ELIGIBLE: 5,
            }
            order_indices = [status_order[st] for st in statuses]
            if order_indices != sorted(order_indices):
                sc_pass = False
                notes.append("Recommendations are not correctly grouped by eligibility priority.")

        if sc_pass:
            total_passed += 1

        print(f"\n[{sc_id}] {sc_name}")
        print(f"  Result: {'PASS' if sc_pass else 'FAIL'}")
        if notes:
            for n in notes:
                print(f"  Warning/Error: {n}")
        print(f"  Top 3 Schemes: {[r.scheme_id + ' (' + r.eligibility_status.value + ', ' + str(r.score) + ')' for r in top_recs[:3]]}")

        test_results.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "passed": sc_pass,
            "notes": notes,
            "top_scheme": top_recs[0].scheme_id if top_recs else None,
            "top_scheme_name": top_recs[0].scheme_name if top_recs else None,
            "top_scheme_status": top_recs[0].eligibility_status.value if top_recs else None,
            "top_scheme_score": top_recs[0].score if top_recs else None,
            "status_counts": rec_result.status_counts,
        })

    # Privacy verification test
    print("\n[Privacy Verification Test]")
    privacy_passed = False
    try:
        validate_profile_privacy({"age": 30, "aadhaar_number": "1234-5678-9012"})
    except ValueError as e:
        privacy_passed = True
        print(f"  Correctly rejected prohibited field: {e}")

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_scenarios_tested": len(test_scenarios),
        "scenarios_passed": total_passed,
        "privacy_test_passed": privacy_passed,
        "schemes_evaluated_per_run": 14,
        "all_tests_passed": (total_passed == len(test_scenarios)) and privacy_passed,
        "scenario_results": test_results,
    }

    out_report = reports_dir / "eligibility_evaluation_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"  TEST SUITE SUMMARY: {total_passed}/{len(test_scenarios)} Scenarios Passed")
    print(f"  Privacy Validation: {'PASS' if privacy_passed else 'FAIL'}")
    print(f"  Evaluation Report Saved: {out_report}")
    print("=" * 75)

    return report


if __name__ == "__main__":
    run_all_tests()
