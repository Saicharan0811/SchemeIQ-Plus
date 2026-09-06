# -*- coding: utf-8 -*-
"""
tests/test_annotation_tool.py — Unit & Integration Tests for Phase 7E Expert Annotation Tool

Tests:
1. Archetype loading from file (structure, validation, privacy checks)
2. Non-PII annotator ID validation
3. Pre-filtering: candidate schemes must strictly include only ELIGIBLE and POTENTIALLY_ELIGIBLE;
   NOT_ELIGIBLE and NOT_APPLICABLE schemes are strictly excluded.
4. Relevance label assignment: y in {1, 2, 3} accepted; out-of-range or disqualified schemes rejected.
5. Multi-expert session independence.
6. Dataset generation and compliance with RankingDatasetValidator and RankingDatasetIO.
"""
import json
import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.eligibility.schemas import EligibilityStatus, UserProfile
from src.recommendation.annotation_tool import (
    ELIGIBLE_CANDIDATE_STATUSES,
    VALID_EXPERT_RELEVANCE_GRADES,
    ExpertAnnotationSession,
    load_archetypes_from_file,
    validate_annotator_id,
)
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import (
    CitizenArchetype,
    DatasetPurpose,
    RankingDataset,
)


@pytest.fixture
def sample_archetype():
    return CitizenArchetype(
        archetype_id="arch_test_farmer_01",
        archetype_name="Telangana Smallholder Farmer",
        narrative="45-year-old male farmer in Mahabubabad with 4 acres cultivable land.",
        profile=UserProfile(
            age=45,
            gender="male",
            state="Telangana",
            residence_type="rural",
            occupation="farmer",
            farmer_status=True,
            land_ownership_status=True,
            land_acres=4.0,
            annual_income=150000,
            pays_income_tax=False,
            has_savings_bank_account=True,
        ),
    )


@pytest.fixture
def sample_archetype_elderly():
    return CitizenArchetype(
        archetype_id="arch_test_elderly_02",
        archetype_name="Senior Citizen in Telangana",
        narrative="72-year-old woman in rural Telangana.",
        profile=UserProfile(
            age=72,
            gender="female",
            state="Telangana",
            residence_type="rural",
            pensioner_category="old_age",
            pensioner_status=True,
            has_savings_bank_account=True,
        ),
    )


def test_validate_annotator_id():
    # Valid non-PII IDs
    assert validate_annotator_id("expert_01") == "expert_01"
    assert validate_annotator_id("policy_researcher_A") == "policy_researcher_A"
    assert validate_annotator_id("welfare_officer_12") == "welfare_officer_12"

    # Empty
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_annotator_id("   ")

    # Email address rejection
    with pytest.raises(ValueError, match="Privacy Violation.*email"):
        validate_annotator_id("expert_john@gov.in")

    # Prohibited keyword rejection
    with pytest.raises(ValueError, match="Privacy Violation.*prohibited"):
        validate_annotator_id("expert_pan_number_verifier")

    # Phone/Aadhaar digit pattern rejection
    with pytest.raises(ValueError, match="Privacy Violation.*10\\+ digits"):
        validate_annotator_id("annotator_9876543210")


def test_load_archetypes_from_file(tmp_path):
    template_path = root / "data" / "ranking" / "archetypes_template.json"
    assert template_path.exists(), "archetypes_template.json must exist"

    archetypes = load_archetypes_from_file(template_path)
    assert len(archetypes) >= 2
    assert archetypes[0].archetype_id == "arch_template_farmer_01"
    assert archetypes[0].profile.state == "Telangana"

    # Privacy check failure when prohibited sensitive field is in profile
    bad_archetypes_file = tmp_path / "bad_archetypes.json"
    with open(bad_archetypes_file, "w", encoding="utf-8") as f:
        json.dump({
            "archetypes": [{
                "archetype_id": "bad_arch",
                "archetype_name": "Violating Archetype",
                "profile": {
                    "age": 30,
                    "state": "Telangana",
                    "aadhaar_number": "1234-5678-9012"
                }
            }]
        }, f)

    with pytest.raises(ValueError, match="Privacy Violation"):
        load_archetypes_from_file(bad_archetypes_file)


def test_candidate_prefiltering_excludes_disqualified_schemes(sample_archetype):
    session = ExpertAnnotationSession(annotator_id="expert_01")
    candidates = session.get_candidate_schemes(sample_archetype)

    assert len(candidates) > 0

    # Ensure EVERY candidate is strictly ELIGIBLE or POTENTIALLY_ELIGIBLE
    for cand in candidates:
        assert cand.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES
        assert cand.eligibility_status not in {
            EligibilityStatus.NOT_ELIGIBLE,
            EligibilityStatus.NOT_APPLICABLE,
        }

    candidate_scheme_ids = {c.scheme_id for c in candidates}

    # Archetype is male farmer aged 45:
    # 1. TS004 (Maha Lakshmi) is female only -> NOT_APPLICABLE -> MUST BE EXCLUDED
    assert "TS004" not in candidate_scheme_ids, "TS004 (female only) must be excluded for male archetype"

    # 2. CT006 (Atal Pension Yojana) max age is 40 -> NOT_ELIGIBLE (age 45) -> MUST BE EXCLUDED
    assert "CT006" not in candidate_scheme_ids, "CT006 (age 18-40) must be excluded for age 45 archetype"

    # 3. TS001 (Rythu Bharosa) and TS002 (Rythu Bima) should be included
    assert "TS001" in candidate_scheme_ids
    assert "TS002" in candidate_scheme_ids


def test_record_judgment_valid_and_invalid(sample_archetype):
    session = ExpertAnnotationSession(annotator_id="expert_01")
    candidates = session.get_candidate_schemes(sample_archetype)
    ts001_eval = next(c for c in candidates if c.scheme_id == "TS001")

    # Valid relevance grades (1, 2, 3)
    rec1 = session.record_judgment(
        archetype=sample_archetype,
        candidate_eval=ts001_eval,
        relevance_label=3,
        justification="Top priority direct investment support for landholding farmer.",
    )
    assert rec1.relevance_label == 3
    assert rec1.scheme_id == "TS001"
    assert rec1.session_id == "expert_01"
    assert rec1.notes == "Top priority direct investment support for landholding farmer."

    # Duplicate judgment in same session rejected
    with pytest.raises(ValueError, match="already recorded"):
        session.record_judgment(
            archetype=sample_archetype,
            candidate_eval=ts001_eval,
            relevance_label=2,
        )

    # Invalid relevance label (0 is not an allowable grade for eligible candidates; 4 is out of range)
    ts002_eval = next(c for c in candidates if c.scheme_id == "TS002")
    with pytest.raises(ValueError, match="Invalid relevance_label"):
        session.record_judgment(
            archetype=sample_archetype,
            candidate_eval=ts002_eval,
            relevance_label=0,
        )

    with pytest.raises(ValueError, match="Invalid relevance_label"):
        session.record_judgment(
            archetype=sample_archetype,
            candidate_eval=ts002_eval,
            relevance_label=4,
        )


def test_cannot_record_judgment_for_disqualified_scheme(sample_archetype):
    session = ExpertAnnotationSession(annotator_id="expert_01")
    all_evals = session.evaluator.evaluate_all(sample_archetype.profile)

    # Find a disqualified scheme (e.g. TS004 or CT006)
    disqualified_eval = next(
        e for e in all_evals
        if e.eligibility_status in {EligibilityStatus.NOT_ELIGIBLE, EligibilityStatus.NOT_APPLICABLE}
    )

    with pytest.raises(ValueError, match="Disqualified schemes are kept outside"):
        session.record_judgment(
            archetype=sample_archetype,
            candidate_eval=disqualified_eval,
            relevance_label=2,
        )


def test_multi_annotator_independence(sample_archetype):
    candidates = ExpertAnnotationSession("temp").get_candidate_schemes(sample_archetype)
    cand_eval = candidates[0]

    # Expert A
    session_a = ExpertAnnotationSession(annotator_id="expert_alice")
    rec_a = session_a.record_judgment(sample_archetype, cand_eval, relevance_label=3, justification="Primary for Alice")

    # Expert B
    session_b = ExpertAnnotationSession(annotator_id="expert_bob")
    rec_b = session_b.record_judgment(sample_archetype, cand_eval, relevance_label=2, justification="Secondary for Bob")

    assert rec_a.session_id == "expert_alice"
    assert rec_b.session_id == "expert_bob"
    assert rec_a.record_id != rec_b.record_id
    assert rec_a.relevance_label == 3
    assert rec_b.relevance_label == 2


def test_build_and_save_annotations_passes_validator(sample_archetype, sample_archetype_elderly, tmp_path):
    session = ExpertAnnotationSession(annotator_id="expert_reviewer_01")

    # Annotate Archetype 1 (Farmer) - 2 candidates
    cands1 = session.get_candidate_schemes(sample_archetype)
    session.record_judgment(sample_archetype, cands1[0], relevance_label=3)
    session.record_judgment(sample_archetype, cands1[1], relevance_label=2)

    # Annotate Archetype 2 (Elderly) - 2 candidates
    cands2 = session.get_candidate_schemes(sample_archetype_elderly)
    session.record_judgment(sample_archetype_elderly, cands2[0], relevance_label=3)
    session.record_judgment(sample_archetype_elderly, cands2[1], relevance_label=1)

    # Build dataset
    dataset = session.build_dataset(dataset_name="Test_Expert_Benchmark")
    assert isinstance(dataset, RankingDataset)
    assert dataset.metadata.purpose == DatasetPurpose.EXPERT_CURATED_BENCHMARK
    assert dataset.metadata.total_records == 4
    assert dataset.metadata.total_groups == 2

    # Verify existing RankingDatasetValidator passes
    report = RankingDatasetValidator.validate_dataset(dataset)
    assert report.is_valid is True
    assert len(report.errors) == 0

    # Save and reload
    save_file = tmp_path / "expert_annotations.json"
    saved_path = session.save_annotations(save_file)
    assert saved_path.exists()

    reloaded = RankingDatasetIO.load_json(saved_path)
    assert len(reloaded.records) == 4
    assert reloaded.metadata.purpose == DatasetPurpose.EXPERT_CURATED_BENCHMARK
