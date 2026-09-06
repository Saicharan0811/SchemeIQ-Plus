# -*- coding: utf-8 -*-
"""
tests/test_annotation_workflow.py — Unit & Integration Tests for Phase 7F Workflow

Tests:
1. Review sheet generation (verifying candidates are strictly ELIGIBLE / POTENTIALLY_ELIGIBLE)
2. Review sheet export to file
3. Batch ingestion of completed review sheets:
   - Valid ingestion with grades 1, 2, 3
   - Rejection of invalid grades (e.g. 0, 4)
   - Rejection of disqualified schemes injected into review sheet
   - Skipping unannotated (null) candidates
4. End-to-end dataset validation using RankingDatasetValidator
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
    ExpertAnnotationSession,
    load_archetypes_from_file,
)
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import (
    CitizenArchetype,
    DatasetPurpose,
    RankingDataset,
)


@pytest.fixture
def sample_archetypes():
    arch1 = CitizenArchetype(
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
    arch2 = CitizenArchetype(
        archetype_id="arch_test_student_02",
        archetype_name="Low-Income Undergraduate Student",
        narrative="20-year-old undergraduate student in Hyderabad.",
        profile=UserProfile(
            age=20,
            gender="female",
            state="Telangana",
            residence_type="urban",
            occupation="student",
            student_status=True,
            annual_income=85000,
            category="SC",
            education_level="graduate",
            has_savings_bank_account=True,
        ),
    )
    return [arch1, arch2]


def test_generate_review_sheet(sample_archetypes):
    session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    sheet = session.generate_review_sheet(sample_archetypes)

    assert "metadata" in sheet
    assert sheet["metadata"]["annotator_id"] == "expert_evaluator_01"
    assert sheet["metadata"]["total_archetypes"] == 2
    assert "rubric" in sheet["metadata"]

    reviews = sheet["candidate_reviews"]
    assert len(reviews) == 2

    for rev in reviews:
        assert rev["eligible_candidates_count"] > 0
        for cand in rev["candidates"]:
            assert cand["eligibility_status"] in {
                EligibilityStatus.ELIGIBLE.value,
                EligibilityStatus.POTENTIALLY_ELIGIBLE.value,
            }
            # Grades should start as null for expert to fill
            assert cand["relevance_grade"] is None
            assert cand["justification"] is None

    # Male farmer: TS004 (female only) and CT006 (age <= 40) must NOT appear in candidates
    farmer_candidates = [c["scheme_id"] for c in reviews[0]["candidates"]]
    assert "TS004" not in farmer_candidates
    assert "CT006" not in farmer_candidates


def test_export_review_sheet_to_file(sample_archetypes, tmp_path):
    session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    out_file = tmp_path / "test_review_sheet.json"

    saved = session.export_review_sheet(sample_archetypes, out_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["candidate_reviews"]) == 2


def test_record_batch_from_completed_review_sheet(sample_archetypes, tmp_path):
    session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    out_file = tmp_path / "test_review_sheet.json"
    session.export_review_sheet(sample_archetypes, out_file)

    # Simulate expert filling in review sheet
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # In archetype 1, fill first candidate as 3, second as 2, leave third as None (skip)
    data["candidate_reviews"][0]["candidates"][0]["relevance_grade"] = 3
    data["candidate_reviews"][0]["candidates"][0]["justification"] = "Top priority crop support."
    data["candidate_reviews"][0]["candidates"][1]["relevance_grade"] = 2

    # In archetype 2, fill first candidate as 3, second as 1
    data["candidate_reviews"][1]["candidates"][0]["relevance_grade"] = 3
    data["candidate_reviews"][1]["candidates"][1]["relevance_grade"] = 1

    completed_file = tmp_path / "completed_review_sheet.json"
    with open(completed_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Ingest completed review sheet
    ingest_session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    records = ingest_session.record_batch_from_review_sheet(
        archetypes=sample_archetypes,
        review_sheet_path=completed_file,
    )

    assert len(records) == 4
    assert records[0].relevance_label == 3
    assert records[0].notes == "Top priority crop support."
    assert records[1].relevance_label == 2
    assert records[2].relevance_label == 3
    assert records[3].relevance_label == 1

    # Validate resulting dataset
    dataset = ingest_session.build_dataset("Test_Batch_Expert_Annotations")
    report = RankingDatasetValidator.validate_dataset(dataset)
    assert report.is_valid is True
    assert len(report.errors) == 0

    # Save and reload
    save_path = tmp_path / "saved_expert_dataset.json"
    ingest_session.save_annotations(save_path)
    reloaded = RankingDatasetIO.load_json(save_path)
    assert len(reloaded.records) == 4
    assert reloaded.metadata.purpose == DatasetPurpose.EXPERT_CURATED_BENCHMARK


def test_batch_ingestion_rejects_invalid_grades(sample_archetypes, tmp_path):
    session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    out_file = tmp_path / "sheet_invalid_grade.json"
    session.export_review_sheet(sample_archetypes, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set invalid grade 0 (not allowed for candidates)
    data["candidate_reviews"][0]["candidates"][0]["relevance_grade"] = 0

    bad_file = tmp_path / "bad_grade.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    session_fail = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    with pytest.raises(ValueError, match="Invalid relevance_label"):
        session_fail.record_batch_from_review_sheet(sample_archetypes, bad_file)


def test_batch_ingestion_rejects_disqualified_scheme(sample_archetypes, tmp_path):
    session = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    out_file = tmp_path / "sheet_disqualified.json"
    session.export_review_sheet(sample_archetypes, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Artificially inject a disqualified scheme (TS004 Maha Lakshmi into male farmer candidates)
    data["candidate_reviews"][0]["candidates"].append({
        "scheme_id": "TS004",
        "scheme_name": "Maha Lakshmi",
        "category": "Women",
        "relevance_grade": 3,
    })

    bad_file = tmp_path / "bad_scheme.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    session_fail = ExpertAnnotationSession(annotator_id="expert_evaluator_01")
    with pytest.raises(ValueError, match="not an eligible candidate"):
        session_fail.record_batch_from_review_sheet(sample_archetypes, bad_file)
