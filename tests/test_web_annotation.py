# -*- coding: utf-8 -*-
"""
tests/test_web_annotation.py — Unit & Integration Tests for Phase 7F.1 Web Annotation UI

Covers:
1. Web UI loading (GET /)
2. Session state & progress reporting (GET /api/session)
3. Strict candidate filtering (only ELIGIBLE and POTENTIALLY_ELIGIBLE; disqualified excluded)
4. Initial null grades (zero labels auto-generated or inferred)
5. Annotator ID validation & non-PII enforcement (POST /api/annotator)
6. Mandatory relevance grade enforcement (POST /api/submit rejects missing grades)
7. Invalid grade rejection (POST /api/submit rejects grades outside {1, 2, 3})
8. Valid judgment submission & progress update
9. Judgment update & duplicate prevention (updating judgment replaces prior record cleanly)
10. Navigation between archetypes (POST /api/navigate)
11. Summary endpoint (GET /api/summary)
12. End-to-end dataset saving and validation with RankingDatasetValidator (POST /api/save)
13. Safe isolation: all file writes directed to tmp_path
"""
import json
from pathlib import Path
import pytest

from src.eligibility.schemas import EligibilityStatus, UserProfile
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import CitizenArchetype, DatasetPurpose, RankingDataset
from src.recommendation.web_annotation import create_app


@pytest.fixture
def mock_archetypes_file(tmp_path):
    """Creates a temporary 2-archetype file for deterministic testing."""
    arch1 = {
        "archetype_id": "arch_test_farmer_01",
        "archetype_name": "Telangana Smallholder Farmer",
        "narrative": "45-year-old male farmer in Mahabubabad with 4 acres cultivable land.",
        "profile": {
            "age": 45,
            "gender": "male",
            "state": "Telangana",
            "residence_type": "rural",
            "occupation": "farmer",
            "farmer_status": True,
            "land_ownership_status": True,
            "land_acres": 4.0,
            "annual_income": 150000,
            "pays_income_tax": False,
            "has_savings_bank_account": True,
        },
    }
    arch2 = {
        "archetype_id": "arch_test_student_02",
        "archetype_name": "Low-Income Undergraduate Student",
        "narrative": "20-year-old undergraduate student in Hyderabad.",
        "profile": {
            "age": 20,
            "gender": "female",
            "state": "Telangana",
            "residence_type": "urban",
            "occupation": "student",
            "student_status": True,
            "annual_income": 85000,
            "category": "SC",
            "education_level": "undergraduate",
            "has_savings_bank_account": True,
        },
    }

    file_path = tmp_path / "test_archetypes.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([arch1, arch2], f)
    return file_path


@pytest.fixture
def test_client(mock_archetypes_file, tmp_path):
    """Initializes a Flask test client with temporary paths."""
    out_file = tmp_path / "test_annotations_output.json"
    app = create_app(
        archetypes_path=mock_archetypes_file,
        annotator_id="expert_reviewer_01",
        output_path=out_file,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, out_file


def test_ui_page_loads(test_client):
    client, _ = test_client
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "SchemeIQ+ Expert Relevance Annotation" in html
    assert "Broadly Relevant" in html
    assert "High Interest" in html
    assert "Primary Choice" in html
    assert "Save Dataset to Disk" in html


def test_session_state_and_zero_auto_inferred_labels(test_client):
    client, _ = test_client
    res = client.get("/api/session")
    assert res.status_code == 200
    data = res.get_json()

    assert data["annotator_id"] == "expert_reviewer_01"
    assert data["current_index"] == 0
    assert data["total_archetypes"] == 2
    assert data["progress_text"] == "Archetype 1 of 2"
    assert data["total_judgments"] == 0
    assert data["archetype"]["archetype_id"] == "arch_test_farmer_01"

    candidates = data["candidates"]
    assert len(candidates) > 0

    # ZERO LABEL GENERATION: all candidate grades must be null initially
    for cand in candidates:
        assert cand["current_grade"] is None, "Grade must not be pre-populated or inferred"
        assert cand["current_justification"] is None
        # Must be strictly ELIGIBLE or POTENTIALLY_ELIGIBLE
        assert cand["eligibility_status"] in {
            EligibilityStatus.ELIGIBLE.value,
            EligibilityStatus.POTENTIALLY_ELIGIBLE.value,
        }

    # Verify disqualified schemes are excluded (Male farmer -> TS004 female only excluded; age 45 -> CT006 max age 40 excluded)
    candidate_ids = {c["scheme_id"] for c in candidates}
    assert "TS004" not in candidate_ids
    assert "CT006" not in candidate_ids


def test_annotator_id_validation(test_client):
    client, _ = test_client

    # Valid non-PII ID
    res = client.post("/api/annotator", json={"annotator_id": "policy_expert_02"})
    assert res.status_code == 200
    assert res.get_json()["annotator_id"] == "policy_expert_02"

    # Reject empty
    res = client.post("/api/annotator", json={"annotator_id": "   "})
    assert res.status_code == 400
    assert "cannot be empty" in res.get_json()["error"]

    # Reject email PII
    res = client.post("/api/annotator", json={"annotator_id": "expert@telangana.gov.in"})
    assert res.status_code == 400
    assert "email" in res.get_json()["error"]

    # Reject phone/Aadhaar digit patterns
    res = client.post("/api/annotator", json={"annotator_id": "expert_9876543210"})
    assert res.status_code == 400
    assert "10+ digits" in res.get_json()["error"]

    # Reject prohibited sensitive keywords
    res = client.post("/api/annotator", json={"annotator_id": "expert_pan_number_verifier"})
    assert res.status_code == 400
    assert "prohibited sensitive keywords" in res.get_json()["error"]


def test_submit_rejects_missing_grades(test_client):
    client, _ = test_client
    # Fetch candidates for archetype 0
    res = client.get("/api/session")
    candidates = res.get_json()["candidates"]

    # Attempt submission with one candidate missing a grade
    partial_judgments = [
        {"scheme_id": candidates[0]["scheme_id"], "grade": 3, "justification": "Primary scheme"}
    ]
    # Rest of candidates left out
    sub_res = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": partial_judgments,
        "next_index": 1,
    })
    assert sub_res.status_code == 400
    err = sub_res.get_json()["error"]
    assert "Missing relevance grade" in err


def test_submit_rejects_invalid_grades(test_client):
    client, _ = test_client
    res = client.get("/api/session")
    candidates = res.get_json()["candidates"]

    # Set invalid grade 0 and 4
    judgments = [{"scheme_id": c["scheme_id"], "grade": 1} for c in candidates]
    judgments[0]["grade"] = 0  # Invalid grade

    sub_res = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": judgments,
        "next_index": 1,
    })
    assert sub_res.status_code == 400
    assert "Invalid relevance grade" in sub_res.get_json()["error"]

    judgments[0]["grade"] = 4  # Out of range
    sub_res2 = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": judgments,
        "next_index": 1,
    })
    assert sub_res2.status_code == 400
    assert "Invalid relevance grade" in sub_res2.get_json()["error"]


def test_submit_rejects_disqualified_scheme(test_client):
    client, _ = test_client
    res = client.get("/api/session")
    candidates = res.get_json()["candidates"]

    judgments = [{"scheme_id": c["scheme_id"], "grade": 1} for c in candidates]
    # Artificially inject TS004 (female only) for male farmer
    judgments.append({"scheme_id": "TS004", "grade": 3})

    sub_res = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": judgments,
    })
    assert sub_res.status_code == 400
    assert "not an eligible candidate" in sub_res.get_json()["error"]


def test_valid_submission_and_duplicate_prevention(test_client):
    client, _ = test_client

    # 1. Fetch Archetype 0 candidates
    res = client.get("/api/session")
    cands0 = res.get_json()["candidates"]
    total_cands0 = len(cands0)

    # Submit valid grades
    judgments0 = [
        {"scheme_id": c["scheme_id"], "grade": (idx % 3) + 1, "justification": f"Note for {c['scheme_id']}"}
        for idx, c in enumerate(cands0)
    ]
    sub_res = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": judgments0,
        "next_index": 1,
    })
    assert sub_res.status_code == 200
    assert sub_res.get_json()["total_records"] == total_cands0
    assert sub_res.get_json()["current_index"] == 1

    # 2. Update existing judgments for Archetype 0 (revise ratings)
    revised_judgments0 = [
        {"scheme_id": c["scheme_id"], "grade": 3, "justification": "Revised to top priority"}
        for c in cands0
    ]
    update_res = client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": revised_judgments0,
        "next_index": 1,
    })
    assert update_res.status_code == 200
    # Total records should remain the same (no duplicates)
    assert update_res.get_json()["total_records"] == total_cands0

    # Verify the updated judgments took effect
    client.post("/api/navigate", json={"target_index": 0})
    res_revisit = client.get("/api/session")
    for cand in res_revisit.get_json()["candidates"]:
        assert cand["current_grade"] == 3
        assert cand["current_justification"] == "Revised to top priority"


def test_navigation(test_client):
    client, _ = test_client

    # Navigate to index 1
    res = client.post("/api/navigate", json={"target_index": 1})
    assert res.status_code == 200
    assert res.get_json()["current_index"] == 1

    # Check session reflects index 1
    sess_res = client.get("/api/session")
    assert sess_res.get_json()["current_index"] == 1
    assert sess_res.get_json()["progress_text"] == "Archetype 2 of 2"
    assert sess_res.get_json()["archetype"]["archetype_id"] == "arch_test_student_02"

    # Reject out of bounds
    res_bad = client.post("/api/navigate", json={"target_index": 5})
    assert res_bad.status_code == 400


def test_save_empty_dataset_rejected(test_client):
    client, _ = test_client
    res = client.post("/api/save", json={})
    assert res.status_code == 400
    assert "No judgments recorded yet" in res.get_json()["error"]


def test_end_to_end_annotation_and_dataset_validation(test_client):
    client, out_file = test_client

    # Annotate Archetype 0 (Farmer)
    res0 = client.get("/api/session")
    cands0 = res0.get_json()["candidates"]
    j0 = [{"scheme_id": c["scheme_id"], "grade": 2} for c in cands0]
    j0[0]["grade"] = 3
    client.post("/api/submit", json={
        "archetype_id": "arch_test_farmer_01",
        "judgments": j0,
        "next_index": 1,
    })

    # Annotate Archetype 1 (Student)
    res1 = client.get("/api/session")
    cands1 = res1.get_json()["candidates"]
    j1 = [{"scheme_id": c["scheme_id"], "grade": 1} for c in cands1]
    j1[0]["grade"] = 3
    client.post("/api/submit", json={
        "archetype_id": "arch_test_student_02",
        "judgments": j1,
        "next_index": 1,
    })

    # Verify summary
    summary_res = client.get("/api/summary")
    assert summary_res.status_code == 200
    summary = summary_res.get_json()
    assert summary["total_records"] == len(cands0) + len(cands1)
    assert summary["label_distribution"]["3"] >= 2

    # Save dataset to disk
    save_res = client.post("/api/save", json={"output_path": str(out_file)})
    assert save_res.status_code == 200
    save_data = save_res.get_json()
    assert save_data["status"] == "ok"
    assert save_data["validator_passed"] is True
    assert save_data["total_groups"] == 2
    assert save_data["total_records"] == len(cands0) + len(cands1)

    # Verify file was written and can be loaded via RankingDatasetIO
    assert out_file.exists()
    loaded_dataset = RankingDatasetIO.load_json(out_file)
    assert isinstance(loaded_dataset, RankingDataset)
    assert loaded_dataset.metadata.purpose == DatasetPurpose.EXPERT_CURATED_BENCHMARK
    assert len(loaded_dataset.records) == len(cands0) + len(cands1)

    # Re-run RankingDatasetValidator directly on the loaded dataset
    report = RankingDatasetValidator.validate_dataset(loaded_dataset)
    assert report.is_valid is True
    assert len(report.errors) == 0
