# -*- coding: utf-8 -*-
"""
src/recommendation/annotation_tool.py — Expert Annotation Tool for SchemeIQ+ (Phase 7E)

Implements a local annotation workflow for collecting REAL expert relevance judgments:
1. Loads citizen archetype profiles from a JSON input file.
2. Runs the deterministic SchemeEligibilityEvaluator.
3. Filters candidates: only ELIGIBLE and POTENTIALLY_ELIGIBLE schemes are presented for annotation.
   NOT_ELIGIBLE and NOT_APPLICABLE schemes are strictly excluded from the ML candidate set.
4. Allows experts to assign relevance grades y in {1, 2, 3} with optional justification.
5. Preserves expert identity strictly as a non-PII annotator identifier (e.g. 'expert_01').
6. Compiles annotations into the Phase 7B RankingDataset schema.
7. Automatically validates all outputs using RankingDatasetValidator before serialization.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.profile import validate_profile_privacy
from src.eligibility.schemas import (
    PROHIBITED_SENSITIVE_KEYWORDS,
    EligibilityStatus,
    SchemeEvaluation,
    UserProfile,
)
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.feature_extractor import RankingFeatureExtractor
from src.recommendation.schemas import (
    CitizenArchetype,
    DatasetPurpose,
    RankingDataset,
    RankingDatasetMetadata,
    RankingRecord,
)

logger = logging.getLogger(__name__)

# Allowable relevance grades for eligible candidate schemes
VALID_EXPERT_RELEVANCE_GRADES = {1, 2, 3}

# Eligible status whitelist for ML candidates
ELIGIBLE_CANDIDATE_STATUSES = {
    EligibilityStatus.ELIGIBLE,
    EligibilityStatus.POTENTIALLY_ELIGIBLE,
}


def validate_annotator_id(annotator_id: str) -> str:
    """
    Validate that annotator_id is a non-empty, non-PII identifier.
    Rejects sensitive terms, email addresses, and phone numbers.
    """
    if not annotator_id or not annotator_id.strip():
        raise ValueError("annotator_id cannot be empty")

    clean_id = annotator_id.strip()
    id_lower = clean_id.lower()

    # Reject common PII patterns
    if "@" in clean_id:
        raise ValueError(f"Privacy Violation: annotator_id '{clean_id}' appears to be an email address. Use a non-PII ID like 'expert_01'.")

    if any(kw in id_lower for kw in PROHIBITED_SENSITIVE_KEYWORDS):
        raise ValueError(f"Privacy Violation: annotator_id '{clean_id}' contains prohibited sensitive keywords.")

    digits_only = "".join(c for c in clean_id if c.isdigit())
    if len(digits_only) >= 10:
        raise ValueError(f"Privacy Violation: annotator_id '{clean_id}' contains 10+ digits, resembling a phone or identity number.")

    return clean_id


def load_archetypes_from_file(file_path: Path) -> List[CitizenArchetype]:
    """
    Load citizen archetypes from a JSON file.
    Supports either a list of archetype dicts or an object with an 'archetypes' list.
    Validates privacy and schemas for every archetype.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archetype file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_list = data.get("archetypes")
        if not isinstance(raw_list, list):
            raise ValueError(f"Invalid archetype format in {file_path}: Expected 'archetypes' list.")
    elif isinstance(data, list):
        raw_list = data
    else:
        raise ValueError(f"Invalid archetype format in {file_path}: Expected JSON list or object with 'archetypes' key.")

    archetypes: List[CitizenArchetype] = []
    seen_ids = set()

    for idx, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            raise ValueError(f"Archetype entry at index {idx} must be a dictionary.")

        # Privacy check on profile attributes
        prof_data = raw.get("profile", {})
        validate_profile_privacy(prof_data if isinstance(prof_data, dict) else prof_data.model_dump())

        arch = CitizenArchetype(**raw)
        if arch.archetype_id in seen_ids:
            raise ValueError(f"Duplicate archetype_id '{arch.archetype_id}' found in {file_path}.")
        seen_ids.add(arch.archetype_id)
        archetypes.append(arch)

    logger.info(f"Loaded {len(archetypes)} citizen archetypes from {file_path}")
    return archetypes


class ExpertAnnotationSession:
    """
    Manages an expert annotation session for citizen archetypes.
    """

    def __init__(
        self,
        annotator_id: str,
        evaluator: Optional[SchemeEligibilityEvaluator] = None,
    ):
        self.annotator_id = validate_annotator_id(annotator_id)
        self.evaluator = evaluator or SchemeEligibilityEvaluator()
        self.records: List[RankingRecord] = []
        self._seen_pairs: set[Tuple[str, str]] = set()

    def get_candidate_schemes(self, archetype: CitizenArchetype) -> List[SchemeEvaluation]:
        """
        Evaluate archetype with deterministic SchemeEligibilityEvaluator and return
        ONLY candidate schemes that are ELIGIBLE or POTENTIALLY_ELIGIBLE.
        Schemes flagged NOT_ELIGIBLE or NOT_APPLICABLE are strictly excluded.
        """
        all_evaluations = self.evaluator.evaluate_all(archetype.profile)

        # Filter strictly to eligible / potentially eligible candidates
        candidate_evals = [
            ev for ev in all_evaluations
            if ev.eligibility_status in ELIGIBLE_CANDIDATE_STATUSES
        ]
        return candidate_evals

    def record_judgment(
        self,
        archetype: CitizenArchetype,
        candidate_eval: SchemeEvaluation,
        relevance_label: int,
        justification: Optional[str] = None,
    ) -> RankingRecord:
        """
        Record an expert relevance judgment for an eligible candidate scheme.

        Args:
            archetype: The citizen archetype being evaluated.
            candidate_eval: The SchemeEvaluation for the candidate scheme.
            relevance_label: Relevance grade y in {1, 2, 3}.
            justification: Optional textual rationale for the assigned grade.

        Returns:
            The created RankingRecord.
        """
        if relevance_label not in VALID_EXPERT_RELEVANCE_GRADES:
            raise ValueError(
                f"Invalid relevance_label '{relevance_label}'. "
                f"Expert relevance grade for candidate schemes must be in {sorted(VALID_EXPERT_RELEVANCE_GRADES)}: "
                "1 = Broadly Relevant, 2 = High Interest, 3 = Primary Choice."
            )

        if candidate_eval.eligibility_status not in ELIGIBLE_CANDIDATE_STATUSES:
            raise ValueError(
                f"Cannot record relevance judgment for scheme '{candidate_eval.scheme_id}' with status "
                f"'{candidate_eval.eligibility_status}'. Disqualified schemes are kept outside the ML candidate set."
            )

        pair_key = (archetype.archetype_id, candidate_eval.scheme_id)
        if pair_key in self._seen_pairs:
            raise ValueError(
                f"Judgment already recorded for candidate pair: archetype '{archetype.archetype_id}', "
                f"scheme '{candidate_eval.scheme_id}' in this session."
            )

        # Extract features using existing RankingFeatureExtractor
        features = RankingFeatureExtractor.extract(archetype.profile, candidate_eval)

        clean_notes = justification.strip() if justification and justification.strip() else None

        record = RankingRecord(
            record_id=f"rec_{archetype.archetype_id}_{candidate_eval.scheme_id}_{self.annotator_id}",
            query_id=archetype.archetype_id,
            session_id=self.annotator_id,
            scheme_id=candidate_eval.scheme_id,
            scheme_name=candidate_eval.scheme_name,
            features=features,
            relevance_label=relevance_label,
            eligibility_status=candidate_eval.eligibility_status.value,
            provenance_source="expert_curation",
            notes=clean_notes,
        )

        self.records.append(record)
        self._seen_pairs.add(pair_key)
        return record

    def record_or_update_judgment(
        self,
        archetype: CitizenArchetype,
        candidate_eval: SchemeEvaluation,
        relevance_label: int,
        justification: Optional[str] = None,
    ) -> RankingRecord:
        """
        Record or update an expert relevance judgment for an eligible candidate scheme.
        If a judgment already exists for this (archetype_id, scheme_id) pair in the session,
        it is replaced cleanly. Guarantees no duplicate records exist in the session.

        Args:
            archetype: The citizen archetype being evaluated.
            candidate_eval: The SchemeEvaluation for the candidate scheme.
            relevance_label: Relevance grade y in {1, 2, 3}.
            justification: Optional textual rationale for the assigned grade.

        Returns:
            The created or updated RankingRecord.
        """
        if relevance_label not in VALID_EXPERT_RELEVANCE_GRADES:
            raise ValueError(
                f"Invalid relevance_label '{relevance_label}'. "
                f"Expert relevance grade for candidate schemes must be in {sorted(VALID_EXPERT_RELEVANCE_GRADES)}: "
                "1 = Broadly Relevant, 2 = High Interest, 3 = Primary Choice."
            )

        if candidate_eval.eligibility_status not in ELIGIBLE_CANDIDATE_STATUSES:
            raise ValueError(
                f"Cannot record relevance judgment for scheme '{candidate_eval.scheme_id}' with status "
                f"'{candidate_eval.eligibility_status}'. Disqualified schemes are kept outside the ML candidate set."
            )

        pair_key = (archetype.archetype_id, candidate_eval.scheme_id)
        if pair_key in self._seen_pairs:
            # Replace existing record to prevent duplicates
            self.records = [
                r for r in self.records
                if not (r.query_id == archetype.archetype_id and r.scheme_id == candidate_eval.scheme_id)
            ]
        else:
            self._seen_pairs.add(pair_key)

        features = RankingFeatureExtractor.extract(archetype.profile, candidate_eval)
        clean_notes = justification.strip() if justification and justification.strip() else None

        record = RankingRecord(
            record_id=f"rec_{archetype.archetype_id}_{candidate_eval.scheme_id}_{self.annotator_id}",
            query_id=archetype.archetype_id,
            session_id=self.annotator_id,
            scheme_id=candidate_eval.scheme_id,
            scheme_name=candidate_eval.scheme_name,
            features=features,
            relevance_label=relevance_label,
            eligibility_status=candidate_eval.eligibility_status.value,
            provenance_source="expert_curation",
            notes=clean_notes,
        )

        self.records.append(record)
        return record

    def build_dataset(self, dataset_name: Optional[str] = None) -> RankingDataset:
        """
        Build, finalize, and validate the RankingDataset model.
        """
        query_groups: Dict[str, int] = {}
        label_dist: Dict[str, int] = {}
        for r in self.records:
            query_groups[r.query_id] = query_groups.get(r.query_id, 0) + 1
            lbl_key = str(r.relevance_label)
            label_dist[lbl_key] = label_dist.get(lbl_key, 0) + 1

        name = dataset_name or f"SchemeIQ_Expert_Annotations_{self.annotator_id}"
        metadata = RankingDatasetMetadata(
            dataset_name=name,
            schema_version="1.0.0",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            purpose=DatasetPurpose.EXPERT_CURATED_BENCHMARK,
            total_records=len(self.records),
            total_groups=len(query_groups),
            label_distribution=label_dist,
            author_or_curator=f"Expert Annotator: {self.annotator_id}",
            disclaimer=(
                "Expert-curated benchmark dataset for SchemeIQ+ Learning-to-Rank. "
                "Relevance labels reflect human welfare expertise on pre-filtered eligible candidates."
            ),
        )

        dataset = RankingDataset(metadata=metadata, records=self.records)

        # Validate with existing Phase 7B validator
        report = RankingDatasetValidator.validate_dataset(dataset)
        if not report.is_valid:
            raise ValueError(f"Annotated dataset failed validation: {report.errors}")

        return dataset

    def save_annotations(self, output_path: Path, dataset_name: Optional[str] = None) -> Path:
        """
        Validate and save the completed annotation dataset to JSON.
        """
        dataset = self.build_dataset(dataset_name=dataset_name)
        return RankingDatasetIO.save_json(dataset, output_path)

    def generate_review_sheet(self, archetypes: List[CitizenArchetype]) -> Dict[str, Any]:
        """
        Generate a structured review sheet containing citizen archetypes and their
        strictly pre-filtered candidate schemes (ELIGIBLE / POTENTIALLY_ELIGIBLE only).
        Relevance grades and justifications are left blank for the expert to fill out.
        """
        candidate_reviews = []
        for arch in archetypes:
            candidates = self.get_candidate_schemes(arch)
            cand_list = []
            for cand in candidates:
                cand_list.append({
                    "scheme_id": cand.scheme_id,
                    "scheme_name": cand.scheme_name,
                    "category": cand.category,
                    "scheme_scope": cand.scheme_scope,
                    "eligibility_status": cand.eligibility_status.value,
                    "deterministic_score": round(cand.score, 2),
                    "matched_rules": [
                        {"field": mr.field, "reason": mr.reason}
                        for mr in cand.matched_rules
                    ],
                    "relevance_grade": None,  # To be assigned by expert: 1, 2, or 3
                    "justification": None,    # Optional rationale text
                })
            candidate_reviews.append({
                "archetype_id": arch.archetype_id,
                "archetype_name": arch.archetype_name,
                "narrative": arch.narrative,
                "demographics_summary": arch.profile.model_dump(exclude_none=True),
                "eligible_candidates_count": len(cand_list),
                "candidates": cand_list,
            })

        return {
            "metadata": {
                "annotator_id": self.annotator_id,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_archetypes": len(archetypes),
                "rubric": {
                    "1": "BROADLY RELEVANT: Baseline eligibility met; secondary/indirect benefit",
                    "2": "HIGH INTEREST: Strong alignment with immediate livelihood/welfare need",
                    "3": "PRIMARY CHOICE: Top priority; maximum direct socio-economic impact",
                },
                "instructions": (
                    "Review each eligible candidate scheme for the citizen archetype. "
                    "Assign 'relevance_grade' as an integer 1, 2, or 3. "
                    "Optionally add a 'justification' text string. "
                    "Do NOT alter archetype_id, scheme_id, or candidate metadata."
                ),
            },
            "candidate_reviews": candidate_reviews,
        }

    def export_review_sheet(self, archetypes: List[CitizenArchetype], output_path: Path) -> Path:
        """
        Export review sheet to a JSON file for offline annotation by an expert.
        """
        sheet = self.generate_review_sheet(archetypes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sheet, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported review sheet for {len(archetypes)} archetypes to {output_path}")
        return output_path

    def record_batch_from_review_sheet(
        self,
        archetypes: List[CitizenArchetype],
        review_sheet_path: Path,
    ) -> List[RankingRecord]:
        """
        Ingest completed expert judgments from an offline review sheet JSON file.
        Validates every judgment against candidate eligibility and allowed grades {1, 2, 3}.
        """
        if not review_sheet_path.exists():
            raise FileNotFoundError(f"Review sheet file not found: {review_sheet_path}")

        with open(review_sheet_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        reviews = data.get("candidate_reviews")
        if not isinstance(reviews, list):
            raise ValueError(f"Malformed review sheet in {review_sheet_path}: missing 'candidate_reviews' list.")

        arch_map = {a.archetype_id: a for a in archetypes}
        recorded = []

        for rev in reviews:
            arch_id = rev.get("archetype_id")
            if not arch_id or arch_id not in arch_map:
                raise ValueError(f"Unknown archetype_id '{arch_id}' in review sheet.")

            archetype = arch_map[arch_id]
            candidates = self.get_candidate_schemes(archetype)
            cand_map = {c.scheme_id: c for c in candidates}

            candidates_data = rev.get("candidates", [])
            for c_data in candidates_data:
                sid = c_data.get("scheme_id")
                grade = c_data.get("relevance_grade")
                justification = c_data.get("justification")

                if grade is None:
                    # Expert skipped this candidate
                    continue

                if not isinstance(grade, int):
                    try:
                        grade = int(grade)
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid relevance_grade '{grade}' for scheme '{sid}'. Must be an integer 1, 2, or 3.")

                if sid not in cand_map:
                    raise ValueError(
                        f"Scheme '{sid}' is not an eligible candidate for archetype '{arch_id}'. "
                        "Disqualified schemes cannot be annotated."
                    )

                rec = self.record_judgment(
                    archetype=archetype,
                    candidate_eval=cand_map[sid],
                    relevance_label=grade,
                    justification=justification,
                )
                recorded.append(rec)

        logger.info(f"Ingested {len(recorded)} expert judgments from {review_sheet_path}")
        return recorded

    def run_cli_session(self, archetypes: List[CitizenArchetype], output_path: Path) -> Path:
        """
        Interactive command-line interface for human experts to review archetypes,
        inspect deterministic evaluations, and assign relevance grades.
        """
        print("\n" + "=" * 75)
        print("  SchemeIQ+ Expert Relevance Annotation Tool (Phase 7E/7F)")
        print(f"  Annotator ID: {self.annotator_id}")
        print(f"  Total Archetypes: {len(archetypes)}")
        print("=" * 75)
        print("\nRelevance Grading Rubric:")
        print("  [1] BROADLY RELEVANT  — Baseline eligibility met; secondary/indirect benefit")
        print("  [2] HIGH INTEREST     — Strong alignment with immediate livelihood/welfare need")
        print("  [3] PRIMARY CHOICE    — Top priority; maximum direct socio-economic impact")
        print("  (Disqualified schemes with NOT_ELIGIBLE / NOT_APPLICABLE are automatically excluded)")

        for a_idx, arch in enumerate(archetypes, start=1):
            print("\n" + "#" * 75)
            print(f"ARCHETYPE {a_idx}/{len(archetypes)}: [{arch.archetype_id}] {arch.archetype_name}")
            if arch.narrative:
                print(f"Context: {arch.narrative}")
            print("-" * 50)
            print("Demographics & Status:")
            for k, v in arch.profile.model_dump(exclude_none=True).items():
                print(f"  • {k}: {v}")

            # Get pre-filtered candidate schemes
            candidates = self.get_candidate_schemes(arch)
            print("-" * 50)
            print(f"Deterministic Candidates ({len(candidates)} eligible/potentially eligible schemes):")

            if not candidates:
                print("  (No candidate schemes met eligibility requirements for this archetype)")
                continue

            for c_idx, cand in enumerate(candidates, start=1):
                print(f"\n  [{c_idx}/{len(candidates)}] Scheme: [{cand.scheme_id}] {cand.scheme_name} ({cand.scheme_scope})")
                print(f"      Category: {cand.category}")
                print(f"      Eligibility Status: {cand.eligibility_status.value}")
                print(f"      Deterministic Score: {cand.score:.1f}")
                if cand.matched_rules:
                    print(f"      Satisfied Rules ({len(cand.matched_rules)}):")
                    for mr in cand.matched_rules:
                        print(f"        ✓ {mr.field}: {mr.reason}")

                # Prompt expert for label
                while True:
                    try:
                        raw_input_label = input("      Assign Relevance Grade [1, 2, or 3] (or 's' to skip): ").strip()
                        if raw_input_label.lower() == "s":
                            print("      Skipped.")
                            break
                        val = int(raw_input_label)
                        if val in VALID_EXPERT_RELEVANCE_GRADES:
                            justification = input("      Optional justification note (press enter to skip): ").strip()
                            self.record_judgment(
                                archetype=arch,
                                candidate_eval=cand,
                                relevance_label=val,
                                justification=justification if justification else None,
                            )
                            print("      ✓ Judgment recorded.")
                            break
                        else:
                            print("      Invalid input. Must be 1, 2, or 3.")
                    except ValueError:
                        print("      Please enter an integer (1, 2, or 3) or 's' to skip.")

        saved_path = self.save_annotations(output_path)
        print("\n" + "=" * 75)
        print(f"Annotation session complete! Total judgments recorded: {len(self.records)}")
        print(f"Dataset successfully validated and saved to: {saved_path}")
        print("=" * 75)
        return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SchemeIQ+ Expert Relevance Annotation Tool (Phase 7F)")
    parser.add_argument("--input", required=True, help="Path to JSON file containing citizen archetype profiles")
    parser.add_argument("--annotator", required=True, help="Non-PII identifier for the expert annotator (e.g. 'expert_01')")
    parser.add_argument("--output", default="data/ranking/expert_annotations.json", help="Destination path for output dataset or review sheet")
    parser.add_argument("--generate-review-sheet", action="store_true", help="Generate an offline review sheet template with pre-filtered candidates")
    parser.add_argument("--ingest-review-sheet", help="Path to a completed offline review sheet JSON file to ingest")
    parser.add_argument("--web", action="store_true", help="Launch the local web-based annotation UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind the web server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the web server (default: 5000)")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.web:
        from src.recommendation.web_annotation import launch_web_annotation
        launch_web_annotation(
            archetypes_path=input_path,
            annotator_id=args.annotator,
            output_path=output_path,
            host=args.host,
            port=args.port,
        )
        return

    archetypes = load_archetypes_from_file(input_path)
    session = ExpertAnnotationSession(annotator_id=args.annotator)

    if args.generate_review_sheet:
        saved = session.export_review_sheet(archetypes=archetypes, output_path=output_path)
        print(f"Generated candidate review sheet at: {saved}")
        return

    if args.ingest_review_sheet:
        sheet_path = Path(args.ingest_review_sheet)
        session.record_batch_from_review_sheet(archetypes=archetypes, review_sheet_path=sheet_path)
        saved = session.save_annotations(output_path=output_path)
        print(f"Successfully ingested judgments and saved validated dataset to: {saved}")
        return

    # Default: Interactive CLI session
    session.run_cli_session(archetypes=archetypes, output_path=output_path)


if __name__ == "__main__":
    main()

