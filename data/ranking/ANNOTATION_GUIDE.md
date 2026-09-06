# SchemeIQ+ Expert Annotation Guide (Phase 7F)

This guide provides operational instructions for domain experts (social welfare researchers, policy analysts, caseworkers) to contribute real relevance judgments for the SchemeIQ+ Learning-to-Rank recommendation engine.

---

## 1. Safety & Architecture Rules

1. **Advisory Scope Only**: Relevance judgments $y \in \{1, 2, 3\}$ train an advisory re-ranking layer. They have **zero authority** to certify or alter statutory legal eligibility.
2. **Deterministic Pre-Filtering**: The system automatically runs `SchemeEligibilityEvaluator` first. Only `ELIGIBLE` and `POTENTIALLY_ELIGIBLE` schemes are presented for annotation. Disqualified schemes (`NOT_ELIGIBLE`, `NOT_APPLICABLE`) are automatically excluded from the candidate pool.
3. **Strict Privacy**: Do not enter or store any Personally Identifiable Information (PII) such as names, phone numbers, email addresses, Aadhaar numbers, PAN numbers, or bank account numbers.

---

## 2. Non-PII Annotator Identity

Experts must identify themselves using an anonymous, non-PII annotator ID.

- **Acceptable IDs**: `expert_01`, `welfare_officer_A`, `policy_researcher_12`, `reviewer_rural_03`
- **Prohibited IDs**: Email addresses (`john.doe@telangana.gov.in`), phone numbers (`9876543210`), or real personal names.

---

## 3. The 3-Tier Relevance Rubric

For each presented candidate scheme, assign an integer grade $y \in \{1, 2, 3\}$:

| Grade | Label | Description & Guidance |
|:---:|:---|:---|
| **`1`** | **Broadly Relevant** | Citizen meets baseline criteria, but the scheme provides only secondary, indirect, or marginal welfare value. |
| **`2`** | **High Interest** | Strong alignment with citizen's primary livelihood or welfare need; recommended for active consideration. |
| **`3`** | **Primary Choice** | Highest-priority intervention offering maximum direct economic/social impact for this specific archetype. |

*(Note: Grade `0` is reserved exclusively for disqualified or irrelevant schemes, which are pre-filtered out by the system).*

---

## 4. How to Run the Real Annotation Workflow

Two workflow options are supported:

### Option A: Interactive Command-Line Interface (CLI)

Best for annotating directly in a terminal session.

```bash
# Run interactive CLI session
python -m src.recommendation.annotation_tool \
  --input data/ranking/real_archetypes.json \
  --annotator expert_01 \
  --output data/ranking/expert_annotations_expert_01.json
```

**During the session:**
1. The tool displays the archetype profile narrative and demographics.
2. Evaluates all 14 official schemes and filters to only eligible candidates.
3. For each candidate, displays scheme details, satisfied rules, and score.
4. Prompts: `Assign Relevance Grade [1, 2, or 3] (or 's' to skip): `
5. Prompts: `Optional justification note (press enter to skip): `
6. Upon completion, validates dataset via `RankingDatasetValidator` and saves to the output path.

---

### Option B: Offline Review Sheet Workflow (Recommended for Teams)

Best for experts who prefer reviewing candidates in a structured document or JSON file.

#### Step 1: Generate Review Sheet Template
```bash
python -m src.recommendation.annotation_tool \
  --generate-review-sheet \
  --input data/ranking/real_archetypes.json \
  --annotator expert_01 \
  --output data/ranking/review_sheet_expert_01.json
```
This inspects the archetypes, extracts all eligible candidates, and produces a structured JSON form with `"relevance_grade": null` and `"justification": null`.

#### Step 2: Fill Out the Review Sheet
Open `review_sheet_expert_01.json` in any text editor and populate the blank fields:
```json
{
  "scheme_id": "TS001",
  "scheme_name": "Rythu Bharosa",
  "category": "Agriculture",
  "eligibility_status": "POTENTIALLY_ELIGIBLE",
  "deterministic_score": 111.0,
  "relevance_grade": 3,
  "justification": "Primary crop investment support for landowning farmer."
}
```

#### Step 3: Ingest Completed Review Sheet
```bash
python -m src.recommendation.annotation_tool \
  --ingest-review-sheet data/ranking/review_sheet_expert_01.json \
  --input data/ranking/real_archetypes.json \
  --annotator expert_01 \
  --output data/ranking/expert_annotations_expert_01.json
```
The tool validates that:
- Every annotated scheme was indeed an eligible candidate.
- All grades are integers strictly in $\{1, 2, 3\}$.
- No prohibited keywords or PII exist.
- The complete dataset passes `RankingDatasetValidator`.

---

## 5. Dataset Storage Location & Schema

- **Location**: `data/ranking/expert_annotations_<annotator_id>.json`
- **Schema**: Conforms to Phase 7B `RankingDataset` with metadata purpose set to `EXPERT_CURATED_BENCHMARK`.
- **Validation**: All datasets are automatically checked with `RankingDatasetValidator` before saving.

---

## 6. Current Collection Status & Phase 7G Readiness

- **Current Real Annotations**: **0** (No synthetic or manufactured data is used).
- **Sufficiency for XGBoost (Phase 7G)**: **Not yet sufficient**. Real annotations from domain experts are required before training can begin.
- **Recommended Threshold for Phase 7G**: At least 50–100 archetype query groups ($500+$ candidate ratings) across diverse demographic segments in Telangana.
