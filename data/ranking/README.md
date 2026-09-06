# SchemeIQ+ Learning-to-Rank (LTR) Training Data Pipeline (Phase 7B)

This directory contains the dataset schema specifications, feature definitions, and non-training demonstration artifacts for the future XGBoost Learning-to-Rank ranking model.

---

## 1. Important Safety & Architectural Principle

> **WARNING: NON-TRAINING FOUNDATION ONLY**  
> XGBoost is **NOT** trained, saved, or active in this phase.  
> Machine learning models have **ZERO authority** to determine statutory government eligibility.  
> The verified deterministic rule engine (`src/eligibility/`) remains the sole authority for eligibility decisions (`ELIGIBLE`, `POTENTIALLY_ELIGIBLE`, `NOT_ELIGIBLE`).  
> XGBoost is designed solely as an **advisory ranking optimizer** to re-order candidate schemes that have already passed deterministic filtering.

---

## 2. Dataset Schema Architecture

Each entry in an LTR dataset is a `RankingRecord` representing an evaluation pair:
$$\text{Record} = (\text{query\_id}, \text{session\_id}, \text{scheme\_id}, \mathbf{x}_{\text{features}}, y_{\text{relevance}}, \text{provenance})$$

### Core Schema Fields

| Field | Type | Description |
|---|---|---|
| `record_id` | `str` | Unique record identifier (e.g. `rec_q001_TS001`) |
| `query_id` | `str` | Group/Query identifier linking all candidate schemes evaluated for a single user interaction |
| `session_id` | `Optional[str]` | Anonymous session UUID (no PII) |
| `scheme_id` | `str` | Verified scheme ID (`CT001`–`CT006`, `TS001`–`TS008`) |
| `scheme_name` | `str` | Official scheme name |
| `features` | `RankingFeatureVector` | Tabular numeric and encoded categorical features |
| `relevance_label` | `int` | Relevance grade: `0` (Not relevant), `1` (Broadly relevant), `2` (High interest), `3` (Primary choice) |
| `eligibility_status`| `str` | Deterministic status from `SchemeEligibilityEvaluator` (`ELIGIBLE`, `POTENTIALLY_ELIGIBLE`, etc.) |
| `provenance_source` | `str` | `expert_curation`, `user_interaction`, or `synthetic_validation_example` |

---

## 3. How Real Data Will Populate This Dataset in Future Phases

To maintain academic and production integrity, **no synthetic training labels have been fabricated**. In future phases, real training records will be collected through two non-synthetic protocols:

### Protocol A: Expert Curation (Cold-Start Ground Truth)
1. Welfare domain experts and policy researchers review $N$ standardized, anonymized citizen profile archetypes (e.g. smallholder tenant farmer in Mahabubabad, low-income college student in Warangal, woman micro-entrepreneur in Wanaparthy).
2. For each archetype, experts review the candidate schemes returned by the deterministic engine and assign relevance grades ($0$ to $3$) based on marginal economic impact and citizen urgency.
3. Records are exported via `RankingDatasetIO.save_json()` with `purpose="EXPERT_CURATED_BENCHMARK"`.

### Protocol B: Anonymized Citizen Interaction Feedback (Phase 7A+ Logs)
1. When users interact with the SchemeIQ+ UI, candidate schemes are displayed.
2. Anonymized, privacy-preserving click signals are captured:
   - Scheme shown but ignored $\rightarrow y = 1$ (Impression)
   - User clicks "Learn More" / views official guidelines $\rightarrow y = 2$ (Targeted Interest)
   - User clicks official portal application link $\rightarrow y = 3$ (Primary Choice / Action)
3. Sessions are aggregated by `query_id` (a random UUID), scrubbed via `validate_profile_privacy()`, and appended to the training corpus with `provenance_source="user_interaction"`.

---

## 4. How Phase 7C Will Use This Dataset

In Phase 7C (Model Training & Benchmarking), the dataset will be consumed as follows:
1. `RankingDatasetIO.load_json()` loads and validates records using `RankingDatasetValidator`.
2. `RankingDatasetIO.to_xgboost_format(dataset)` extracts:
   - `X`: Tabular feature dictionary matrix
   - `y`: Relevance grade targets ($0..3$)
   - `qid`: Query group identifiers
   - `group_counts`: Array of group lengths passed to `xgb.DMatrix(..., group=group_counts)`
3. Model is trained using `objective="rank:ndcg"` or `objective="rank:pairwise"`.
4. Evaluated using Grouped $K$-Fold Cross-Validation on NDCG@3, NDCG@5, and MAP.
