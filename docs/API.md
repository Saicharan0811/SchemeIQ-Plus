# SchemeIQ+ Production REST API Contract Documentation

**Version:** 1.0.0  
**Base URL:** `http://127.0.0.1:5000` (or configured host/port)  
**Protocol:** HTTP/1.1 (JSON)  
**CORS:** Enabled (`Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`)

---

## 1. Architectural Overview & Safety Invariants

The SchemeIQ+ API exposes deterministic statutory eligibility validation, advisory machine-learning re-ranking, and grounded RAG question answering for welfare schemes in Telangana and Central India.

### Core Invariants

1. **Deterministic Eligibility Authority**:
   - The deterministic rule engine (`SchemeEligibilityEvaluator`) is the **sole authority** for applicant eligibility status.
   - Candidates evaluated as `NOT_ELIGIBLE` or `NOT_APPLICABLE` are **never** presented as eligible recommendations.
2. **Advisory Machine Learning (XGBoost)**:
   - `enable_xgboost` is strictly opt-in (`false` by default).
   - When enabled, XGBoost **only** reorders candidates already marked `ELIGIBLE` or `POTENTIALLY_ELIGIBLE`. It never changes eligibility statuses or creates eligibility decisions.
   - Any runtime failure in XGBoost cleanly falls back to Phase 6 deterministic heuristic ordering without raising HTTP 500 errors.
3. **Strict Privacy Pre-Screening**:
   - Incoming payloads are scrutinized by `validate_profile_privacy()` before profile instantiation.
   - Prohibited identifiers (`aadhaar_number`, `pan_number`, `bank_account_number`, passwords, etc.) are immediately rejected with **HTTP 400 Bad Request** (`code="PRIVACY_VIOLATION"`).
4. **Grounded RAG Integrity**:
   - Natural language answers are generated strictly from verified official government orders (G.O.s) and notifications stored in ChromaDB, with complete source citations and official URLs.

---

## 2. Standard Error Response Format

All error responses adhere to a consistent, sanitized JSON contract:

```json
{
  "error": "Human-readable error description",
  "code": "MACHINE_READABLE_ERROR_CODE",
  "details": [ ... ]
}
```

### Common Error Codes

| HTTP Status | Code | Meaning |
|---|---|---|
| `400` | `INVALID_CONTENT_TYPE` | Request header must be `application/json`. |
| `400` | `MALFORMED_JSON` | Body cannot be parsed as valid JSON. |
| `400` | `PRIVACY_VIOLATION` | Prohibited sensitive PII (Aadhaar, PAN, bank account) detected. |
| `400` | `VALIDATION_ERROR` | Profile demographic or financial attributes violate bounds. |
| `400` | `INVALID_TOP_K` | `top_k` must be an integer between 1 and 14 (or 1 and 20 for RAG). |
| `400` | `INVALID_BOOLEAN_FLAG` | Boolean flag is not a strict boolean (`true` or `false`). |
| `400` | `EMPTY_QUERY` | Search query is missing or empty. |
| `400` | `QUERY_TOO_SHORT` | Search query is under 5 characters. |
| `404` | `SCHEME_NOT_FOUND` | Requested `scheme_id` does not match the 14 official schemes. |
| `405` | `METHOD_NOT_ALLOWED` | HTTP method not permitted on endpoint. |
| `500` | `INTERNAL_SERVER_ERROR` | Sanitized server error response. |

---

## 3. Endpoints

### 3.1. Health Check

#### `GET /health`
Returns service status, application version, official schemes count, and XGBoost model availability.

**Headers:**
```http
Accept: application/json
```

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "service": "SchemeIQ+ Production API",
  "version": "1.0.0",
  "timestamp": "2026-09-05T08:24:34.123456+00:00",
  "xgboost_model_available": true,
  "official_schemes_count": 14
}
```

---

### 3.2. Citizen Scheme Recommendation

#### `POST /api/recommend`
Evaluates citizen demographic and financial attributes against all 14 statutory schemes, computing transparent scores and ranked recommendations.

**Headers:**
```http
Content-Type: application/json
Accept: application/json
```

**Request Body Schema:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `profile` | `object` | Yes | — | Citizen profile attributes (see UserProfile fields below). |
| `top_k` | `integer` | No | `5` | Number of top recommendations to return (range: 1..14). |
| `query` | `string` | No | `null` | Optional natural language contextual query (e.g., "crop loan"). |
| `include_rag_explanation` | `boolean` | No | `false` | If `true`, fetches official RAG context for top recommendations. |
| `enable_xgboost` | `boolean` | No | `false` | If `true`, applies advisory XGBoost LTR re-ranking to eligible pool. |

**UserProfile Fields (`profile` object):**

*Demographics:*
- `age` (`integer`, 0..120): Age in years.
- `gender` (`string`): `'female'`, `'male'`, `'transgender'`, `'other'`.
- `state` (`string`): State of residence (e.g. `'Telangana'`).
- `district` (`string`, optional): District name.
- `residence_type` (`string`): `'rural'` or `'urban'`.
- `category` (`string`): `'SC'`, `'ST'`, `'BC'`, `'EBC'`, `'Minority'`, `'General'`, `'OBC'`.
- `annual_income` (`number`, $\ge 0$): Gross annual family income in INR.

*Occupation & Assets:*
- `occupation` (`string`): Primary occupation (e.g. `'farmer'`, `'student'`, `'business_owner'`).
- `farmer_status` (`boolean`): Actively engaged in agriculture.
- `land_ownership_status` (`boolean`): Owns cultivable land.
- `land_acres` (`number`, $\ge 0$): Cultivable landholding in acres.
- `student_status` (`boolean`): Currently enrolled student.
- `business_owner_status` (`boolean`): Owns micro/small enterprise.
- `project_cost` (`number`, $\ge 0$): Estimated project cost in INR.
- `loan_requirement` (`number`, $\ge 0$): Loan requirement in INR.
- `owns_pucca_house` (`boolean`): Owns pucca house anywhere in India.
- `is_bpl_or_white_ration_card` (`boolean`): Holds Food Security Card / BPL card.
- `pays_income_tax` (`boolean`): Active income tax payer.
- `has_savings_bank_account` (`boolean`): Active savings bank account holder.

**Example Request:**
```json
{
  "profile": {
    "age": 42,
    "gender": "male",
    "state": "Telangana",
    "district": "Nalgonda",
    "residence_type": "rural",
    "occupation": "farmer",
    "farmer_status": true,
    "land_ownership_status": true,
    "land_acres": 3.5,
    "annual_income": 140000,
    "pays_income_tax": false,
    "is_bpl_or_white_ration_card": true
  },
  "top_k": 3,
  "enable_xgboost": true,
  "include_rag_explanation": false
}
```

**Response (`200 OK`):**
```json
{
  "total_schemes_evaluated": 14,
  "evaluation_timestamp": "2026-09-05T08:24:34.567890+00:00",
  "disclaimer": "Preliminary Assessment Disclaimer: This eligibility assessment is strictly derived from the SchemeIQ+ verified official source dataset...",
  "status_counts": {
    "ELIGIBLE": 2,
    "POTENTIALLY_ELIGIBLE": 9,
    "NOT_ELIGIBLE": 3
  },
  "user_profile_summary": {
    "age": 42,
    "state": "Telangana",
    "occupation": "farmer",
    "annual_income": 140000.0,
    "is_farmer": true,
    "has_land": true,
    "has_bpl_card": true
  },
  "recommendations": [
    {
      "scheme_id": "TS001",
      "scheme_name": "Rythu Bharosa",
      "scheme_scope": "State",
      "category": "Agriculture",
      "eligibility_status": "ELIGIBLE",
      "confidence": "HIGH",
      "score": 141.0,
      "scoring_breakdown": {
        "base_status_score": 100.0,
        "rule_pass_ratio_boost": 15.0,
        "state_scope_match_boost": 10.0,
        "domain_specificity_boost": 16.0,
        "total_score": 141.0,
        "xgboost_score": 1.4285
      },
      "official_url": "https://rythubharosa.telangana.gov.in/",
      "source_authority": "Department of Agriculture, Government of Telangana",
      "matched_rules": [
        {
          "rule_id": "TS001_farmer_status",
          "field": "farmer_status",
          "operator": "eq",
          "status": "PASS",
          "user_value": true,
          "expected_value": true,
          "confidence": "HIGH",
          "reason": "Satisfies verified farmer condition"
        }
      ],
      "failed_rules": [],
      "unknown_rules": [],
      "missing_information": [],
      "unstructured_criteria": [],
      "explanation_summary": "Satisfies 2 verified criteria: farmer_status, land_ownership_status."
    }
  ]
}
```

---

### 3.3. Official Scheme Information Lookup

#### `GET /api/schemes/<scheme_id>`
Retrieves verified scheme metadata, statutory rule definitions, and official policy URLs directly from the verified dataset.

**Parameters:**
- `scheme_id` (path, string, required): Official scheme ID (e.g. `TS001`, `CT001`, `CT002`). Case-insensitive.

**Example Request:**
```http
GET /api/schemes/TS001 HTTP/1.1
Host: 127.0.0.1:5000
```

**Response (`200 OK`):**
```json
{
  "scheme_id": "TS001",
  "scheme_name": "Rythu Bharosa",
  "scheme_scope": "State",
  "category": "Agriculture",
  "official_url": "https://rythubharosa.telangana.gov.in/",
  "source_authority": "Department of Agriculture, Government of Telangana",
  "source_filename": "TS001_30_Rythu_Bharosa.html",
  "rules": [
    {
      "rule_id": "TS001_farmer_status",
      "field": "farmer_status",
      "operator": "eq",
      "expected_value": true,
      "required": true,
      "rule_type": "hard_filter"
    }
  ],
  "unstructured_criteria": []
}
```

**Response (`404 Not Found`):**
```json
{
  "error": "Scheme 'UNKNOWN' not found. Must be one of the 14 official schemes.",
  "code": "SCHEME_NOT_FOUND"
}
```

---

### 3.4. Grounded RAG Question Answering

#### `POST /api/ask`
Submits a natural language citizen inquiry and returns an answer grounded strictly in verified government source documents with citations.

**Headers:**
```http
Content-Type: application/json
```

**Request Body Schema:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Yes | — | Citizen inquiry (5..2000 characters). |
| `top_k` | `integer` | No | `5` | Maximum number of context chunks retrieved (range: 1..20). |

**Example Request:**
```json
{
  "query": "Who is eligible for Rythu Bharosa financial assistance?",
  "top_k": 3
}
```

**Response (`200 OK`):**
```json
{
  "question": "Who is eligible for Rythu Bharosa financial assistance?",
  "answer": "Under Rythu Bharosa, financial assistance is provided to eligible landholding farmers...",
  "detected_scheme_id": "TS001",
  "detected_scheme_name": "Rythu Bharosa",
  "grounding_status": "GROUNDED",
  "grounding_note": "",
  "retrieval_count": 3,
  "retrieved_scheme_ids": ["TS001"],
  "sources": [
    {
      "source_id": "1",
      "scheme_id": "TS001",
      "scheme_name": "Rythu Bharosa",
      "document_title": "Rythu Bharosa Guidelines",
      "source_filename": "TS001_30_Rythu_Bharosa.txt",
      "official_url": "https://rythubharosa.telangana.gov.in/",
      "chunk_id": "TS001_TS001_30_Rythu_Bharosa_001_cbcc1f38"
    }
  ],
  "generation_timestamp": "2026-09-05T08:24:35.123456+00:00",
  "context_chunks_used": 2
}
```

---

## 4. Frontend Integration Guide (React / TypeScript)

### Recommended Fetch Function Example

```typescript
// api.ts - SchemeIQ+ Client
const API_BASE = "http://127.0.0.1:5000";

export async function getRecommendations(profile: Record<string, any>, enableXGBoost = false) {
  const response = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile,
      top_k: 5,
      enable_xgboost: enableXGBoost,
      include_rag_explanation: false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function askQuestion(query: string) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 3 }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || `HTTP ${response.status}`);
  }

  return response.json();
}
```
