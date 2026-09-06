# SchemeIQ+ Derived Eligibility Rules Layer

This directory contains the structured, explainable eligibility rules derived from the locked SchemeIQ+ official government corpus and verified scheme metadata.

## Files
- `scheme_rules.json`: Complete 14-scheme rule database specifying field names, operators, expected values, required flags, rule types, confidence levels, and exact source provenance (filename, official URL, excerpt).
- `eligibility_schema.json`: JSON Schema validating the structure of `scheme_rules.json`.

## Grounding & Provenance Principles
1. **Zero Rule Hallucination**: Every rule maps directly to official source documents in `data/processed/documents/` or verified government portal metadata.
2. **Explicit Unstructured Criteria**: Criteria that rely on local departmental appraisal, biometric eKYC, or discretionary verification are explicitly cataloged under `unstructured_criteria` rather than forced into synthetic deterministic thresholds.
3. **Data Minimization**: Rule fields only model attributes necessary for preliminary entitlement screening.
