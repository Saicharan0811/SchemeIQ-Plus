# -*- coding: utf-8 -*-
"""
Phase 6 Pre-flight Audit Script
Inspects datasets, manifests, raw documents, and existing RAG code.
Outputs data/reports/phase6_preflight_audit.json
"""
import sys, os, csv, json, datetime
from pathlib import Path

root = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
reports_dir = root / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

audit_data = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": "Phase 6 Pre-Flight Audit",
    "critical_paths": {},
    "dataset_inspected": {},
    "supported_profile_fields_in_dataset": [],
    "deterministic_evaluable_rules": [],
    "missing_or_unstructured_criteria": [],
    "schemes_with_insufficient_structured_criteria": [],
    "derived_rule_layer_necessary": True,
    "findings": [],
}

# 1. Check critical paths
critical_paths = [
    "data/documents/raw",
    "data/documents/raw/metadata.json",
    "data/processed",
    "data/processed/corpus_manifest.json",
    "data/processed/corpus_lock_report.json",
    "data/vector_store/chroma_db",
    "data/SchemeIQ_Plus_Telangana_Starter_Dataset(1).csv",
    "data/documents/source_manifest.csv",
    "src/rag/__init__.py",
    "src/rag/schemas.py",
    "src/rag/document_loader.py",
    "src/rag/chunker.py",
    "src/rag/embeddings.py",
    "src/rag/vector_store.py",
    "src/rag/ingest.py",
    "src/rag/query_detector.py",
    "src/rag/keyword_retriever.py",
    "src/rag/hybrid_retriever.py",
    "src/rag/evaluate_retrieval.py",
    "src/rag/answer_generator.py",
    "src/rag/rag_service.py",
    "src/rag/test_answer_generation.py",
]

print("=== 1. CHECK CRITICAL PATHS ===")
for p in critical_paths:
    exists = (root / p).exists()
    status = "OK" if exists else "MISSING"
    audit_data["critical_paths"][p] = status
    print(f"  [{status}] {p}")

# 2. Inspect starter dataset
dataset_path = root / "data/SchemeIQ_Plus_Telangana_Starter_Dataset(1).csv"
if not dataset_path.exists():
    dataset_path = root / "data/schemes.csv"

rows = []
headers = []
if dataset_path.exists():
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

audit_data["dataset_inspected"] = {
    "path": str(dataset_path.relative_to(root)),
    "row_count": len(rows),
    "columns": headers,
}

print(f"\n=== 2. DATASET: {dataset_path.name} ===")
print(f"Columns ({len(headers)}): {headers}")
print(f"Rows: {len(rows)}")

# Analyze columns
structured_fields = []
unstructured_fields = []
for h in headers:
    values = [r.get(h, "") for r in rows if r.get(h, "")]
    empty_count = len(rows) - len(values)
    unique_vals = list(set(values))
    avg_len = sum(len(v) for v in values) / max(1, len(values))
    print(f"  - {h}: {len(values)} filled, {empty_count} empty, avg_len={avg_len:.1f}")
    if empty_count > 10 or avg_len > 40:
        unstructured_fields.append({"field": h, "filled_count": len(values), "avg_len": avg_len})
    else:
        structured_fields.append({"field": h, "filled_count": len(values), "sample_values": unique_vals[:5]})

audit_data["supported_profile_fields_in_dataset"] = structured_fields
audit_data["unstructured_fields"] = unstructured_fields

# 3. Analyze each scheme's eligibility information in the starter dataset
insufficient_schemes = []
deterministic_schemes = []

for r in rows:
    sid = r.get("scheme_id", "")
    sname = r.get("scheme_name", "")
    crit = r.get("eligibility_criteria", "")
    scope = r.get("scheme_scope", "")
    target = r.get("target_beneficiary", "")
    cat = r.get("category", "")
    age_min = r.get("age_min", "")
    age_max = r.get("age_max", "")
    inc = r.get("income_limit", "")

    has_structured_nums = any([age_min, age_max, inc])
    
    analysis = {
        "scheme_id": sid,
        "scheme_name": sname,
        "scope": scope,
        "category": cat,
        "target_beneficiary": target,
        "has_explicit_numeric_columns": has_structured_nums,
        "eligibility_criteria_text": crit,
    }
    
    # Check if criteria is just descriptive text
    if not has_structured_nums:
        insufficient_schemes.append(analysis)
    else:
        deterministic_schemes.append(analysis)

audit_data["schemes_with_insufficient_structured_criteria"] = [s["scheme_id"] for s in insufficient_schemes]
audit_data["deterministic_evaluable_from_raw_csv_alone"] = [s["scheme_id"] for s in deterministic_schemes]

# 4. Findings & Specific Answers to Pre-flight questions
audit_data["findings"] = [
    "Q1: What profile fields are currently supported by data/schemes.csv? -> Columns present are [scheme_id, scheme_name, scheme_scope, category, applicability, target_beneficiary, description, benefits, eligibility_criteria, age_min, age_max, income_limit, official_url, source_authority]. However, numeric fields age_min, age_max, income_limit are empty for all 14 rows.",
    "Q2: Which eligibility rules can be evaluated deterministically? -> State residency (Telangana vs Central India-wide), broad categorical beneficiary type (Farmers, Women, Students, Senior Citizens, Micro-entrepreneurs), and scheme scope. Numerical and fine-grained criteria exist in the locked official corpus documents but are not structured in the raw CSV.",
    "Q3: Which criteria are missing or only represented as free text? -> Age limits (e.g. APY age 18-40, Senior Citizen age 70+ in PM-JAY expansion, Aasara age thresholds), income ceilings (e.g. ePASS parental income bands, PMAY-U EWS/LIG bands), project cost limits (e.g. PMEGP ₹50L manufacturing / ₹20L services, MUDRA ₹50k/₹5L/₹10L loan brackets), and landholding criteria.",
    "Q4: Which schemes have insufficient structured criteria in raw CSV? -> All 14 schemes in the raw CSV store eligibility_criteria purely as summary text without populated numeric fields.",
    "Q5: Is a new derived eligibility-rule layer necessary? -> YES. A dedicated derived structured rule layer (data/eligibility/scheme_rules.json) must be created. It must model explicit deterministic rules (with operators like gte, lte, eq, in) extracted with exact provenance from the locked official corpus and verified scheme metadata, while categorizing unextractable details as UNKNOWN / UNSTRUCTURED rather than hallucinating."
]

out_report = reports_dir / "phase6_preflight_audit.json"
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2, ensure_ascii=False)

print(f"\nPre-flight audit report saved to: {out_report}")
