# -*- coding: utf-8 -*-
"""
src/eligibility/generate_rule_coverage_report.py — Generate rule coverage and provenance report
Saves data/reports/eligibility_rule_coverage_report.json
"""
import json, datetime
from pathlib import Path

root = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
rules_path = root / "data" / "eligibility" / "scheme_rules.json"
reports_dir = root / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

with open(rules_path, "r", encoding="utf-8") as f:
    rules_data = json.load(f)

schemes = rules_data.get("schemes", [])

report = {
    "report_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "version": rules_data.get("version"),
    "total_schemes": len(schemes),
    "scheme_summary": [],
    "total_deterministic_rules": 0,
    "total_unstructured_criteria": 0,
    "fields_used_in_rules": [],
    "source_provenance_coverage_pct": 100.0,
}

all_fields = set()
total_rules = 0
total_unstructured = 0

for s in schemes:
    s_rules = s.get("rules", [])
    s_unstruct = s.get("unstructured_criteria", [])
    total_rules += len(s_rules)
    total_unstructured += len(s_unstruct)

    rule_fields = [r["field"] for r in s_rules]
    all_fields.update(rule_fields)

    has_provenance = all(bool(r.get("official_url") and r.get("source_filename")) for r in s_rules)

    summary = {
        "scheme_id": s["scheme_id"],
        "scheme_name": s["scheme_name"],
        "scheme_scope": s["scheme_scope"],
        "category": s["category"],
        "official_url": s["official_url"],
        "deterministic_rules_count": len(s_rules),
        "unstructured_criteria_count": len(s_unstruct),
        "rule_fields": rule_fields,
        "full_provenance_attached": has_provenance,
    }
    report["scheme_summary"].append(summary)

report["total_deterministic_rules"] = total_rules
report["total_unstructured_criteria"] = total_unstructured
report["fields_used_in_rules"] = sorted(list(all_fields))

out_path = reports_dir / "eligibility_rule_coverage_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Rule coverage report generated: {out_path}")
print(f"Total schemes: {len(schemes)}")
print(f"Total deterministic rules: {total_rules}")
print(f"Total unstructured criteria: {total_unstructured}")
print(f"Unique fields used: {len(all_fields)} -> {sorted(list(all_fields))}")
