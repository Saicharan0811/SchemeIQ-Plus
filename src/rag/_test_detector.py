import sys
sys.path.insert(0, "C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
from src.rag.query_detector import detect_scheme_with_details

tests = [
    ("Who is eligible for Rythu Bharosa?", "TS001"),
    ("What health coverage is provided under Ayushman Bharat PM-JAY?", "CT002"),
    ("What is the maximum project cost under PMEGP?", "CT005"),
    ("What subsidy is available under PMEGP?", "CT005"),
    ("What are the loan categories under MUDRA?", "CT004"),
    ("Who can apply for Telangana ePASS scholarships?", "TS006"),
    ("What benefits are provided under the MCH Kit Scheme?", "TS007"),
    ("What pension is available under Aasara?", "TS005"),
    ("Tell me about housing for all urban", "CT003"),
]

passed = 0
for q, exp in tests:
    d = detect_scheme_with_details(q)
    got = d["scheme_id"]
    ok = got == exp
    if ok:
        passed += 1
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {q[:55]:<55} exp={exp} got={got} alias={d['matched_alias']}")
print(f"\nPassed: {passed}/{len(tests)}")
