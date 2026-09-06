# -*- coding: utf-8 -*-
"""
query_detector.py — Deterministic scheme detection from query text.

Uses a curated alias map to identify scheme_id from natural language queries
without relying on LLM inference. Returns None when no scheme is detected,
allowing the hybrid retriever to perform unconstrained retrieval.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Scheme alias map — canonical scheme_id → list of aliases (lowercased)
# ---------------------------------------------------------------------------
SCHEME_ALIASES: dict[str, list[str]] = {
    "TS001": [
        "rythu bharosa", "rythu bharasa", "rythu", "rhb",
        "farmers investment support telangana",
    ],
    "TS002": [
        "rythu bima", "rythu bhima", "rythu bheema",
        "farmer life insurance telangana", "rytu bima",
    ],
    "TS003": [
        "cheyutha", "aarogyasri", "arogyasri telangana",
        "telangana aarogyasri", "cheyuta",
    ],
    "TS004": [
        "maha lakshmi", "mahalakshmi", "maha laxmi", "mahalaxmi",
        "telangana mahalakshmi", "free bus pass women",
    ],
    "TS005": [
        "aasara", "aasara pension", "telangana aasara",
        "aasara pensions", "social security pension telangana",
        "widow pension telangana", "old age pension telangana",
    ],
    "TS006": [
        "epass", "e-pass", "e pass", "telangana epass",
        "epass scholarships", "telangana scholarships", "ts epass",
    ],
    "TS007": [
        "mch kit", "mch kit scheme", "kcr kit", "kcr kit scheme",
        "maternal child health kit", "mother child health kit",
        "maternity kit telangana", "newborn kit telangana",
        "mch", "chfw mch",
    ],
    "TS008": [
        "telangana diagnostics", "free diagnostic services telangana",
        "tdiagnostics", "free diagnostics telangana",
        "diagnostics telangana", "ts diagnostics",
    ],
    "CT001": [
        "pm-kisan", "pm kisan", "kisan samman nidhi",
        "pradhan mantri kisan samman nidhi", "pmksn", "pm kisan nidhi",
    ],
    "CT002": [
        "pm-jay", "pmjay", "ayushman bharat", "ab pm-jay",
        "ayushman bharat pradhan mantri jan arogya yojana",
        "pradhan mantri jan arogya yojana", "jan arogya yojana", "ab pmjay",
    ],
    "CT003": [
        "pmay", "pmay-u", "pmay urban", "pradhan mantri awas yojana",
        "pradhan mantri awas yojana urban", "pm awas yojana",
        "housing for all urban", "pmay u 2.0",
    ],
    "CT004": [
        "mudra", "mudra loan", "pradhan mantri mudra yojana",
        "pmmy", "mudra yojana", "shishu loan", "kishore loan", "tarun loan",
    ],
    "CT005": [
        "pmegp", "prime minister employment generation programme",
        "pm employment generation", "employment generation programme",
        "pmegp loan", "khadi village industries pmegp",
    ],
    "CT006": [
        "atal pension yojana", "apy", "atal pension",
        "nps atal pension", "pradhan mantri pension yojana",
    ],
}

# Pre-compile patterns for efficiency — longest match first to avoid partial matches
_COMPILED: list[tuple[str, list[re.Pattern]]] = []

def _compile_patterns() -> None:
    """Build compiled regex patterns sorted by alias length (longest first)."""
    global _COMPILED
    _COMPILED = []
    for scheme_id, aliases in SCHEME_ALIASES.items():
        # Sort aliases by length descending so longer ones match first
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        patterns = [
            re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            for alias in sorted_aliases
        ]
        _COMPILED.append((scheme_id, patterns))


_compile_patterns()


def detect_scheme(query: str) -> Optional[str]:
    """
    Detect a scheme_id from the query string using deterministic alias matching.

    Returns the first matching scheme_id (by alias length priority),
    or None if no scheme is detected.

    Args:
        query: Natural language query string.

    Returns:
        scheme_id string (e.g. "TS007") or None.
    """
    if not query or not query.strip():
        return None

    query_stripped = query.strip()

    # Check each scheme; aliases are already sorted longest-first
    for scheme_id, patterns in _COMPILED:
        for pattern in patterns:
            if pattern.search(query_stripped):
                return scheme_id

    return None


def detect_scheme_with_details(query: str) -> dict:
    """
    Like detect_scheme() but returns a dict with full detection details.
    Useful for logging and diagnostics.

    Returns:
        {
            "detected": bool,
            "scheme_id": str | None,
            "matched_alias": str | None,
        }
    """
    if not query or not query.strip():
        return {"detected": False, "scheme_id": None, "matched_alias": None}

    query_stripped = query.strip()

    for scheme_id, patterns in _COMPILED:
        # Rebuild aliases for this scheme to report matched text
        aliases = SCHEME_ALIASES[scheme_id]
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        for alias, pattern in zip(sorted_aliases, patterns):
            m = pattern.search(query_stripped)
            if m:
                return {
                    "detected": True,
                    "scheme_id": scheme_id,
                    "matched_alias": alias,
                }

    return {"detected": False, "scheme_id": None, "matched_alias": None}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        ("Who is eligible for Rythu Bharosa?", "TS001"),
        ("What health coverage is provided under Ayushman Bharat PM-JAY?", "CT002"),
        ("What is the maximum project cost under PMEGP?", "CT005"),
        ("What subsidy is available under PMEGP?", "CT005"),
        ("What are the loan categories under MUDRA?", "CT004"),
        ("Who can apply for Telangana ePASS scholarships?", "TS006"),
        ("What benefits are provided under the MCH Kit Scheme?", "TS007"),
        ("What pension is available under Aasara?", "TS005"),
        ("Tell me about housing schemes", None),    # Should return None
    ]

    print(f"{'Query':<60} {'Expected':<10} {'Got':<10} {'OK'}")
    print("-" * 90)
    passed = 0
    for qtext, expected in test_queries:
        result = detect_scheme(qtext)
        ok = result == expected
        if ok:
            passed += 1
        print(f"{qtext[:58]:<60} {str(expected):<10} {str(result):<10} {'OK' if ok else 'FAIL'}")

    print(f"\nPassed: {passed}/{len(test_queries)}")
