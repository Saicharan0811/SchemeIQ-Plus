# Phase 6: Source Recovery Report

## Inspection & Discovery Phase
Following the user instructions, I first inspected the current project state (August 22nd run).

**Status of Previous Recovery Attempts (Aug 15)**:
*   **Successes**: TS008 (Telangana Diagnostics) successfully downloaded a valid official sub-domain page (`tdiagnostics.telangana.gov.in`) containing 135 words of scheme-specific data, replacing the previous duplicate. 
*   **Failures / Low Relevance**:
    *   **TS002 (Rythu Bima)**: Downloaded `myscheme.gov.in/schemes/rbs`, but `requests` cannot render this Next.js SPA, resulting in a "Something went wrong" React error page (WARNING_LOW_RELEVANCE).
    *   **TS004 (Maha Lakshmi)**: Downloaded `myscheme.gov.in/schemes/mlsfbt`, which resulted in the same JS error page.
    *   **CT005 (PMEGP)**: Downloaded an MSME portal page, but it was a Next.js wrapper yielding 0 extracted words (WARNING_LOW_CONTENT).
    *   **TS001 (Rythu Bharosa)**: The candidate `agriculture.telangana.gov.in` timed out previously.
    *   **CT002 (PM-JAY)**: The old URL `pmjay.gov.in` timed out.

## Action Plan: New Official Candidates
To ensure robust text content for our RAG pipeline, we must avoid SPA wrappers and use direct HTML/PDFs.

| Scheme | Issue | New Official URL | Authority |
| :--- | :--- | :--- | :--- |
| **TS001** (Rythu Bharosa) | Previous candidates timed out / low content | `https://rythubharosa.telangana.gov.in/` | Official Scheme Portal |
| **TS002** (Rythu Bima) | myScheme JS wrapper failed | `https://rythubandhu.telangana.gov.in/` | Official Scheme Portal |
| **TS004** (Maha Lakshmi) | Duplicate / myScheme JS wrapper failed | `https://www.transport.telangana.gov.in/html/maha-lakshmi-scheme.html` | Telangana Transport Dept |
| **CT002** (PM-JAY) | Old URL timed out | `https://nha.gov.in/PM-JAY/about-pm-jay` | National Health Authority |
| **CT005** (PMEGP) | MSME page yielded 0 words (JS) | `https://dcmsme.gov.in/schemes/PMEGP_guidelines1.pdf` | DCMSME |

## Next Steps
1. The candidates are mapped out in `data/documents/recovery_candidates.csv`.
2. Await user approval before executing the recovery pipeline script.
3. Upon approval, I will download these 5 remaining sources and append them to `metadata.json` with the `recovery: true` flag to preserve project integrity.
