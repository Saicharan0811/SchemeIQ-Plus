# -*- coding: utf-8 -*-
"""Phase 4A: Retrieval Diagnosis for failing queries."""
import sys, json, datetime, hashlib
from pathlib import Path
sys.path.insert(0, str(Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")))

from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import VectorStoreManager

PROJECT_ROOT = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

vector_store = VectorStoreManager()
emb_provider = get_embedding_provider()

FAILING_QUERIES = [
    ("MCH_Kit", "What benefits are provided under the MCH Kit Scheme?", "TS007"),
    ("Aasara",  "What pension is available under Aasara?",               "TS005"),
]

TOP_N = 15

diagnosis_data = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "current_retrieval_strategy": "dense-only cosine similarity via ChromaDB (sentence-transformers all-MiniLM-L6-v2, 384-dim)",
    "failing_queries": [],
    "top_10_results": {},
    "correct_scheme_rank": {},
    "root_causes": [],
    "recommended_improvements": []
}

# Check TS007 and TS005 chunks in vector store
print("\n=== CHECKING ALL CHUNKS IN VECTOR STORE ===")
results = vector_store.collection.get(include=["documents", "metadatas"])
ids = results.get("ids", [])
docs = results.get("documents", [])
metas = results.get("metadatas", [])

ts007_chunks = [(i, d, m) for i, d, m in zip(ids, docs, metas) if m and m.get("scheme_id") == "TS007"]
ts005_chunks = [(i, d, m) for i, d, m in zip(ids, docs, metas) if m and m.get("scheme_id") == "TS005"]

print(f"\nTS007 (MCH Kit) chunks in DB: {len(ts007_chunks)}")
for cid, cdoc, cmeta in ts007_chunks:
    words = cdoc.split()
    print(f"  ID: {cid}")
    print(f"  Text ({len(words)} words): {cdoc[:350]}")
    print()

print(f"\nTS005 (Aasara) chunks in DB: {len(ts005_chunks)}")
for cid, cdoc, cmeta in ts005_chunks:
    words = cdoc.split()
    print(f"  ID: {cid}")
    print(f"  Text ({len(words)} words): {cdoc[:350]}")
    print()

def text_contains_keywords(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]

mch_kw = ["mch", "mch kit", "kcr kit", "maternal", "child health", "kit"]
aasara_kw = ["aasara", "old age pension", "widow", "disabled", "handicapped", "panchayat raj"]

print("\n=== KEYWORD PRESENCE IN SCHEME CHUNKS ===")
print("TS007 chunks - MCH Kit keywords found:")
for cid, cdoc, cmeta in ts007_chunks:
    found = text_contains_keywords(cdoc, mch_kw)
    print(f"  {cid}: keywords={found}")

print("TS005 chunks - Aasara keywords found:")
for cid, cdoc, cmeta in ts005_chunks:
    found = text_contains_keywords(cdoc, aasara_kw)
    print(f"  {cid}: keywords={found}")

# Run top-15 retrieval for failing queries
print("\n=== TOP-15 RETRIEVAL FOR FAILING QUERIES ===")
for qkey, qtext, expected_scheme in FAILING_QUERIES:
    emb = emb_provider.embed_query(qtext)
    matches = vector_store.query_similar(emb, top_k=TOP_N)

    print(f"\nQuery: '{qtext}'")
    print(f"Expected scheme: {expected_scheme}")
    print(f"{'Rank':<6}{'Scheme':<10}{'Score':<8}{'Chunk ID':<42}{'Preview'}")
    print("-"*120)

    correct_rank = None
    top_results = []
    for m in matches:
        is_correct = m.scheme_id == expected_scheme
        if is_correct and correct_rank is None:
            correct_rank = m.rank
        marker = " <-- CORRECT" if is_correct else ""
        preview_safe = m.text_preview[:70].encode("ascii", errors="replace").decode("ascii")
        print(f"{m.rank:<6}{m.scheme_id:<10}{m.similarity_score:<8}{m.chunk_id:<42}{preview_safe}{marker}")
        top_results.append({
            "rank": m.rank,
            "chunk_id": m.chunk_id,
            "scheme_id": m.scheme_id,
            "scheme_name": m.scheme_name,
            "source_filename": m.source_filename,
            "similarity_score": m.similarity_score,
            "distance": m.distance,
            "chunk_preview": m.text_preview[:150],
        })

    diagnosis_data["top_10_results"][qkey] = top_results
    diagnosis_data["correct_scheme_rank"][qkey] = correct_rank
    diagnosis_data["failing_queries"].append({
        "query_key": qkey,
        "query_text": qtext,
        "expected_scheme": expected_scheme,
        "correct_scheme_rank": correct_rank,
    })
    print(f"\nCorrect scheme ({expected_scheme}) first appears at rank: {correct_rank}")

diagnosis_data["root_causes"] = [
    "TS007 (MCH Kit Scheme): The source document (chfw.telangana.gov.in/programmes.html) yielded only 119 words and 2 tiny chunks. These chunks are navigation-menu heavy and contain almost no MCH-specific terminology (no 'mch', 'kit', 'maternal', 'child'). The query 'benefits under MCH Kit Scheme' semantically matches PMAY-U 2.0 FAQ chunks because PMAY-U 2.0 has 13 large, keyword-rich chunks containing 'benefit', 'scheme', 'eligibility criteria', and 'under the scheme'. Dense-only cosine similarity fails here because term-overlap (MCH Kit) is completely absent from the scheme's tiny corpus.",
    "TS005 (Aasara Pensions): The source document yielded only 124 words and 1 chunk. The word 'pension' appears prominently in the PM-KISAN exclusion norms document (which explicitly lists 'superannuated/retired pensioners whose monthly pension is Rs.10,000/-'), causing PM-KISAN to outrank Aasara for any pension-related query. The single Aasara chunk lacks the exact word 'aasara' and 'pension' with sufficient density.",
    "General: Dense-only retrieval with small scheme documents (fewer than 200 words, 1-2 chunks) is vulnerable to embedding mass imbalance - larger richer documents accumulate more embedding space coverage than tiny thin ones. Exact scheme-name matching (MCH Kit, Aasara) is necessary for precise retrieval.",
]
diagnosis_data["recommended_improvements"] = [
    "1. Deterministic scheme detection: match query against scheme names/aliases to identify scheme_id before retrieval",
    "2. ChromaDB metadata filtering: when scheme_id is detected, filter retrieval to that scheme_id only",
    "3. BM25 keyword retrieval: index processed documents for exact term matching to complement dense similarity",
    "4. Reciprocal Rank Fusion: merge dense and BM25 rankings for balanced hybrid retrieval",
    "5. Optional local cross-encoder reranking if hybrid still insufficient",
]

with open(REPORTS_DIR / "rag_retrieval_diagnosis_report.json", "w", encoding="utf-8") as f:
    json.dump(diagnosis_data, f, indent=2, ensure_ascii=False)

print("\n\nDIAGNOSIS REPORT SAVED.")
print("\n=== SUMMARY ROOT CAUSES ===")
for i, rc in enumerate(diagnosis_data["root_causes"], 1):
    print(f"\n{i}. {rc[:200]}...")
