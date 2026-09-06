# -*- coding: utf-8 -*-
"""
evaluate_retrieval.py — Phase 4C: Evaluation benchmark for hybrid retrieval.

Runs 8 standardised benchmark queries through both the old dense-only
retriever and the new hybrid retriever, then reports:
    - Top-1 Accuracy (correct scheme at rank 1)
    - Top-3 Recall (correct scheme appears in top 3)
    - Mean Reciprocal Rank (MRR)
    - Per-query breakdown

Writes results to data/reports/rag_retrieval_evaluation_report.json.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")))
sys.stdout.reconfigure(encoding="utf-8")

from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import VectorStoreManager
from src.rag.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path("C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus")
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark query set (8 queries — same as Phase 3 smoke tests)
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES: list[dict] = [
    {
        "query_id": "Q1",
        "query": "Who is eligible for Rythu Bharosa?",
        "expected_scheme_id": "TS001",
        "expected_scheme_name": "Rythu Bharosa",
    },
    {
        "query_id": "Q2",
        "query": "What health coverage is provided under Ayushman Bharat PM-JAY?",
        "expected_scheme_id": "CT002",
        "expected_scheme_name": "Ayushman Bharat PM-JAY",
    },
    {
        "query_id": "Q3",
        "query": "What is the maximum project cost under PMEGP?",
        "expected_scheme_id": "CT005",
        "expected_scheme_name": "PMEGP",
    },
    {
        "query_id": "Q4",
        "query": "What subsidy is available under PMEGP?",
        "expected_scheme_id": "CT005",
        "expected_scheme_name": "PMEGP",
    },
    {
        "query_id": "Q5",
        "query": "What are the loan categories under MUDRA?",
        "expected_scheme_id": "CT004",
        "expected_scheme_name": "MUDRA",
    },
    {
        "query_id": "Q6",
        "query": "Who can apply for Telangana ePASS scholarships?",
        "expected_scheme_id": "TS006",
        "expected_scheme_name": "Telangana ePASS",
    },
    {
        "query_id": "Q7",
        "query": "What benefits are provided under the MCH Kit Scheme?",
        "expected_scheme_id": "TS007",
        "expected_scheme_name": "MCH Kit Scheme",
    },
    {
        "query_id": "Q8",
        "query": "What pension is available under Aasara?",
        "expected_scheme_id": "TS005",
        "expected_scheme_name": "Aasara Pensions",
    },
]

TOP_K_EVAL = 10   # check rank within top-10


# ---------------------------------------------------------------------------
# Dense-only baseline evaluation
# ---------------------------------------------------------------------------
def run_dense_baseline(
    queries: list[dict],
    vector_store: VectorStoreManager,
    emb_provider,
    top_k: int = TOP_K_EVAL,
) -> list[dict]:
    results = []
    for q in queries:
        emb = emb_provider.embed_query(q["query"])
        matches = vector_store.query_similar(emb, top_k=top_k)

        top1_scheme = matches[0].scheme_id if matches else None
        correct_rank = next(
            (m.rank for m in matches if m.scheme_id == q["expected_scheme_id"]), None
        )
        top3_recall = any(
            m.scheme_id == q["expected_scheme_id"] for m in matches[:3]
        )
        mrr = (1.0 / correct_rank) if correct_rank else 0.0

        top_results = [
            {
                "rank": m.rank,
                "scheme_id": m.scheme_id,
                "scheme_name": m.scheme_name,
                "similarity_score": m.similarity_score,
                "chunk_id": m.chunk_id,
                "text_preview": m.text_preview[:150],
            }
            for m in matches[:5]
        ]

        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "expected_scheme_id": q["expected_scheme_id"],
            "top1_scheme_id": top1_scheme,
            "top1_correct": top1_scheme == q["expected_scheme_id"],
            "top3_recall": top3_recall,
            "correct_rank": correct_rank,
            "mrr": round(mrr, 4),
            "top5_results": top_results,
        })

    return results


# ---------------------------------------------------------------------------
# Hybrid retrieval evaluation
# ---------------------------------------------------------------------------
def run_hybrid_eval(
    queries: list[dict],
    retriever: HybridRetriever,
    top_k: int = TOP_K_EVAL,
) -> list[dict]:
    results = []
    for q in queries:
        matches = retriever.retrieve(q["query"], final_top_k=top_k)

        top1_scheme = matches[0].scheme_id if matches else None
        correct_rank = next(
            (m.rank for m in matches if m.scheme_id == q["expected_scheme_id"]), None
        )
        top3_recall = any(
            m.scheme_id == q["expected_scheme_id"] for m in matches[:3]
        )
        mrr = (1.0 / correct_rank) if correct_rank else 0.0
        detected = matches[0].detection if matches else {}

        top_results = [
            {
                "rank": m.rank,
                "scheme_id": m.scheme_id,
                "scheme_name": m.scheme_name,
                "rrf_score": round(m.rrf_score, 6),
                "dense_rank": m.dense_rank,
                "keyword_rank": m.keyword_rank,
                "dense_score": round(m.dense_score, 4) if m.dense_score else None,
                "bm25_score": round(m.bm25_score, 4) if m.bm25_score else None,
                "chunk_id": m.chunk_id,
                "text_preview": m.text_preview[:150],
            }
            for m in matches[:5]
        ]

        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "expected_scheme_id": q["expected_scheme_id"],
            "top1_scheme_id": top1_scheme,
            "top1_correct": top1_scheme == q["expected_scheme_id"],
            "top3_recall": top3_recall,
            "correct_rank": correct_rank,
            "mrr": round(mrr, 4),
            "scheme_detection": detected,
            "top5_results": top_results,
        })

    return results


# ---------------------------------------------------------------------------
# Compute aggregate metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    top1_acc = sum(1 for r in results if r["top1_correct"]) / n
    top3_recall = sum(1 for r in results if r["top3_recall"]) / n
    mrr = sum(r["mrr"] for r in results) / n
    return {
        "top1_accuracy": round(top1_acc, 4),
        "top1_count": sum(1 for r in results if r["top1_correct"]),
        "top3_recall": round(top3_recall, 4),
        "top3_count": sum(1 for r in results if r["top3_recall"]),
        "mrr": round(mrr, 4),
        "total_queries": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== PHASE 4C: RETRIEVAL EVALUATION BENCHMARK ===\n")

    vector_store = VectorStoreManager()
    emb_provider = get_embedding_provider()
    hybrid = HybridRetriever(
        vector_store=vector_store,
        dense_candidates=20,
        keyword_candidates=20,
        rrf_k=60,
        final_top_k=TOP_K_EVAL,
    )

    print("Running dense-only baseline...")
    dense_results = run_dense_baseline(BENCHMARK_QUERIES, vector_store, emb_provider)
    dense_metrics = compute_metrics(dense_results)

    print("Running hybrid retrieval evaluation...")
    hybrid_results = run_hybrid_eval(BENCHMARK_QUERIES, hybrid)
    hybrid_metrics = compute_metrics(hybrid_results)

    # Print comparison table
    print(f"\n{'Query':<6}{'Expected':<10}{'Dense Top-1':<14}{'Dense Rank':<12}{'Hybrid Top-1':<14}{'Hybrid Rank':<12}{'Detection'}")
    print("-" * 100)
    for d, h in zip(dense_results, hybrid_results):
        det = h.get("scheme_detection", {})
        det_str = f"{det.get('scheme_id','?')}({det.get('matched_alias','?')})" if det.get("detected") else "none"
        d_ok = "OK" if d["top1_correct"] else "FAIL"
        h_ok = "OK" if h["top1_correct"] else "FAIL"
        print(
            f"{d['query_id']:<6}"
            f"{d['expected_scheme_id']:<10}"
            f"{d['top1_scheme_id']:<8}({d_ok})   "
            f"{str(d['correct_rank']):<12}"
            f"{h['top1_scheme_id']:<8}({h_ok})   "
            f"{str(h['correct_rank']):<12}"
            f"{det_str}"
        )

    print(f"\n{'Metric':<25}{'Dense-Only':<20}{'Hybrid'}")
    print("-" * 55)
    metrics_to_show = [
        ("Top-1 Accuracy", f"{dense_metrics['top1_count']}/{dense_metrics['total_queries']}", f"{hybrid_metrics['top1_count']}/{hybrid_metrics['total_queries']}"),
        ("Top-3 Recall",   f"{dense_metrics['top3_count']}/{dense_metrics['total_queries']}", f"{hybrid_metrics['top3_count']}/{hybrid_metrics['total_queries']}"),
        ("MRR",            f"{dense_metrics['mrr']:.4f}", f"{hybrid_metrics['mrr']:.4f}"),
    ]
    for label, dv, hv in metrics_to_show:
        print(f"{label:<25}{dv:<20}{hv}")

    # Check targets
    target_top1 = hybrid_metrics["top1_count"] >= 8
    target_top3 = hybrid_metrics["top3_count"] >= 8
    mch_rank1 = next((r for r in hybrid_results if r["query_id"] == "Q7"), {}).get("correct_rank") == 1
    aasara_rank1 = next((r for r in hybrid_results if r["query_id"] == "Q8"), {}).get("correct_rank") == 1

    targets_met = {
        "top1_accuracy_8_8": target_top1,
        "top3_recall_8_8": target_top3,
        "mch_kit_rank_1": mch_rank1,
        "aasara_rank_1": aasara_rank1,
        "all_targets_met": all([target_top1, target_top3, mch_rank1, aasara_rank1]),
    }

    print(f"\n=== TARGET VERIFICATION ===")
    for k, v in targets_met.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # Write report
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_config": {
            "total_queries": len(BENCHMARK_QUERIES),
            "top_k_checked": TOP_K_EVAL,
            "hybrid_config": {
                "dense_candidates": 20,
                "keyword_candidates": 20,
                "rrf_k": 60,
                "final_top_k": TOP_K_EVAL,
            },
        },
        "dense_only_baseline": {
            "metrics": dense_metrics,
            "per_query": dense_results,
        },
        "hybrid_retrieval": {
            "metrics": hybrid_metrics,
            "per_query": hybrid_results,
        },
        "improvement": {
            "top1_delta": hybrid_metrics["top1_count"] - dense_metrics["top1_count"],
            "top3_delta": hybrid_metrics["top3_count"] - dense_metrics["top3_count"],
            "mrr_delta": round(hybrid_metrics["mrr"] - dense_metrics["mrr"], 4),
        },
        "targets": targets_met,
        "status": "PASS" if targets_met["all_targets_met"] else "PARTIAL",
    }

    out_path = REPORTS_DIR / "rag_retrieval_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation report saved: {out_path}")
    overall = "EVALUATION PASSED" if targets_met["all_targets_met"] else "EVALUATION PARTIAL — REVIEW TARGETS"
    print(f"\n{'='*60}")
    print(f"  {overall}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
