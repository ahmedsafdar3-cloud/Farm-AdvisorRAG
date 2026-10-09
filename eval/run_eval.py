"""
Retrieval evaluation: for each question, checks whether the chunk retrieved
from the *expected* source document appears in the top-k results.

Reports Hit@k (did we retrieve at least one correct-source chunk in the top k)
and MRR (Mean Reciprocal Rank — rewards ranking the right source higher, not
just including it somewhere in the top k).

Run: python eval/run_eval.py
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag_engine import RAGEngine

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")


def main(k=4, output=None):
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    engine = RAGEngine(top_k=k)

    hits = 0
    reciprocal_ranks = []
    results_log = []

    for item in eval_set:
        question = item["question"]
        expected = item["expected_source"]

        retrieved = engine.retrieve(question, k=k)
        retrieved_sources = [c["source"] for c in retrieved]

        hit = expected in retrieved_sources
        hits += int(hit)

        if hit:
            rank = retrieved_sources.index(expected) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        results_log.append({
            "question": question,
            "expected": expected,
            "retrieved": retrieved_sources,
            "hit": hit,
        })

    hit_rate = hits / len(eval_set)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION  (top_k={k}, {len(eval_set)} questions)")
    print(f"{'='*60}")
    for r in results_log:
        status = "PASS" if r["hit"] else "MISS"
        print(f"[{status}] {r['question'][:65]}")
        print(f"     expected: {r['expected']:<28} retrieved: {r['retrieved']}")
    print(f"{'='*60}")
    print(f"Hit@{k}: {hit_rate:.1%}")
    print(f"MRR:    {mrr:.3f}")
    print(f"{'='*60}\n")

    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "backend": engine.embedder.backend, "top_k": k,
              "question_count": len(eval_set), "hit_rate": hit_rate,
              "mrr": mrr, "cases": results_log}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate source retrieval; no LLM calls.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    main(args.top_k, args.output)
