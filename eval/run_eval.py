"""
Retrieval evaluation: for each question, checks whether the chunk retrieved
from the *expected* source document appears in the top-k results.

Reports Hit@k (did we retrieve at least one correct-source chunk in the top k)
and MRR (Mean Reciprocal Rank — rewards ranking the right source higher, not
just including it somewhere in the top k).

Run: python eval/run_eval.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag_engine import RAGEngine

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")


def main(k=4):
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
        status = "✓" if r["hit"] else "✗"
        print(f"[{status}] {r['question'][:65]}")
        print(f"     expected: {r['expected']:<28} retrieved: {r['retrieved']}")
    print(f"{'='*60}")
    print(f"Hit@{k}: {hit_rate:.1%}")
    print(f"MRR:    {mrr:.3f}")
    print(f"{'='*60}\n")

    return {"hit_rate": hit_rate, "mrr": mrr}


if __name__ == "__main__":
    main()
