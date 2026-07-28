"""
Evaluation harness: runs a fixed set of test questions through the
full RAG pipeline (hybrid search + metadata filtering + neighbor
expansion + generation) and scores results against expected files
and keywords.

Usage:
    python src/evaluate.py --questions tests/eval_questions.json
"""

import argparse
import json
import sys

sys.path.insert(0, ".")
from src.query import (
    get_clients,
    ensure_search_pipeline,
    hybrid_retrieve,
    fetch_neighbors,
    generate_answer,
)


def expand_with_neighbors(opensearch, top_hits):
    seen = {(c["file"], c["chunk_index"]) for c in top_hits}
    chunks = list(top_hits)
    for chunk in top_hits:
        for neighbor in fetch_neighbors(opensearch, chunk):
            key = (neighbor["file"], neighbor["chunk_index"])
            if key not in seen:
                seen.add(key)
                chunks.append(neighbor)
    return chunks


def evaluate_question(opensearch, bedrock, case):
    question = case["question"]
    expected_files = set(case["expected_files"])
    expected_keywords = case["expected_keywords"]

    top_hits = hybrid_retrieve(opensearch, bedrock, question, k=8)
    chunks = expand_with_neighbors(opensearch, top_hits)

    retrieved_files = {c["file"] for c in chunks}
    if expected_files:
        retrieval_hit = bool(retrieved_files & expected_files)
    else:
        retrieval_hit = None

    answer = generate_answer(bedrock, question, chunks)
    answer_lower = answer.lower()
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    keyword_coverage = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0

    return {
        "question": question,
        "retrieval_hit": retrieval_hit,
        "retrieved_files": sorted(retrieved_files),
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched_keywords,
        "missed_keywords": [kw for kw in expected_keywords if kw not in matched_keywords],
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="tests/eval_questions.json")
    args = parser.parse_args()

    with open(args.questions) as f:
        cases = json.load(f)

    bedrock, opensearch = get_clients()
    ensure_search_pipeline(opensearch)

    results = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case['question']}")
        result = evaluate_question(opensearch, bedrock, case)
        results.append(result)
        if result["retrieval_hit"] is None:
            status = "N/A (graceful-failure test)"
        else:
            status = "PASS" if result["retrieval_hit"] else "FAIL"
        print(f"  Retrieval: {status}  Keyword coverage: {result['keyword_coverage']:.0%}")

    scored_results = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_hit_rate = (
        sum(r["retrieval_hit"] for r in scored_results) / len(scored_results)
        if scored_results else None
    )
    avg_keyword_coverage = sum(r["keyword_coverage"] for r in results) / len(results)

    print("\n" + "=" * 60)
    if retrieval_hit_rate is not None:
        excluded = len(results) - len(scored_results)
        print(f"Retrieval hit rate:     {retrieval_hit_rate:.0%} ({sum(r['retrieval_hit'] for r in scored_results)}/{len(scored_results)}) -- {excluded} graceful-failure test(s) excluded")
    print(f"Avg keyword coverage:   {avg_keyword_coverage:.0%}")

    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to data/eval_results.json")


if __name__ == "__main__":
    main()