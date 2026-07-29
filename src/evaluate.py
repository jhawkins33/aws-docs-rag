"""
Evaluation harness: runs a fixed set of test questions through the
full RAG pipeline (hybrid search + metadata filtering + neighbor
expansion + generation) and scores results against expected outcomes.

Two scoring modes:
  - Keyword coverage (default): fast, free, deterministic, but can
    penalize a correct answer phrased differently than expected.
  - LLM-as-judge (--judge): Claude evaluates whether the answer
    actually satisfies natural-language criteria. More semantically
    aware, but costs an extra Bedrock call per question and is not
    perfectly deterministic between runs.

Usage:
    python src/evaluate.py --questions tests/eval_questions.json
    python src/evaluate.py --judge
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
    GENERATION_MODEL_ID,
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


def llm_judge(bedrock, question, answer, criteria):
    """
    Ask Claude to judge whether the generated answer satisfies the
    natural-language criteria for a correct answer. Returns a dict
    with a pass/fail verdict and a short reason.
    """
    prompt = f"""You are evaluating whether an AI-generated answer correctly addresses a question, based on specific criteria.

Question: {question}

Criteria for a correct answer: {criteria}

Generated answer: {answer}

Does the generated answer satisfy the criteria? Respond with ONLY a JSON object in this exact format, no other text:
{{"verdict": "PASS" or "FAIL", "reason": "one sentence explaining why"}}"""

    response = bedrock.invoke_model(
        modelId=GENERATION_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"].strip()

    try:
        # Claude sometimes wraps JSON in markdown code fences despite
        # being told not to — strip them before parsing.
        clean = text.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(clean)
        return {"pass": verdict["verdict"] == "PASS", "reason": verdict["reason"]}
    except (json.JSONDecodeError, KeyError):
        return {"pass": None, "reason": f"Could not parse judge response: {text}"}


def evaluate_question(opensearch, bedrock, case, use_judge):
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

    result = {
        "question": question,
        "retrieval_hit": retrieval_hit,
        "retrieved_files": sorted(retrieved_files),
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched_keywords,
        "missed_keywords": [kw for kw in expected_keywords if kw not in matched_keywords],
        "answer": answer,
    }

    if use_judge:
        judge_result = llm_judge(bedrock, question, answer, case["judge_criteria"])
        result["judge_pass"] = judge_result["pass"]
        result["judge_reason"] = judge_result["reason"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="tests/eval_questions.json")
    parser.add_argument("--judge", action="store_true", help="Also score answers with LLM-as-judge")
    args = parser.parse_args()

    with open(args.questions) as f:
        cases = json.load(f)

    bedrock, opensearch = get_clients()
    ensure_search_pipeline(opensearch)

    results = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case['question']}")
        result = evaluate_question(opensearch, bedrock, case, args.judge)
        results.append(result)

        if result["retrieval_hit"] is None:
            status = "N/A (graceful-failure test)"
        else:
            status = "PASS" if result["retrieval_hit"] else "FAIL"
        line = f"  Retrieval: {status}  Keyword coverage: {result['keyword_coverage']:.0%}"
        if args.judge:
            judge_status = "PASS" if result["judge_pass"] else ("FAIL" if result["judge_pass"] is False else "UNPARSEABLE")
            line += f"  Judge: {judge_status}"
        print(line)
        if args.judge:
            print(f"    Judge reason: {result['judge_reason']}")

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

    if args.judge:
        judged = [r for r in results if r["judge_pass"] is not None]
        judge_pass_rate = sum(r["judge_pass"] for r in judged) / len(judged) if judged else None
        if judge_pass_rate is not None:
            print(f"LLM-judge pass rate:    {judge_pass_rate:.0%} ({sum(r['judge_pass'] for r in judged)}/{len(judged)})")

    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to data/eval_results.json")


if __name__ == "__main__":
    main()