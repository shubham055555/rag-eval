import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json

from src.retrieval.reranker import retrieve_and_rerank

QUESTIONS_PATH = "data/evaluation/questions.json"
TOP_K = 3

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

correct = 0
failures = []

print("\n========================================")
print("RERANKER EVALUATION")
print("========================================")
print(f"Questions: {len(questions)}")
print(f"Top-K: {TOP_K}")

for index, item in enumerate(questions, start=1):

    query = item["question"]
    expected_source = item["expected_source"]

    results = retrieve_and_rerank(query)

    retrieved_sources = [
        result["source"]
        for result in results
    ]

    passed = expected_source in retrieved_sources

    if passed:
        correct += 1
        status = "PASS"
    else:
        status = "FAIL"
        failures.append({
            "question": query,
            "expected": expected_source,
            "retrieved": retrieved_sources
        })

    print(
        f"{index:02d}. {status} | "
        f"Expected: {expected_source} | "
        f"Retrieved: {retrieved_sources}"
    )

recall = correct / len(questions)

print("\n========================================")
print("FINAL RERANKER RESULT")
print("========================================")
print(f"Correct: {correct}/{len(questions)}")
print(f"Recall@{TOP_K}: {recall:.2%}")
print(f"Failures: {len(failures)}")
print("========================================")

if failures:
    print("\nFAILURE ANALYSIS")

    for failure in failures:
        print("\n----------------------------------------")
        print("Question:")
        print(failure["question"])
        print("\nExpected:")
        print(failure["expected"])
        print("\nRetrieved:")
        print(failure["retrieved"])
