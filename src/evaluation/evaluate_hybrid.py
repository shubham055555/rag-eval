import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
from src.retrieval.hybrid_search import hybrid_search

QUESTIONS_PATH = "data/evaluation/questions.json"
TOP_K = 3

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

correct = 0

print("\n========================================")
print("HYBRID SEARCH EVALUATION")
print("========================================")
print(f"Questions: {len(questions)}")
print(f"Top-K: {TOP_K}")

for index, item in enumerate(questions, start=1):

    query = item["question"]
    expected_source = item["expected_source"]

    results = hybrid_search(query, top_k=TOP_K)

    retrieved_sources = [
        result["source"]
        for result in results
    ]

    passed = expected_source in retrieved_sources

    if passed:
        correct += 1
        result_status = "PASS"
    else:
        result_status = "FAIL"

    print(
        f"{index:02d}. {result_status} | "
        f"Expected: {expected_source} | "
        f"Retrieved: {retrieved_sources}"
    )

recall = correct / len(questions)

print("\n========================================")
print("FINAL HYBRID RESULT")
print("========================================")
print(f"Correct: {correct}/{len(questions)}")
print(f"Recall@{TOP_K}: {recall:.2%}")
print("========================================")
