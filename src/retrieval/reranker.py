import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_search import hybrid_search

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 6
TOP_K = 3

print("Loading reranker model...")
reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query, candidates, top_k=TOP_K):

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["reranker_score"] = float(score)
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return ranked[:top_k]


def retrieve_and_rerank(query):

    candidates = hybrid_search(
        query,
        top_k=CANDIDATE_K
    )

    return rerank(
        query,
        candidates,
        top_k=TOP_K
    )


if __name__ == "__main__":

    query = "How many weeks of parental leave are employees eligible for?"

    results = retrieve_and_rerank(query)

    print("\n========================================")
    print("RERANKED SEARCH RESULTS")
    print("========================================")

    print(f"\nQuery:\n{query}")

    for result in results:

        print("\n----------------------------------------")
        print(f"Rank: {result['rank']}")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")

        print("\nContent:")
        print(result["text"])
