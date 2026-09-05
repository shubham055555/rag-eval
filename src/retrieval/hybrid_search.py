import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.ingestion.semantic_chunker import load_and_chunk_documents

CHROMA_PATH = "chroma_db_semantic"
COLLECTION_NAME = "semantic_documents"
TOP_K = 3

print("Loading documents...")
chunks = load_and_chunk_documents()

documents = [chunk["text"] for chunk in chunks]
sources = [chunk["source"] for chunk in chunks]
chunk_ids = [chunk["chunk_id"] for chunk in chunks]

# -----------------------------
# BM25 keyword retrieval
# -----------------------------
tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

# -----------------------------
# Dense retrieval
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def min_max_normalize(scores):
    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def hybrid_search(query, top_k=TOP_K, alpha=0.5):

    # =============================
    # 1. BM25 scores
    # =============================
    bm25_scores = bm25.get_scores(query.lower().split())

    # =============================
    # 2. Dense similarity
    # =============================
    query_embedding = model.encode(query).tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)
    )

    dense_ids = dense_results["ids"][0]
    dense_distances = dense_results["distances"][0]

    dense_score_map = {}

    for doc_id, distance in zip(dense_ids, dense_distances):
        # Chroma distance -> similarity
        dense_score_map[doc_id] = 1 / (1 + distance)

    dense_scores = [
        dense_score_map.get(str(i), 0.0)
        for i in range(len(documents))
    ]

    # =============================
    # 3. Normalize scores
    # =============================
    bm25_normalized = min_max_normalize(bm25_scores)
    dense_normalized = min_max_normalize(dense_scores)

    # =============================
    # 4. Combine
    # =============================
    hybrid_scores = []

    for i in range(len(documents)):
        score = (
            alpha * dense_normalized[i]
            + (1 - alpha) * bm25_normalized[i]
        )

        hybrid_scores.append(score)

    # =============================
    # 5. Rank
    # =============================
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: hybrid_scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "rank": len(results) + 1,
            "source": sources[index],
            "chunk_id": chunk_ids[index],
            "score": hybrid_scores[index],
            "bm25_score": bm25_normalized[index],
            "dense_score": dense_normalized[index],
            "text": documents[index]
        })

    return results


if __name__ == "__main__":

    query = "How many weeks of parental leave are employees eligible for?"

    results = hybrid_search(query)

    print("\n========================================")
    print("HYBRID SEARCH RESULTS")
    print("========================================")

    print(f"\nQuery:\n{query}")

    for result in results:

        print("\n----------------------------------------")
        print(f"Rank: {result['rank']}")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Hybrid Score: {result['score']:.4f}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Dense Score: {result['dense_score']:.4f}")

        print("\nContent:")
        print(result["text"])
