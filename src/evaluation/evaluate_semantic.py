import json
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db_semantic"
COLLECTION_NAME = "semantic_documents"
QUESTIONS_PATH = "data/evaluation/questions.json"
TOP_K = 3

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

correct = 0

print("\n========================================")
print("SEMANTIC CHUNKING EVALUATION")
print("========================================")
print(f"Questions: {len(questions)}")
print(f"Top-K: {TOP_K}")

for index, item in enumerate(questions, start=1):

    query = item["question"]
    expected_source = item["expected_source"]

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    retrieved_sources = [
        metadata["source"]
        for metadata in results["metadatas"][0]
    ]

    passed = expected_source in retrieved_sources

    if passed:
        correct += 1
        result = "PASS"
    else:
        result = "FAIL"

    print(
        f"{index:02d}. {result} | "
        f"Expected: {expected_source} | "
        f"Retrieved: {retrieved_sources}"
    )

recall = correct / len(questions)

print("\n========================================")
print("FINAL SEMANTIC RESULT")
print("========================================")
print(f"Correct: {correct}/{len(questions)}")
print(f"Recall@{TOP_K}: {recall:.2%}")
print("========================================")
