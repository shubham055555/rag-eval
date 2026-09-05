import json

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "chroma_db"
QUESTIONS_PATH = "data/evaluation/questions.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    name="documents"
)


def evaluate():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    correct = 0

    for item in questions:
        query = item["question"]
        expected_source = item["expected_source"]

        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        retrieved_sources = [
            metadata["source"]
            for metadata in results["metadatas"][0]
        ]

        if expected_source in retrieved_sources:
            correct += 1

        print("\nQuestion:")
        print(query)

        print("Retrieved:")
        print(retrieved_sources)

        print(
            "Result:",
            "PASS" if expected_source in retrieved_sources else "FAIL"
        )

    accuracy = correct / len(questions)

    print("\n==============================")
    print(f"Correct: {correct}/{len(questions)}")
    print(f"Recall@3: {accuracy:.2%}")
    print("==============================")


if __name__ == "__main__":
    evaluate()