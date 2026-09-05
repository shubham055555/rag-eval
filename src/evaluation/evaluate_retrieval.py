import json

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "chroma_db"
QUESTIONS_PATH = "data/evaluation/questions.json"
TOP_K = 3


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name="documents"
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate():

    # Load evaluation questions
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    total_questions = len(questions)
    correct = 0

    print("\n========================================")
    print("RAG RETRIEVAL EVALUATION")
    print("========================================")

    print(f"Questions: {total_questions}")
    print(f"Top-K: {TOP_K}")

    # ----------------------------------------------
    # Evaluate every question
    # ----------------------------------------------

    for index, item in enumerate(questions, start=1):

        query = item["question"]
        expected_source = item["expected_source"]
        expected_answer = item["expected_answer"]

        # Create query embedding
        query_embedding = model.encode(
            query
        ).tolist()

        # Retrieve documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )

        retrieved_sources = [
            metadata["source"]
            for metadata in results["metadatas"][0]
        ]

        # ------------------------------------------
        # Check retrieval
        # ------------------------------------------

        passed = expected_source in retrieved_sources

        if passed:

            correct += 1

            result = "PASS"

        else:

            result = "FAIL"

        # ------------------------------------------
        # Print result
        # ------------------------------------------

        print("\n----------------------------------------")
        print(f"Question {index}/{total_questions}")
        print("----------------------------------------")

        print("Question:")
        print(query)

        print("\nExpected Source:")
        print(expected_source)

        print("\nRetrieved Sources:")
        print(retrieved_sources)

        print("\nResult:")
        print(result)

        # ------------------------------------------
        # Failure analysis
        # ------------------------------------------

        if not passed:

            print("\n!!! FAILURE ANALYSIS !!!")

            print("\nExpected Answer:")
            print(expected_answer)

            print("\nRetrieved Chunks:")

            for rank, document in enumerate(
                results["documents"][0],
                start=1
            ):

                metadata = results["metadatas"][0][rank - 1]

                print(f"\n--- Rank {rank} ---")

                print(
                    f"Source: {metadata['source']}"
                )

                print(
                    f"Chunk ID: {metadata['chunk_id']}"
                )

                print("\nContent:")

                print(document)

    # ----------------------------------------------
    # Calculate Recall@K
    # ----------------------------------------------

    recall = correct / total_questions

    print("\n")
    print("========================================")
    print("FINAL EVALUATION RESULT")
    print("========================================")

    print(
        f"Correct: {correct}/{total_questions}"
    )

    print(
        f"Recall@{TOP_K}: {recall:.2%}"
    )

    print("========================================")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    evaluate()