import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db_semantic"
COLLECTION_NAME = "semantic_documents"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

query = "How many weeks of parental leave are employees eligible for?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nQUERY:")
print(query)

print("\nSEMANTIC RETRIEVAL RESULTS:")

for i, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][i]

    print("\n------------------------------")
    print(f"Rank: {i + 1}")
    print(f"Source: {metadata['source']}")
    print(f"Chunk ID: {metadata['chunk_id']}")
    print("\nContent:")
    print(document)
