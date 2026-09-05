import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    name="documents"
)


query = "How many days of parental leave are employees eligible for?"

query_embedding = model.encode(query).tolist()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


print("\nQUERY:")
print(query)

print("\nRETRIEVED DOCUMENTS:")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print("\n---")
    print(f"Rank: {i + 1}")
    print(f"Source: {metadata['source']}")
    print(f"Chunk ID: {metadata['chunk_id']}")
    print(document)