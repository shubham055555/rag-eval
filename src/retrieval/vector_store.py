import chromadb
from sentence_transformers import SentenceTransformer

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.chunker import load_and_chunk_documents


CHROMA_PATH = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="documents"
    )

    chunks = load_and_chunk_documents()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts).tolist()

    ids = [
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {len(texts)} chunks in ChromaDB")

    return collection


if __name__ == "__main__":
    create_vector_store()