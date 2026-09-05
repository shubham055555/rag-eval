from pathlib import Path


DOCUMENTS_DIR = Path("data/documents")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str):
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_and_chunk_documents():
    all_chunks = []

    for file_path in DOCUMENTS_DIR.glob("*"):
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": file_path.name,
                "chunk_id": i
            })

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n---")
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(chunk["text"][:200])