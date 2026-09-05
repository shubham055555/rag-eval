from pathlib import Path
import re

DOCUMENTS_DIR = Path("data/documents")
MAX_CHUNK_SIZE = 800


def split_into_paragraphs(text: str):
    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def create_semantic_chunks(text: str):
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) <= MAX_CHUNK_SIZE:

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_and_chunk_documents():

    all_chunks = []

    for file_path in DOCUMENTS_DIR.glob("*"):

        if not file_path.is_file():
            continue

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = create_semantic_chunks(text)

        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": file_path.name,
                    "chunk_id": chunk_id
                }
            )

    return all_chunks


if __name__ == "__main__":

    chunks = load_and_chunk_documents()

    print(f"Total semantic chunks: {len(chunks)}")

    for chunk in chunks[:5]:

        print("\n--------------------------------")
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("--------------------------------")
        print(chunk["text"])
