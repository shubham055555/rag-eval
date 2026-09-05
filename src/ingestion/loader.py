from pathlib import Path


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*"):
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")

            documents.append({
                "source": file_path.name,
                "text": text
            })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded {len(docs)} document(s)")

    for doc in docs:
        print(f"- {doc['source']}: {len(doc['text'])} characters")