from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "chroma_data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "policy_documents"


def main():
    # 1. Find our Markdown documents.
    files = sorted(DATA_DIR.glob("*.md"))

    if not files:
        raise ValueError("No Markdown files found in the data folder.")

    # Sizes here are measured in characters, not tokens.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        length_function=len,
    )

    texts = []
    metadata = []
    ids = []

    # 2. Read and split each document.
    for file in files:
        text = file.read_text(encoding="utf-8-sig")

        if not text.strip():
            raise ValueError(f"The document is empty: {file.name}")

        chunks = splitter.split_text(text)

        print(f"{file.name}: {len(chunks)} chunks")

        for chunk_number, chunk in enumerate(chunks, start=1):
            texts.append(chunk)

            metadata.append({
                "source": file.name,
                "chunk": chunk_number,
            })

            ids.append(f"{file.name}::chunk_{chunk_number}")

    # 3. Load the embedding model and encode the chunks.
    print("\nLoading embedding model...")
    print("The first run may download model files. Please wait.")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\nCreating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding array shape: {embeddings.shape}")

    # 4. Open a local database that persists after the script exits.
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    # Upsert adds new records or updates records with matching IDs.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
    )

    # Remove outdated chunks if documents were shortened or deleted.
    existing_ids = collection.get()["ids"]
    current_ids = set(ids)
    outdated_ids = [
        record_id
        for record_id in existing_ids
        if record_id not in current_ids
    ]

    if outdated_ids:
        collection.delete(ids=outdated_ids)

    print("\nIndexing complete!")
    print(f"Documents processed: {len(files)}")
    print(f"Chunks stored: {collection.count()}")
    print(f"Database folder: {DATABASE_DIR}")


if __name__ == "__main__":
    main()