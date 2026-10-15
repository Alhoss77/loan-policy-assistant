import chromadb
from sentence_transformers import SentenceTransformer

from ingest import DATABASE_DIR, MODEL_NAME, COLLECTION_NAME


def main():
    # Open the database created by ingest.py.
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() == 0:
        raise ValueError("The database is empty. Run ingest.py first.")

    # Use the SAME embedding model used for the documents.
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\nSearch is ready. Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        # Convert the question into an embedding.
        question_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        )

        # Retrieve the three closest chunks.
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # [0] selects the results for our single question.
        documents = results["documents"][0]
        metadata = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (text, source, distance) in enumerate(
            zip(documents, metadata, distances),
            start=1,
        ):
            print(f"\n--- Result {rank} ---")
            print(f"Source: {source['source']}")
            print(f"Chunk: {source['chunk']}")
            print(f"Distance: {distance:.4f}")
            print(text)


if __name__ == "__main__":
    main()