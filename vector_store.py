import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME, EMBED_BATCH_SIZE


def build_vector_store(
    all_chunks: list[dict],
    db_path: str,
    embed_model: SentenceTransformer,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(db_path)
    collection = client.get_or_create_collection("enterprise_rag")

    # Efficient ID check: fetch only IDs, not documents
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already embedded — skipping.\n")
        return collection

    print(f"Embedding {len(new_chunks)} new chunks (batch size {EMBED_BATCH_SIZE})…")

    # Single batched encode call — 10-20x faster than one-by-one
    texts = [c["text"] for c in new_chunks]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
    ).tolist()

    metadatas = [
        {"source_file": str(c["source_file"]), "chunk_id": str(c["chunk_id"]), **({}if (page := c.get("page")) is None else {"page": int(page)}), **({}if (row := c.get("row")) is None else {"row": int(row)})}
        for c in new_chunks
    ]

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print("Embedding complete.\n")
    return collection