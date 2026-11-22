"""Step 2: turn every chunk into an embedding and store it in Chroma.

Run this again whenever you change the PDFs or chunk settings.
"""
import json
import chromadb
from openai import OpenAI

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, EMBED_MODEL, CHUNKS_FILE, VECTOR_FOLDER, COLLECTION_NAME,
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
BATCH_SIZE = 100   # embed 100 chunks per API call (faster and cheaper than one by one)


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embeddings(texts):
    """Embed a list of texts in one API call."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def store_chunks(chunks):
    db = chromadb.PersistentClient(path=VECTOR_FOLDER)

    # start fresh so old chunks don't mix with new ones
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # cosine distance: 0 = same meaning, 2 = opposite
    collection = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[f"chunk_{start + i}" for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            embeddings=get_embeddings([c["text"] for c in batch]),
            metadatas=[
                {"scheme": c["scheme"], "file_name": c["file_name"], "page": c["page"]}
                for c in batch
            ],
        )
        print(f"stored {start + len(batch)}/{len(chunks)}")

    print(f"Done: {collection.count()} chunks in Chroma ({VECTOR_FOLDER}/)")


if __name__ == "__main__":
    store_chunks(load_chunks())
