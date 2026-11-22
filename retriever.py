"""Step 3: find the chunks most similar in meaning to a question."""
import chromadb
from openai import OpenAI

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, EMBED_MODEL, VECTOR_FOLDER, COLLECTION_NAME, TOP_K,
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
_collection = None


def get_collection():
    """Open the Chroma collection once and reuse it."""
    global _collection
    if _collection is None:
        db = chromadb.PersistentClient(path=VECTOR_FOLDER)
        _collection = db.get_collection(COLLECTION_NAME)
    return _collection


def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def retrieve(question, scheme, top_k=TOP_K):
    """Return the top_k chunks for this scheme, each with a similarity score (0..1)."""
    results = get_collection().query(
        query_embeddings=[get_embedding(question)],
        n_results=top_k,
        where={"scheme": scheme},
    )
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "file_name": meta["file_name"],
            "page": meta["page"],
            "score": round(1 - dist, 3),   # cosine similarity: higher = closer meaning
        })
    return chunks


if __name__ == "__main__":
    for c in retrieve("Who is not eligible for PM-KISAN?", "PM-KMY"):
        print(f"{c['file_name']} page {c['page']}  (score {c['score']})")
        print(c["text"][:300], "\n")
