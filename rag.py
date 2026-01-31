# pip install qdrant-client sentence-transformers
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os 


COLLECTION = "docs"
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")  # e.g., "https://your-cluster-url"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # if needed
# 1) Vector DB client (local Qdrant in Docker: http://localhost:6333)
qdrant = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

# 2) Embedder (swap this with your OpenAI/legal embedder later)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = embedder.get_sentence_embedding_dimension()

# 3) Create collection
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

def upsert_chunks(doc_id: str, chunks: list[str]):
    vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
    points = []
    for i, (text, vec) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=f"{doc_id}:{i}",
                vector=vec,
                payload={"doc_id": doc_id, "chunk_index": i, "text": text},
            )
        )
    qdrant.upsert(collection_name=COLLECTION, points=points)

def retrieve(doc_id: str, query: str, k: int = 12):
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=k,
        query_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
    )
    # sort by chunk_index if you want “document order” instead of pure similarity
    chunks = [h.payload["text"] for h in hits]
    return chunks

"""

def build_summary_prompt(retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    return (
        "You are summarizing a legal document using ONLY the provided excerpts.\n"
        "Write:\n"
        "1) Executive summary (<=250 words)\n"
        "2) Key facts (bullets)\n"
        "3) Issues / disputes / unknowns (bullets)\n\n"
        f"EXCERPTS:\n{context}\n"
    )
"""