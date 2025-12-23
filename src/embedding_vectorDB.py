import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PROCESSED_DIR = os.path.join(ROOT_DIR, "processed_data")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load documents
with open(os.path.join(PROCESSED_DIR, "documents.json"), "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [doc["text"] for doc in documents]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
embedding_matrix = np.array(embeddings).astype("float32")

# Save embeddings
np.save(os.path.join(EMBEDDINGS_DIR, "wiki_embeddings.npy"), embedding_matrix)

# Normalize for cosine similarity
faiss.normalize_L2(embedding_matrix)

# Build FAISS index (Inner Product = Cosine Similarity)
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedding_matrix)

# Save FAISS index
faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "wiki_faiss.index"))

# Store metadata for retrieval
doc_store = {doc["id"]: {"text": doc["text"], "source": doc["source"]} for doc in documents}

# Optional: save metadata
with open(os.path.join(EMBEDDINGS_DIR, "doc_store.json"), "w", encoding="utf-8") as f:
    json.dump(doc_store, f, ensure_ascii=False, indent=4)

print(f"Embeddings shape: {embedding_matrix.shape}")
print(f"FAISS index contains {index.ntotal} vectors.")

# Retrieval function
def search_vector_db(query, top_k=5):
    query_embedding = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = [doc_store[idx]["text"] for idx in indices[0]]
    return results

# Quick test
if __name__ == "__main__":
    query = "What is artificial intelligence?"
    results = search_vector_db(query)
    for i, res in enumerate(results, 1):
        print(f"\nResult {i}:\n{res[:400]}")
