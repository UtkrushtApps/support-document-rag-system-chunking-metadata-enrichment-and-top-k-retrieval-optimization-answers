import re
from typing import List, Dict, Any
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Assume support articles are present as a list of dicts:
# Each dict: {'text': ..., 'category': ..., 'priority': ..., 'date': ...}
# We assume these are loaded into 'support_articles'.

class SupportDocRAG:
    def __init__(self, collection_name: str = "support_docs", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model),
            metadata={"hnsw:space": "cosine"}  # force cosine similarity
        )

    def chunk_text(self, text: str, max_tokens: int = 200, overlap: int = 50) -> List[str]:
        # Use naive whitespace tokenization (can substitute with tiktoken for better accuracy)
        words = text.split()
        n = len(words)
        chunks = []
        i = 0
        while i < n:
            chunk = words[i:i+max_tokens]
            chunks.append(' '.join(chunk))
            if i + max_tokens >= n:
                break
            i += max_tokens - overlap
        return chunks

    def chunk_and_embed_articles(self, support_articles: List[Dict[str, Any]], max_tokens: int = 200, overlap: int = 50) -> None:
        batch_chunks = []
        batch_metadata = []
        batch_ids = []
        all_embeddings = []
        for article in support_articles:
            chunks = self.chunk_text(article['text'], max_tokens, overlap)
            for idx, chunk in enumerate(chunks):
                meta = {
                    "category": article.get("category", "unknown"),
                    "priority": article.get("priority", "normal"),
                    "date": article.get("date", "unknown"),
                    "article_id": article.get("id", "unknown"),
                    "chunk_idx": idx
                }
                batch_chunks.append(chunk)
                batch_metadata.append(meta)
                batch_ids.append(str(uuid.uuid4()))
        # Embed all chunks
        if batch_chunks:
            embeddings = self.model.encode(batch_chunks, show_progress_bar=True, normalize_embeddings=True)
            # Store in Chroma
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadata,
                ids=batch_ids
            )

    def query(self, user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([user_query], normalize_embeddings=True)[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Chromadb returns lists under key per query (since batched queries are possible), so use [0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        output = []
        for doc, meta, dist in zip(docs, metas, distances):
            result = {
                "document": doc,
                "metadata": meta,
                "cosine_distance": dist,
                "cosine_similarity": 1-dist
            }
            output.append(result)
        # Sort by similarity descending for reporting (though already in top order)
        output.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return output

# Example (assuming support_articles is loaded as described)
if __name__ == "__main__":
    # Example articles (for test/demo)
    support_articles = [
        {
            "id": "a1",
            "text": "How to reset your password. If you've forgotten your password, click on 'Forgot password' on the login page...",
            "category": "Account",
            "priority": "high",
            "date": "2024-02-21"
        },
        {
            "id": "a2",
            "text": "Troubleshooting login issues. Common reasons for login problems: wrong credentials, account lockout...",
            "category": "Account",
            "priority": "medium",
            "date": "2024-04-12"
        }
    ]

    rag = SupportDocRAG()
    print("Processing and chunking articles...")
    rag.chunk_and_embed_articles(support_articles)
    test_queries = [
        "How do I change my password?",
        "Cannot sign into my account",
        "Lockout troubleshooting steps",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = rag.query(q, top_k=5)
        for idx, res in enumerate(results, 1):
            print(f"{idx}. [Sim: {res['cosine_similarity']:.3f}] [{res['metadata']['category']}] Priority: {res['metadata']['priority']} | {res['document']}")
