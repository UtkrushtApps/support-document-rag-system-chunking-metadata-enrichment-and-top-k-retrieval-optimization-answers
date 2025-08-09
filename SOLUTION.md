# Solution Steps

1. 1. Implement the SupportDocRAG class that encapsulates all logic for chunking, embedding, metadata enrichment, and Chroma retrieval.

2. 2. Implement the chunk_text method: break input text into 200-token chunks (using whitespace), with 50-token overlap between subsequent chunks.

3. 3. Implement the chunk_and_embed_articles method: for each support article, produce chunks, embed them with SentenceTransformer, and store them in the Chroma vector collection along with category, priority, date, article_id, and chunk_idx as metadata.

4. 4. Set up Chroma collection for cosine similarity by specifying hnsw:space as cosine and use sentence-transformers and built-in embedding function. Store each chunk, embedding, and its associated metadata in the collection.

5. 5. Implement the query method: embed the query, search Chroma for top-k (k=5) most similar chunks by cosine similarity, and return matched documents along with their metadata and similarity scores.

6. 6. Provide a main/demo block and a pytest file for functional verification, including spot checks for retrieval accuracy and metadata inclusion using various support queries.

