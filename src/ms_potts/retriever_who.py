# retriever_who.py

import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

class WHOBookRetriever:
    def __init__(self):
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load WHO-specific embeddings
        path = os.path.join(os.path.dirname(__file__), 'data', 'who_embeddings.csv')
        self.df = pd.read_csv(path)

        self.embeddings = np.array(
            self.df['embedding'].apply(lambda x: np.array(eval(x), dtype=np.float32)).tolist()
        )
        self.text_chunks = self.df['sentence_chunk'].tolist()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode([query])[0]

    def retrieve(self, query: str) -> str:
        query_embedding = self.embed_query(query)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        similarities = np.dot(self.embeddings, query_norm) / embedding_norms

        max_score = np.max(similarities)
        if max_score < 0.30:
            return "OUT_OF_SCOPE: This question is outside WHO nutrition content."

        best_idx = np.argmax(similarities)
        return self.text_chunks[best_idx]
