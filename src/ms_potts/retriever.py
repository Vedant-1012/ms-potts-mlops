# retriever.py

import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Path to embeddings.csv
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'embeddings.csv')
        self.knowledge_df = pd.read_csv(data_path)
        
        # Load embeddings
        self.knowledge_embeddings = np.array(
            self.knowledge_df['embedding'].apply(lambda x: np.array(eval(x), dtype=np.float32)).tolist()
        )
        self.knowledge_texts = self.knowledge_df['sentence_chunk'].tolist()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode([query])[0]

    def retrieve(self, query: str) -> str:
        query_embedding = self.embed_query(query)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        knowledge_norms = np.linalg.norm(self.knowledge_embeddings, axis=1)
        similarities = np.dot(self.knowledge_embeddings, query_norm) / knowledge_norms

        max_score = np.max(similarities)
        if max_score < 0.15:
            return "OUT_OF_SCOPE: This question is outside nutrition domain."
        
        most_relevant_idx = np.argmax(similarities)
        return self.knowledge_texts[most_relevant_idx]