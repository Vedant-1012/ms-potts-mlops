# potts.py

import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

class IntentClassifier:
    def __init__(self):
        # Load embeddings for intent templates
        embed_path = os.path.join(os.path.dirname(__file__), 'data', 'intent_embeddings_all.csv')
        self.intent_df = pd.read_csv(embed_path)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.intent_texts = self.intent_df['Intent'].tolist()
        self.intent_categories = self.intent_df['Category'].tolist()
        self.intent_embeddings = np.array(
            self.intent_df.drop(['Intent', 'Category'], axis=1).applymap(float).to_numpy()
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed the user query."""
        return self.model.encode([query])[0]

    def compute_similarity(self, query_embedding: np.ndarray) -> int:
        """Compute cosine similarity and find best matching intent."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        intent_norms = np.linalg.norm(self.intent_embeddings, axis=1)
        similarities = np.dot(self.intent_embeddings, query_norm) / intent_norms

        best_idx = np.argmax(similarities)
        return best_idx, similarities[best_idx]

    def classify_from_embedding(self, query_embedding: np.ndarray) -> dict:
        """Classify user query."""
        idx, score = self.compute_similarity(query_embedding)
        return {
            "top_intent": self.intent_texts[idx],
            "top_category": self.intent_categories[idx],
            "confidence_score": float(score)
        }
