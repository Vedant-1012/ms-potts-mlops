# generate_knowledge_embeddings.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# File paths
knowledge_file = os.path.join('data', 'knowledge_texts.txt')
output_file = os.path.join('data', 'embeddings.csv')

# Read knowledge chunks
if not os.path.exists(knowledge_file):
    raise FileNotFoundError(f"Knowledge file not found at {knowledge_file}")

with open(knowledge_file, 'r', encoding='utf-8') as f:
    knowledge_texts = [line.strip() for line in f if line.strip()]

# Embed knowledge
embeddings = model.encode(knowledge_texts)

# Create dataframe
df = pd.DataFrame({
    'sentence_chunk': knowledge_texts,
    'embedding': [embedding.tolist() for embedding in embeddings]
})

# Save to CSV
os.makedirs('data', exist_ok=True)
df.to_csv(output_file, index=False)

print(f"✅ embeddings.csv generated successfully with {len(df)} entries at {output_file}")
