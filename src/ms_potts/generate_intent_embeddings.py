# generate_intent_embeddings.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Create the /data folder if not exists
os.makedirs('data', exist_ok=True)

# Initialize the sentence-transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define your example intents and their categories
intent_examples = [
    ("I ate an omelette today.", "Meal-Logging"),
    ("Suggest a high-protein meal plan.", "Meal-Planning-Recipes"),
    ("What are the benefits of iron?", "Educational-Content"),
    ("How much water should a 25-year-old drink?", "Personalized-Health-Advice")
]

# Separate texts and categories
texts = [example[0] for example in intent_examples]
categories = [example[1] for example in intent_examples]

# Generate embeddings
embeddings = model.encode(texts)

# Create dataframe
df = pd.DataFrame(embeddings)
df['Intent'] = texts
df['Category'] = categories

# Save to CSV
csv_path = os.path.join('data', 'intent_embeddings_all.csv')
df.to_csv(csv_path, index=False)

print(f"✅ intent_embeddings_all.csv generated at {csv_path}")
