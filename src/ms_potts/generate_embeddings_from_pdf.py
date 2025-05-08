import os
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Step 1: Load and extract text from PDF ----------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# ---------- Step 2: Split text into chunks ----------
def chunk_text(text, max_chars=500):
    import textwrap
    chunks = textwrap.wrap(text, max_chars, break_long_words=False, replace_whitespace=False)
    return [chunk.strip().replace("\n", " ") for chunk in chunks if chunk.strip()]

# ---------- Step 3: Embed each chunk ----------
def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# ---------- Step 4: Save to CSV ----------
def save_embeddings(chunks, embeddings, output_csv='embeddings.csv'):
    df = pd.DataFrame({
        "sentence_chunk": chunks,
        "embedding": [emb.tolist() for emb in embeddings]
    })
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} chunks with embeddings to {output_csv}")

# ---------- Run the pipeline ----------
if __name__ == "__main__":
    pdf_file = "WHO_NHD_00.6.pdf" 
    print("📖 Extracting text...")
    raw_text = extract_text_from_pdf(pdf_file)

    print("✂️ Splitting into chunks...")
    chunks = chunk_text(raw_text, max_chars=500)

    print(f"🔍 Total chunks: {len(chunks)}")
    print("💡 Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    print("💾 Saving to CSV...")
    save_embeddings(chunks, embeddings, output_csv="who_embeddings.csv")
