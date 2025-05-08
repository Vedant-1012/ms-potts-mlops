
# Phase 1: Project Design and Model Development

**Project Title**: Ms. Potts: A Nutrition Assistant Using Retrieval-Augmented Generation  
**Date**: 2025-05-06

---

## 1. Project Overview

*Ms. Potts* is a modular nutrition assistant chatbot that leverages Retrieval-Augmented Generation (RAG) to provide grounded answers to user queries. The goal is to reduce hallucinations commonly found in generic language models by grounding responses in curated nutrition documents. The system uses `sentence-transformers` for semantic embeddings, FAISS for similarity search, and Gemini API for response generation.

---

## 2. Third-Party Tools Used

| Tool/Library             | Purpose                                           |
|--------------------------|---------------------------------------------------|
| `sentence-transformers`  | Embedding nutrition-related documents             |
| `FAISS`                  | Fast approximate nearest-neighbor search          |
| `google-generativeai`    | Generates final responses using Gemini LLM        |

---

## 3. Dataset Summary

| Component               | File                             | Description                                                   |
|-------------------------|----------------------------------|---------------------------------------------------------------|
| Raw Nutrition Data      | `data/nutrition_data.json`       | Scraped content from public nutrition sources                 |
| Chunked Documents       | `chunks/chunked_docs.json`       | Sentence-level chunks prepared for embedding and retrieval    |
| Intent Examples         | `data/intent_examples.json`      | Manually written examples for similarity-based intent matching|
| FAISS Index             | `retriever/faiss_index.index`    | Vector index of embedded documents                            |

---

## 4. Repository Structure

The project is organized into clean, modular folders:
- `data/`: Contains raw and synthetic query datasets
- `chunks/`: Preprocessed and chunked document segments
- `retriever/`: Scripts to generate embeddings and build FAISS index
- `model_gemini.py`: Generates responses using Gemini API
- `app.py`: Flask app entry point for query handling

---

## 5. Environment Setup

We use a `requirements.txt` file for dependency management:
```bash
pip install -r requirements.txt
```
Includes:
- faiss-cpu
- sentence-transformers
- google-generativeai
- flask
- python-dotenv

---

## 6. Data Pipeline Summary

| Step         | Script / Path                  | Description                                                  |
|--------------|--------------------------------|--------------------------------------------------------------|
| Preprocessing| `scripts/preprocess_data.py`   | Cleans and chunks raw nutrition data                         |
| Embedding    | `retriever/generate_embeddings.py` | Creates dense embeddings using `sentence-transformers`   |
| Indexing     | `retriever/build_faiss_index.py` | Builds FAISS index from embeddings                         |

---

## 7. Initial Testing

We have completed the full RAG flow:
- Query → Semantic search → Gemini-generated response
- Manual verification of output shows reduced hallucinations

---

## 8. Notes and Learnings

There’s still ongoing work on:
- Better evaluation methods for responses
- Refining intent classification
- Possible DVC integration for data and index versioning

---
