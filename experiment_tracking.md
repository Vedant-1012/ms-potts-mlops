
# 🧪 Experiment Tracking (MLflow)

## 1. Overview

**Tool Used:** [MLflow](https://mlflow.org/)  
**Tracking URI:** `http://localhost:5000` (local MLflow UI)  
**Experiment Name:** `ms_potts_nutrition`

MLflow was integrated into the Ms. Potts project to track key experiment data including:

- Model parameters
- Performance metrics
- Output artifacts
- System-level metrics

---

## 2. What's Tracked

### ✅ Parameters
- `model_name`: e.g., `gemini-1.5-flash`
- `embedding_model`: e.g., `all-MiniLM-L6-v2`
- `batch_size`, `learning_rate`, `epochs`, `optimizer`

### ✅ Metrics
- `latency_ms`: time to generate Gemini response
- `tokens_generated`: token count of the final response
- `embedding_time`, `retrieval_time`
- `accuracy`, `loss`, `val_accuracy`, `val_loss` (logged during training runs)

### ✅ Artifacts
- JSON output files (e.g., [`sample_output.json`](../mlruns/850473561665037880/81169bb1fcf84c18aff5324428555d57/artifacts/sample_output.json))
- Response traces for debugging
- Run metadata and parameter snapshots stored in [`meta.yaml`](../mlruns/850473561665037880/81169bb1fcf84c18aff5324428555d57/meta.yaml)

---

## 3. Storage Details

- **Local path:** `./mlruns`
- **Command to launch UI:**

```bash
mlflow ui --backend-store-uri ./mlruns
```

- **Open in browser:** [http://localhost:5000](http://localhost:5000)

---

## 4. Tracked Components

| Category           | Metric / Info                            |
|--------------------|-------------------------------------------|
| Model Performance  | Latency (ms), Tokens generated            |
| Embedding Module   | Embedding time, Max similarity score      |
| Retrieval Module   | Retrieval time, Out-of-scope detection    |
| Query Data         | Intent classification, Query length       |
| System Metrics     | CPU usage, Memory usage, Disk usage       |
| Run Logs & Meta    | [`meta.yaml`](../mlruns/.../meta.yaml)   |

---

## 5. Screenshots

To be added to the README or documentation:

- ✅ MLflow experiment dashboard
- ✅ Individual run view showing params and metrics
- ✅ Side-by-side comparison of different runs

---

## 6. Sample Run Output (Logged Artifact)

```json
{
  "query": "What is a balanced diet?",
  "response": "A balanced diet includes a variety of foods...",
  "context_used": "...FAISS retrieval content...",
  "user_context": {
    "goal": "weight loss"
  }
}
```

📎 [View artifact → sample_output.json](../mlruns/850473561665037880/81169bb1fcf84c18aff5324428555d57/artifacts/sample_output.json)

---

## 7. Future Improvements

- Integrate model registration and versioning
- Sync to remote tracking server (e.g., S3 or Databricks)
- Add visual dashboards for real-time insights
