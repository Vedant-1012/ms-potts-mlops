# 📘 Logging

Ms. Potts uses an advanced logging system powered by a custom `EnhancedLogger` for structured and human-readable logs.

---

## ✅ Logger: EnhancedLogger

**Location**: `utils/enhanced_logging.py`  
**Used In**: `main.py`, `GeminiModel`, and FastAPI endpoints

### Features:
- Console + file logging
- Optional JSON output
- Rich formatting
- Timestamped logs

---

## 🗂️ Log Storage

- **Directory**: `./logs/`
- **Format**: `ms_potts_YYYY-MM-DD.log`
- **Rotation**: Daily log files
- **Enabled JSON Output**: ✅ (`json_output: true`)

---

## 📝 Sample Log Entries

### ➤ Query Log (Info)

```json
{
  "timestamp": "2025-05-22T17:42:15.123Z",
  "level": "INFO",
  "name": "ms_potts",
  "message": "Processing query request",
  "module": "main",
  "function": "query_endpoint",
  "line": 78,
  "request_id": "req-1716394935",
  "query": "What foods are high in vitamin C?",
  "user_context_keys": ["age", "weight", "dietary_restrictions"]
}
```

### ➤ Error Log (Exception)

```json
{
  "timestamp": "2025-05-22T17:45:32.456Z",
  "level": "ERROR",
  "name": "ms_potts",
  "message": "Exception in query endpoint: Failed to connect to Gemini API",
  "module": "main",
  "function": "query_endpoint",
  "line": 112,
  "request_id": "req-1716395132",
  "exception": "ConnectionError: Failed to connect to Gemini API: Connection timed out"
}
```

---

## 🛠️ How to Interpret

- **request_id**: Correlates logs for a single request
- **user_context_keys**: Lists what user info was passed
- **query**: Shows the original question
- **exception**: Captures the failure reason in detail