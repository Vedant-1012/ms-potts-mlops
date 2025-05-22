# ⚙️ Configuration Management with Hydra

This project uses [Hydra](https://hydra.cc) for structured, hierarchical configuration with environment-specific overrides.

---

## 📂 Directory Structure

```
src/ms_potts/conf/
├── config.yaml              # Main configuration
├── development/
│   └── config.yaml          # Development overrides
├── production/
│   └── config.yaml          # Production overrides
└── testing/
    └── config.yaml          # Testing overrides
```

---

## 🧠 Features

- Single source of truth (`conf/config.yaml`)
- Environment overrides (e.g., `+mode=development`)
- Command-line overrides for experiments
- Environment variable integration (e.g., `${oc.env:GEMINI_API_KEY}`)

---

## 🧪 Example Usage

### ▶️ Run with development configuration

```bash
python -m src.ms_potts.main +mode=development
```

### 🎯 Override model settings

```bash
python -m src.ms_potts.main model.temperature=0.3 model.max_tokens=2048
```

### 🛠 Change port and log level

```bash
python -m src.ms_potts.main app.port=9000 logging.level=debug
```

---

## ✅ Decorator Usage

```python
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
```

---

## 📘 Key Configuration Values

```yaml
model:
  name: sentence-transformers
  model_path: all-MiniLM-L6-v2
  use_gpu: true
  temperature: 0.2

gemini:
  model_name: gemini-1.5-flash
  max_tokens: 1024

app:
  host: 0.0.0.0
  port: 8080
```

Use Hydra’s flexibility to experiment, switch environments, and keep your code clean!