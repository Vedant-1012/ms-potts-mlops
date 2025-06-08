## Phase 3 Continuous Machine Learning (CML) and Deployment

# Ms. Potts MLOps Pipeline

A reproducible MLOps setup for **Ms. Potts**, our AI‐powered nutrition assistant. This repository includes:

- A FastAPI backend (`/src/ms_potts/main.py`)
- A Gradio frontend (`/src/ms_potts/interface.py`)
- DVC‐managed data artifacts
- Unit tests with pytest
- GitHub Actions CI/CD
- Pre-commit linting (ruff & black)
- Dockerfiles for backend & frontend
- Deployment to Google Cloud Run

---

## Repository Structure

- **.github/workflows/ci-cd.yml** — CI/CD pipeline
- **data/** — DVC-tracked datasets
- **src/ms_potts/**
  - **main.py** — FastAPI backend
  - **interface.py** — Gradio frontend
  - **utils/** — utility modules (profiling, monitoring, etc.)
- **tests/**
  - **test_experiment_tracking.py**
  - **test_gemini.py**
- **Dockerfile.backend** — backend Dockerfile
- **Dockerfile.frontend** — frontend Dockerfile
- **requirements.txt** — runtime deps
- **requirements-dev.txt** — dev & CI deps
- **.pre-commit-config.yaml** — pre-commit hook config
- **dvc.yaml** & **.dvc/** — DVC pipeline & metadata
---

## Prerequisites

1. **Local tools**
   - Python 3.10
   - Docker & Docker Compose
   - Git
   - [gcloud](https://cloud.google.com/sdk/docs/install) CLI

2. **GCP setup**
   - Create a Service Account with roles:
     - Artifact Registry Admin
     - Cloud Run Admin
     - Service Account User
   - Store its JSON key in GitHub secret `GCP_SA_KEY`.
   - Add `GCP_PROJECT_ID` and your Gemini API key as `GEMINI_API_KEY`.

---

## Local Development

### 1. Clone & Python env

```bash
git clone https://github.com/your-org/ms-potts-mlops.git
cd ms-potts-mlops
python -m venv .venv && source .venv/bin/activate

```
### 2. Install Dependencies

```bash
# create & activate venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# install runtime requirements
pip install -r requirements.txt

# install dev & CI requirements
pip install -r requirements-dev.txt
```

### 3. Run Linter & Unit Tests

```bash
# lint with Ruff
ruff check .

# run pytest with coverage
pytest --cov=src --cov-report=term-missing
```

### 4. Pull Data with DVC

```bash
# install DVC (if not already)
pip install dvc

# pull remote data artifacts
dvc pull

# verify status
dvc status
```

### 5. Run Locally with Docker
#### 5.1 Build & Run Backend

```bash
docker build -t ms-potts-backend -f Dockerfile.backend .
docker run -d --name backend -e PORT=8080 -p 8080:8080 ms-potts-backend
```
#### 5.2 Build & Run Frontend

```bash
# substitute BACKEND_URL with your backend address if needed
docker build -t ms-potts-frontend -f Dockerfile.frontend .
docker run -d --name frontend \
  -e PORT=7860 \
  -e BACKEND_URL=http://localhost:8080 \
  -p 7860:7860 \
  ms-potts-frontend
```

### 6. CI/CD with GitHub Actions
#### We use a single workflow at .github/workflows/ci-cd.yml that:
1.	CI job
	- 	Checks out code
	-	Sets up Python
	-	Installs requirements-dev.txt + dvc
	-	Runs ruff + pytest
	-	Pulls data via dvc pull
2.	Deploy job (on push to main)
	-	Authenticates to GCP
	-	Builds & pushes backend + frontend Docker images to Artifact Registry
	-	Deploys to Cloud Run (with --memory 1Gi and GEMINI_API_KEY + BACKEND_URL env vars)

### 7. Deploy to Google Cloud Run
#### 7.1    Authenticate & configure

```bash
gcloud auth activate-service-account --key-file path/to/key.json
gcloud config set project $GCP_PROJECT_ID
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```
#### 7.2    Build & push backend

```bash
docker build -t us-central1-docker.pkg.dev/$GCP_PROJECT_ID/dietitian-chatbot-nli/backend:latest \
  -f Dockerfile.backend .
docker push us-central1-docker.pkg.dev/$GCP_PROJECT_ID/dietitian-chatbot-nli/backend:latest
```

#### 7.3    Deploy backend

```bash
gcloud run deploy ms-potts-backend \
  --image us-central1-docker.pkg.dev/$GCP_PROJECT_ID/dietitian-chatbot-nli/backend:latest \
  --platform managed --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY
```
