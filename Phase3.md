## Phase 3 Continuous Machine Learning (CML) and Deployment

# DEMO:
[GCP APP LINK](https://ms-potts-frontend-1033579299449.us-central1.run.app)

## Step-1 User Profile Creation
![User Peofile](images/UserProfile.png)

## Step-2 User query for Nutrition Advice
![Nutrtion Advise Response](images/Response.png)

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
```markdown
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # CI/CD pipeline for automated builds and deployment
├── data/                          # DVC-tracked datasets
│   ├── raw/                       # Original, immutable data
│   ├── interim/                   # Intermediate transformed data
│   └── processed/                 # Final datasets used for modeling or inference
├── src/
│   └── ms_potts/
│       ├── main.py                # FastAPI backend entrypoint
│       ├── interface.py           # Gradio frontend UI
│       └── utils/                 # Utility modules (profiling, logging, monitoring, etc.)
│           ├── enhanced_logging.py
│           ├── monitoring.py
│           ├── profiling.py
│           └── …
├── tests/
│   ├── test_experiment_tracking.py
│   └── test_gemini.py
├── Dockerfile.backend            # Dockerfile for backend (FastAPI)
├── Dockerfile.frontend           # Dockerfile for frontend (Gradio)
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Development and CI dependencies
├── .pre-commit-config.yaml       # Pre-commit hooks for linting, formatting, etc.
├── dvc.yaml                      # DVC pipeline definition
└── .dvc/                         # DVC metadata and cache tracking
```

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

![GCP Setup](images/GCPSetup.png)
---

## Local Development

### 1. Clone & Python env

```bash
git clone https://github.com/your-org/ms-potts-mlops.git
cd ms-potts-mlops
python -m venv .venv && source .venv/bin/activate

```
### 2. Install runtime deps

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install dev deps + DVC

```bash
pip install -r requirements-dev.txt
pip install dvc
```

### 4a. Pull Data with DVC
.pre-commit-config.yaml:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.4.1
    hooks:
      - id: ruff
        language_version: python3.10
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```
```bash
pre-commit install
pre-commit run --all-files
```

### 4b. Linting & Testing
```bash
ruff check .
pytest --cov=src --cov-report=term-missing
```
![Ruff Check](images/ruff.png)

![Pytest Execution](images/pytest.png)

### 5. Dockerization
#### Backend

Dockerfile.backend
```bash
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "ms_potts.main:app", "--host", "0.0.0.0", "--port", "8080"]


FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "ms_potts.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build & run locally
```bash
docker build -t ms-potts-backend:local -f Dockerfile.backend .
docker run -d --name backend -p 8080:8080 ms-potts-backend:local
curl http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Hello?", "context":{"user_profile":{}}}'
```

#### Frontend
Dockerfile.frontend
```bash
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Create any needed directories
RUN mkdir -p data/raw data/processed data/interim data/external \
    models \
    reports/figures

# Expose the port your app will run on
EXPOSE 8080

# Launch the Gradio interface; it reads PORT and defaults to 7860 locally.
CMD ["python", "src/ms_potts/interface.py"]

```
Build & run locally
```bash
docker build -t ms-potts-frontend:local -f Dockerfile.frontend .
docker network create potts-net
docker run -d \
  --name backend --network potts-net \
  -e PORT=8080 -p 8080:8080 \
  ms-potts-backend:local

docker run --network potts-net \
  -e PORT=8080 -p 7860:7860 \
  ms-potts-frontend:local
# then browse http://localhost:7860
```


### 6. CI/CD with GitHub Actions
#### We use a single workflow at .github/workflows/ci.yml that:
```yml
name: CI/CD – build, push, deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ------------------------------------------------------------------- #
  # 1️⃣ CI – Lint, unit-tests, DVC                                      #
  # ------------------------------------------------------------------- #
  ci:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dev dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt dvc

      - name: Lint & run tests
        run: |
          ruff check .
          pytest --cov=src --cov-report=term-missing

      - name: Pull data with DVC
        run: |
          dvc pull
          dvc status

  # ------------------------------------------------------------------- #
  # 2️⃣ Deploy – build images, push to AR, deploy to Cloud Run           #
  # ------------------------------------------------------------------- #
  deploy:
    needs: ci
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: "${{ secrets.GCP_SA_KEY }}"

      - name: Configure GCP project
        run: gcloud config set project "${{ secrets.GCP_PROJECT_ID }}"

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

      - name: Build & push backend image
        run: |
          SHA_TAG="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/dietitian-chatbot-nli/backend:${{ github.sha }}"
          LATEST_TAG="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/dietitian-chatbot-nli/backend:latest"

          docker build -t "$SHA_TAG" -f Dockerfile.backend .
          docker tag "$SHA_TAG" "$LATEST_TAG"
          docker push "$SHA_TAG"
          docker push "$LATEST_TAG"

          echo "BACKEND_IMAGE=$SHA_TAG" >> $GITHUB_ENV

      - name: Deploy backend to Cloud Run
        run: |
          gcloud run deploy ms-potts-backend \
            --image "$BACKEND_IMAGE" \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated \
            --memory 2Gi \
            --timeout 900s \
            --set-env-vars GEMINI_API_KEY="${{ secrets.GEMINI_API_KEY }}"

      - name: Fetch backend URL
        run: |
          echo "BACKEND_URL=$(gcloud run services describe ms-potts-backend \
            --platform managed --region us-central1 \
            --format='value(status.url)')" >> $GITHUB_ENV

      - name: Build & push frontend image
        run: |
          SHA_TAG="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/dietitian-chatbot-nli/frontend:${{ github.sha }}"
          LATEST_TAG="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/dietitian-chatbot-nli/frontend:latest"

          docker build -t "$SHA_TAG" -f Dockerfile.frontend .
          docker tag "$SHA_TAG" "$LATEST_TAG"
          docker push "$SHA_TAG"
          docker push "$LATEST_TAG"

          echo "FRONTEND_IMAGE=$SHA_TAG" >> $GITHUB_ENV

      - name: Deploy frontend to Cloud Run
        run: |
          gcloud run deploy ms-potts-frontend \
            --image "$FRONTEND_IMAGE" \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated \
            --memory 1Gi \
            --timeout 300s \
            --set-env-vars GEMINI_API_KEY="${{ secrets.GEMINI_API_KEY }}",BACKEND_URL="${{ env.BACKEND_URL }}"


```

### 7. GCP Setup & Secrets

1. **Service Account**
   - Create a service account:
     `github-ci-cd@dietitian-chatbot-nli.iam.gserviceaccount.com`
   - Grant the following roles to it:
     - `roles/run.admin`
     - `roles/iam.serviceAccountUser`
     - `roles/artifactregistry.admin`
   - Create and download a JSON key for this account.
   - In your GitHub repo **Settings → Secrets → Actions**, add a new secret:
     - **Name:** `GCP_SA_KEY`
     - **Value:** _Paste the entire JSON key contents here_

2. **Gemini API Key**
   - Obtain your Gemini (Google Generative AI) API key.
   - In your GitHub repo **Settings → Secrets → Actions**, add:
     - **Name:** `GEMINI_API_KEY`
     - **Value:** _Your Gemini API key_

### 8. Retrieve Service URLs

```bash
gcloud run services describe ms-potts-backend \
  --platform managed --region us-central1 \
  --format="value(status.url)"

gcloud run services describe ms-potts-frontend \
  --platform managed --region us-central1 \
  --format="value(status.url)"
```
