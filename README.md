# 🎬 LLM Movie Recommendation System

An LLM-powered movie recommendation chatbot built with RAG architecture, featuring production-grade MLOps practices including experiment tracking, CI/CD pipelines, containerization, and automated testing.

## Architecture

```
User Query → Gradio UI → FAISS Retriever → LangChain QA Chain → LLM Response
                              ↑                    ↑
                     SentenceTransformers    Gemma-3-4B-IT
                      (all-MiniLM-L6-v2)        (HF)
                              ↑
                        TMDB 5000 Dataset
```

The system uses **Retrieval-Augmented Generation (RAG)** to ground LLM responses in real movie data:

1. **Preprocessing** — TMDB movie and credits CSVs are merged, cleaned, and enriched with structured fields (genres, cast, director)
2. **Indexing** — Movie documents are embedded using SentenceTransformers and stored in a FAISS vector index
3. **Retrieval** — User queries are embedded and matched against the FAISS index to find the top-k most relevant movies
4. **Generation** — Retrieved context is passed to the LLM (Gemma-3-4B-IT) with a structured prompt to generate personalized recommendations

## Tech Stack

| Layer | Tools |
|---|---|
| LLM | Google Gemma-3-4B-IT via HuggingFace |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Orchestration | LangChain (RetrievalQA, PromptTemplate, ConversationBufferMemory) |
| UI | Gradio |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Code Quality | Ruff, pre-commit hooks |
| Testing | pytest |
| Data | TMDB 5000 Movies + Credits dataset |

## Project Structure

```
LLM_Movie_Recommendation/
├── .github/
│   └── workflows/
│       └── pipeline.yml            # CI/CD: lint → test → build → push
├── src/
│   ├── __init__.py
│   ├── embeddings.py               # Embedding model loading + config
│   ├── retriever.py                # FAISS index loading + retrieval
│   ├── chain.py                    # LangChain QA pipeline
│   ├── preprocessing.py            # TMDB data cleaning + document creation
│   └── monitoring.py               # MLflow tracking (optional, graceful fallback)
├── scripts/
│   ├── __init__.py
│   └── build_index.py              # Rebuild FAISS index from raw data
├── tests/
│   ├── test_config.py              # Config validation
│   ├── test_preprocessing.py       # Data parsing tests
│   └── test_embeddings.py          # Embedding + config tests
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── updated_movies.csv
├── indexes/
│   └── faiss_index_/               # FAISS vector index
├── notebooks/
│   └── LLM_recommendation.ipynb    # Original exploration notebook
├── app.py                          # Gradio application entrypoint
├── config.yaml                     # Centralized configuration
├── Dockerfile                      # Container build
├── docker-compose.yml              # Local dev: app + MLflow
├── docker-compose.prod.yml         # Production deployment
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev/test dependencies
├── conftest.py                     # pytest path configuration
├── ruff.toml                       # Linter configuration
├── .pre-commit-config.yaml         # Pre-commit hooks
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- [HuggingFace account](https://huggingface.co/) with access to [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)

### Installation

```bash
git clone https://github.com/shrsai123/LLM_Movie_Recommendation.git
cd LLM_Movie_Recommendation

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Authenticate with HuggingFace (required for Gemma)
huggingface-cli login
```

### Build the FAISS Index

```bash
python scripts/build_index.py
```

This processes the raw TMDB CSVs, creates embeddings, and saves the FAISS index to `indexes/faiss_index_/`.

### Run the App

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### Run with MLflow Tracking

```bash
# Terminal 1: Start MLflow
python -m mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: Start app
python app.py
```

- App: http://localhost:7860
- MLflow UI: http://localhost:5000

MLflow is optional — the app runs without it and logs a warning.

## Docker

### Local Development

```bash
# Build and run with Docker Compose (app + MLflow)
docker compose up --build
```

### Run Pre-built Image

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/shrsai123/movie-recommender:latest

# Run with local FAISS index mounted
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token_here \
  -v "%cd%\indexes\faiss_index_:/usr/src/app/indexes/faiss_index_" \
  ghcr.io/shrsai123/movie-recommender:latest
```

## MLOps Features

### CI/CD Pipeline

Every push to `main` triggers the GitHub Actions pipeline:

```
Push to main → Lint (ruff) → Test (pytest) → Build Docker → Smoke Test → Push to GHCR
```

The pipeline validates code quality, runs unit tests, builds the Docker image, verifies it starts correctly, and publishes to GitHub Container Registry.

### Experiment Tracking

MLflow tracks every user query with:
- **Query text** — what the user asked
- **Latency** — response time in seconds
- **Response length** — character count of the generated response
- **History length** — conversation turn count

### Code Quality

Pre-commit hooks enforce standards on every commit:
- **Ruff** — linting and import sorting
- **ruff-format** — consistent code formatting
- **trailing-whitespace** — clean file endings
- **check-yaml** — valid YAML configuration
- **check-added-large-files** — prevents accidental large file commits

### Configuration Management

All parameters are centralized in `config.yaml`:

```yaml
model:
  embedding: "all-MiniLM-L6-v2"
  llm: "google/gemma-3-4b-it"
retrieval:
  top_k: 5
  index_path: "indexes/faiss_index_"
  chunk_size: 1000
app:
  host: "0.0.0.0"
  port: 7860
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "movie-recommender"
```

## Development

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Linter

```bash
python -m ruff check src/ app.py scripts/
python -m ruff format src/ app.py scripts/
```

### Evaluate Similar-Movie Rankings

Create one fixed TMDB candidate snapshot, then use a Gemini API judge to generate provisional relevance labels:

```bash
python scripts/evaluate_recommender.py prepare --out evaluation/runs/second
$env:GEMINI_API_KEY="your_api_key"
python scripts/evaluate_recommender.py auto-label --run evaluation/runs/second --provider gemini --model gemini-3.8-flash --passes 2
python scripts/evaluate_recommender.py score --run evaluation/runs/second
```

The judge sees only each source and candidate's title, overview, and genre names. TMDB order and reranker scores are withheld. Gemini structured output uses a JSON Schema for the `relevance` and `reason` fields. `labels.csv` records the LLM label, reason, provider, model, prompt version, and any two-pass disagreement as `needs_review`. Existing non-empty labels are treated as manual overrides and are not overwritten unless `--refresh-llm` is used for labels previously created by the LLM. Review all `needs_review` rows and a random 20â€“30% sample before treating the results as evaluation data.

### Rebuild Index After Data Changes

```bash
python scripts/build_index.py
```

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) containing:
- **tmdb_5000_movies.csv** — titles, overviews, genres, ratings, release dates
- **tmdb_5000_credits.csv** — cast and crew information

## License

MIT
