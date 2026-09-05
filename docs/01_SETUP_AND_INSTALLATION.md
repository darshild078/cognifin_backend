# 🚀 Setup & Installation Guide

This guide walks you through setting up, configuring, and running the **CogniFin AI Backend** from scratch.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your machine:

- **Python 3.10+** (Python 3.11 recommended)
- **Git**
- **Docker & Docker Desktop** (for running local MongoDB)
- **Google Gemini API Key** (Free from [Google AI Studio](https://aistudio.google.com/apikey))

---

## 🛠️ Step-by-Step Installation

### Step 1: Clone the Repository

```bash
# Clone the backend repository
git clone https://github.com/darshild078/cognifin_backend.git
cd cognifin_backend
```

---

### Step 2: Create and Activate Virtual Environment

```bash
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\activate

# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables (`.env`)

Copy the example configuration file:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# Server & CORS
FRONTEND_URL=http://localhost:5173
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Document & Index Cache
PDF_PATH=data/sample.pdf
INDEX_CACHE_DIR=index_cache
ASSET_MODE=local

# Google Gemini LLM (Paste your API key here)
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Database (Local Docker MongoDB)
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=finsightai

# Authentication & JWT
JWT_SECRET=cognifin-jwt-secret-key-32chars-long!
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback

# Retrieval & Search Settings
RETRIEVAL_K=10
FINAL_K=5
TOP_K=5
SIMILARITY_THRESHOLD=0.30
RERANKER_ENABLED=true
BM25_ENABLED=true
CACHE_ENABLED=true
```

---

### Step 5: Start Local MongoDB with Docker

Start the isolated MongoDB and Mongo Express dashboard using Docker Compose:

```bash
docker compose up -d
```

- **MongoDB Server:** `localhost:27017`
- **Mongo Express Web UI:** [http://localhost:8081](http://localhost:8081)

*(Note: If Docker is offline, the backend automatically operates with an in-memory fallback store so registration, login, and retrieval continue without crashing).*

---

### Step 6: Verify `index_cache`

Ensure the precomputed vector index exists in `index_cache/`:

```text
backend/index_cache/
├── faiss.index             # 718 MB (233,706 vectors)
├── chunks.pkl              # 247 MB (Raw chunk texts)
├── chunk_metadata.pkl      # 22.8 MB (Company, Year, Page metadata)
├── document_registry.json  # 51 KB (139 documents catalog)
├── bm25.pkl                # 244 MB (Pre-computed sparse index)
└── lookup_index.json       # 28 KB (Inverted document index)
```

---

### Step 7: Run the FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

Once started, you will see:
```text
2026-09-05 [INFO] cognifin: Initializing CogniFin AI application...
2026-09-05 [INFO] cognifin.db: MongoDB connected and indexes verified.
2026-09-05 [INFO] app.rag.retriever_pipeline: Loaded index from cache: 233706 chunks
2026-09-05 [INFO] app.rag.corpus_manager: Registry loaded: 139 document(s), 233706 vectors
2026-09-05 [INFO] app.rag.bm25_retriever: 📂 BM25 index loaded from disk: 233706 documents in 2.56s
2026-09-05 [INFO] cognifin.rag.llm: LLM Client initialized with Google GenAI SDK (model: gemini-2.0-flash)
2026-09-05 [INFO] cognifin: CogniFin AI application ready.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 🔍 Interactive API Documentation

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** `curl http://localhost:8000/health`
