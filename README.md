# 📊 CogniFin AI — Backend API & RAG Engine

> **Enterprise-Grade Financial RAG System for Indian Market Disclosures**  
> Built with **FastAPI**, **PyTorch / Hugging Face Transformers**, **FAISS**, and **Google GenAI SDK**.

---

## 🌟 Overview

**CogniFin AI** is a production-grade Retrieval-Augmented Generation (RAG) platform specialized in analyzing Indian financial disclosures, including:
- **SEBI Draft Red Herring Prospectuses (DRHP & RHP)**
- **BSE / NSE Annual Reports & Integrated Reports (FY2020 – FY2025)**
- **Quarterly Earnings Reports & Balance Sheets**

The backend features a **hybrid dense-sparse retrieval pipeline** (`bge-base-en-v1.5` + `BM25Okapi`), a cross-encoder reranker (`bge-reranker-base`), and deterministic grounded answer generation via **Google's Gemini 2.0 Flash SDK** with verified source citations and confidence scoring.

---

## ⚡ Quick Start (3 Steps)

### 1. Clone & Install
```bash
# Clone the repository
git clone https://github.com/darshild078/cognifin_backend.git
cd cognifin_backend

# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows (PowerShell)
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment & Start Database
```bash
# Copy environment configuration
copy .env.example .env

# Start local MongoDB with Docker
docker compose up -d
```
*(Add your `GEMINI_API_KEY` to `.env` from [Google AI Studio](https://aistudio.google.com/apikey)).*

### 3. Run the Server
```bash
uvicorn app.main:app --reload
```
- **API URL:** `http://localhost:8000`
- **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Mongo Express UI:** [http://localhost:8081](http://localhost:8081)

---

## 📚 Complete Documentation Suite (`/docs`)

For detailed walkthroughs and architectural deep dives, see the dedicated guides:

| Guide | Description |
|---|---|
| **[`01_SETUP_AND_INSTALLATION.md`](docs/01_SETUP_AND_INSTALLATION.md)** | Full environment setup, Docker configuration, and troubleshooting. |
| **[`02_ARCHITECTURE_AND_DESIGN.md`](docs/02_ARCHITECTURE_AND_DESIGN.md)** | Layered domain architecture, standard API envelopes, and design principles. |
| **[`03_RAG_PIPELINE_AND_RETRIEVAL.md`](docs/03_RAG_PIPELINE_AND_RETRIEVAL.md)** | Multi-stage search engine (Dense + BM25 + Reranking + Gemini LLM). |
| **[`04_DATASET_AND_INDEX_GENERATION.md`](docs/04_DATASET_AND_INDEX_GENERATION.md)** | Corpus of 139 documents (233K vectors), chunking strategy, and batch ingestion. |
| **[`05_API_REFERENCE.md`](docs/05_API_REFERENCE.md)** | REST endpoints specification, request/response schemas, and Postman guide. |

---

## 📂 Project Architecture

```text
backend/
├── app/
│   ├── api/                 # Thin route controllers (no /v1 prefix)
│   ├── core/                # Config, logging, database, error handlers
│   ├── schemas/             # Pydantic DTOs & response envelopes
│   ├── services/            # Domain service layer (rag_service, auth_service)
│   ├── rag/                 # Vector engine, BM25, reranker, LLM client
│   └── utils/               # Text processing & helpers
├── docs/                    # Complete architectural & operational guides
├── tests/                   # Automated evaluation test suites
├── data/                    # PDF documents
├── index_cache/             # Serialized FAISS (233K vectors) & BM25 indexes
├── docker-compose.yml       # Local MongoDB container definition
├── requirements.txt         # Package dependencies
└── README.md                # This documentation index
```

---

## 🧪 Testing & Evaluation

Run the standalone evaluation framework against the 233,706 vector corpus:

```bash
python tests/evaluation.py
```

---

## 📄 License & Attribution

Developed for financial intelligence and document analysis. All public corporate filings belong to their respective corporate entities (SEBI, BSE, NSE).
