# 🏛️ Architecture & System Design

CogniFin AI is built on a clean, decoupled **Layered Domain Architecture** using **FastAPI**, **PyTorch / Transformers**, and **Google GenAI SDK**.

---

## 🏗️ High-Level System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API (No /v1 prefix)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application Layer                       │
│  ├── Request ID & Access Logging Middleware                           │
│  ├── Session & CORS Middleware                                         │
│  └── Global Error Handlers (AppException, ValidationError, 500)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ Auth Routes  │             │  RAG Routes  │             │ Conv. Routes │
│ (/login, /me)│             │(/chat,/retr.)│             │(/conversat.) │
└──────┬───────┘             └──────┬───────┘             └──────┬───────┘
       │                            │                            │
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│ AuthService  │             │  RagService  │             │ ConvService  │
└──────┬───────┘             └──────┬───────┘             └──────┬───────┘
       │                            │                            │
       │                            ▼                            │
       │                   ┌──────────────────┐                  │
       │                   │  RAG Pipeline    │                  │
       │                   │ ├── Dense (BGE)  │                  │
       │                   │ ├── Sparse (BM25)│                  │
       │                   │ ├── Reranker     │                  │
       │                   │ └── LLMClient    │                  │
       │                   │    (Google GenAI)│                  │
       │                   └────────┬─────────┘                  │
       │                            │                            │
       ▼                            ▼                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Persistence & Cache Layer                         │
│  ├── MongoDB (Users, Conversations) / In-Memory Fallback               │
│  ├── FAISS Vector Store (233,706 vectors)                              │
│  └── Disk Caches (bm25.pkl, lookup_index.json, response_cache)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
backend/
├── app/
│   ├── main.py                  # Application entry point, lifespan, CORS, middleware
│   ├── api/
│   │   ├── router.py            # Aggregated API router (all endpoints at root level)
│   │   ├── deps.py              # JWT authentication & OAuth dependency injectors
│   │   └── routes/              # Route controllers (thin handlers)
│   │       ├── auth.py          # /register, /login, /me, /auth/google, /auth/callback
│   │       ├── chat.py          # /chat (RAG answer generation)
│   │       ├── retrieval.py     # /retrieve (evidence search)
│   │       ├── upload.py        # /upload (dynamic PDF processing)
│   │       ├── conversations.py # /conversations (CRUD chat history)
│   │       └── health.py        # /health (diagnostics & readiness)
│   ├── core/                    # Infrastructure & shared utilities
│   │   ├── config.py            # Pydantic Settings & environment validation
│   │   ├── constants.py         # ErrorCode, IntentType enums
│   │   ├── database.py          # MongoDB client with transparent in-memory fallback
│   │   ├── exceptions.py        # AppException hierarchy
│   │   ├── handlers.py          # Global API envelope error handlers
│   │   ├── logging.py           # Structured single-line logger with Request-ID
│   │   ├── middleware.py        # X-Request-ID and access timing middleware
│   │   └── security.py          # bcrypt password hashing & Jose JWT encoding
│   ├── schemas/                 # Pydantic Data Transfer Objects (DTOs)
│   │   ├── common.py            # ApiResponse[T], ApiErrorResponse envelope
│   │   ├── auth.py              # RegisterRequest, LoginRequest, AuthResponseData
│   │   ├── chat.py              # ChatRequest, ChatResponseData, EvidenceItem
│   │   ├── retrieval.py         # RetrieveRequest, RetrieveResponseData
│   │   ├── conversation.py      # ConversationSummary, ConversationDetail
│   │   └── health.py            # HealthResponseData
│   ├── services/                # Domain Business Logic
│   │   ├── auth_service.py      # Registration, verification, JWT management
│   │   ├── conversation_service.py # Chat persistence and retrieval
│   │   ├── rag_service.py       # Orchestration of RAG search & generation
│   │   └── upload_service.py    # PDF parsing and ephemeral session indexing
│   ├── rag/                     # Core RAG Algorithms & Retrieval Engine
│   │   ├── llm_client.py        # Unified Google GenAI SDK wrapper
│   │   ├── retriever_pipeline.py# BGE embedding model & FAISS index
│   │   ├── corpus_manager.py    # Document catalog & integrity verification
│   │   ├── corpus_router.py     # Session vs. Global corpus routing
│   │   ├── bm25_retriever.py    # BM25 sparse lexical search & pickle cache
│   │   ├── reranker.py          # BGE Cross-Encoder reranker
│   │   ├── intelligent_parser.py# Structured intent parser
│   │   ├── multi_query.py       # Query expansion generator
│   │   ├── prompt_builder.py    # Grounded prompt synthesis & citation extraction
│   │   ├── confidence_scorer.py # Grounding confidence metrics
│   │   └── response_cache.py    # In-memory query result cache
│   └── utils/                   # Helpers
│       ├── text.py              # String normalization & cleaning
│       └── follow_up.py         # Dynamic follow-up question extractor
├── tests/                       # Automated evaluation suites
├── data/                        # Raw PDF source documents
├── index_cache/                 # Pre-computed index files (FAISS, BM25, Registry)
├── docker-compose.yml           # Local MongoDB & Mongo Express containers
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation index
```

---

## 🛡️ Core Design Principles

### 1. Thin Controllers, Thick Services
Route handlers in `app/api/routes/` contain zero database queries or algorithmic logic. They parse requests, delegate execution to `app/services/`, and return standard response envelopes.

### 2. Standardized Response Envelope (`ApiResponse[T]`)
Every single API endpoint returns a uniform JSON envelope:

```json
{
  "success": true,
  "message": "Passages retrieved successfully.",
  "data": { ... }
}
```

On errors, the system returns:

```json
{
  "success": false,
  "error": {
    "code": "AUTH_FAILED",
    "message": "Invalid email or password.",
    "details": null
  }
}
```

### 3. Structured Single-Line Logging
Every log entry includes timestamp, level, thread-local request ID (`req_id`), module name, and structured key-values:
```text
2026-09-05 14:01:44 [INFO] [req_id=e7e904cd] cognifin.access: method=POST path=/login status=200 duration=298.5ms
```

### 4. Zero Raw HTTP Calls (100% SDK Driven)
All external communications are mediated through official SDKs:
- **Google GenAI SDK** for LLM generation.
- **PyMongo / Motor** for MongoDB.
- **Faiss** for vector similarity.
- **Sentence-Transformers & CrossEncoder** for embeddings.
