# 🔌 REST API Reference & Postman Guide

All API routes in CogniFin AI operate at the root resource level (**no `/v1` prefix**) and follow the standard `ApiResponse[T]` envelope specification.

---

## 📑 Endpoints Summary

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | **`/health`** | Server and vector index health status | No |
| `POST` | **`/register`** | Register a new user account | No |
| `POST` | **`/login`** | Authenticate user & receive JWT token | No |
| `GET` | **`/me`** | Retrieve authenticated user profile | Bearer JWT |
| `GET` | **`/auth/google`** | Initiate Google OAuth 2.0 flow | No |
| `GET` | **`/auth/callback`** | Complete Google OAuth & redirect | No |
| `POST` | **`/chat`** | Submit question & generate grounded answer | Optional (Session / JWT) |
| `POST` | **`/retrieve`** | Retrieve raw matching passages/evidence | No |
| `POST` | **`/upload`** | Upload & index an ephemeral PDF session | Bearer JWT |
| `GET` | **`/conversations`** | List user chat conversations | Bearer JWT |
| `GET` | **`/conversations/{id}`** | Get full message history of a chat | Bearer JWT |
| `PATCH` | **`/conversations/{id}`** | Rename conversation title | Bearer JWT |
| `DELETE` | **`/conversations/{id}`** | Delete conversation | Bearer JWT |

---

## 🔍 Detailed Endpoint Documentation

### 1. Health & Diagnostics
#### `GET /health`
Returns service readiness and corpus statistics.

**Response `(200 OK)`:**
```json
{
  "success": true,
  "message": "Service is healthy.",
  "data": {
    "status": "ok",
    "indexed": true,
    "num_chunks": 233706,
    "generation_ready": true
  }
}
```

---

### 2. Authentication
#### `POST /register`
Creates a new local user account.

**Request:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "SecurePassword123!"
}
```

**Response `(201 Created)`:**
```json
{
  "success": true,
  "message": "Account registered successfully.",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "6a9bd370766f90fef2ab535a",
      "name": "Jane Doe",
      "email": "jane@example.com",
      "profile_picture": ""
    }
  }
}
```

#### `POST /login`
Authenticates a user with email and password.

**Request:**
```json
{
  "email": "jane@example.com",
  "password": "SecurePassword123!"
}
```

**Response `(200 OK)`:**
```json
{
  "success": true,
  "message": "Login successful.",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "6a9bd370766f90fef2ab535a",
      "name": "Jane Doe",
      "email": "jane@example.com",
      "profile_picture": ""
    }
  }
}
```

---

### 3. RAG Search & Generation
#### `POST /chat`
Generates a grounded financial answer from the corpus with source citations.

**Request:**
```json
{
  "question": "What is Tata Motors total revenue in FY24?",
  "top_k": 5,
  "conversation_id": null
}
```

**Response `(200 OK)`:**
```json
{
  "success": true,
  "message": "Answer generated successfully.",
  "data": {
    "answer": "Tata Motors reported a total automotive revenue of ₹4,30,104 crores in FY24, representing a 1.4% change [chunk_219646]...",
    "citations": [
      {
        "citation_id": 1,
        "chunk_id": "chunk_219646",
        "score": 0.7304,
        "document_label": "Tata Motors FY24 Annual Report",
        "page_number": 42
      }
    ],
    "evidence": [
      {
        "chunk_id": "chunk_219646",
        "snippet": "Automotive operations are our most significant segment...",
        "page_number": 42,
        "document_label": "Tata Motors FY24 Annual Report",
        "pdf_url": "/pdfs/Tata_Motors_2024.pdf"
      }
    ],
    "conversation_id": "6a9bd362882ab2e47ec7ed90",
    "metadata": {
      "confidence": 0.94,
      "confidence_label": "High",
      "intent": "lookup",
      "latency_ms": 1120,
      "sources_used": 5,
      "model": "gemini-2.0-flash"
    },
    "follow_ups": [
      "What was Jaguar Land Rover's contribution to Tata Motors revenue?",
      "How did electric vehicle sales perform in FY24?"
    ]
  }
}
```

#### `POST /retrieve`
Raw passage retrieval without LLM generation.

**Request:**
```json
{
  "query": "Reliance Industries retail revenue growth",
  "top_k": 5
}
```

---

### 4. Postman Collection

The complete Postman Collection is located directly in the `docs/` folder:
- File path: [`docs/cognifin_api.postman_collection.json`](file:///c:/Projects/finsightai/backend/docs/cognifin_api.postman_collection.json)

#### How to import into Postman:
1. Open Postman.
2. Click **Import** in the top left.
3. Select or drag-and-drop [`docs/cognifin_api.postman_collection.json`](file:///c:/Projects/finsightai/backend/docs/cognifin_api.postman_collection.json).
4. The collection is organized into 4 folders:
   - **System:** Health Check (`GET /health`)
   - **Authentication:** Register, Login, Get Profile (`GET /me`), Google OAuth
   - **RAG & Retrieval:** Grounded Chat (`POST /chat`), Raw Retrieval (`POST /retrieve`), PDF Upload (`POST /upload`)
   - **Conversations:** List, Get History, Rename, and Delete conversations
5. Pre-configured environment variables:
   - `baseUrl`: `http://localhost:8000`
   - `token`: Automatically captured from `/login` and `/register` responses
   - `conversationId`: Captured from chat requests
