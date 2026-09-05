# 🧠 RAG Pipeline & Retrieval Engine

CogniFin AI implements a **Multi-Stage Hybrid RAG Architecture** specifically optimized for complex, multi-year financial tables, balance sheets, and regulatory filings.

---

## 🔄 End-to-End Retrieval Flow

```text
User Question: "What was Tata Motors FY24 automotive revenue?"
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 1. Query Understanding & Entity Parsing  │
   │    Company: "tata_motors" | Year: "2024" │
   └────────────────────┬────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│ 2A. Dense Search │            │ 2B. BM25 Search  │
│ BGE Embeddings   │            │ Lexical Token    │
│ (768 dimensions) │            │ Matching         │
└────────┬─────────┘            └────────┬─────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
   ┌─────────────────────────────────────────┐
   │ 3. Reciprocal Rank Fusion (RRF) & Merge │
   │    Combines dense and sparse candidates │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 4. Cross-Encoder Reranker               │
   │    BAAI/bge-reranker-base deep scoring  │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 5. Metadata Boosting & Deduplication    │
   │    Boosts matching company/year chunks  │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 6. Grounded Answer Generation           │
   │    Google GenAI SDK (gemini-2.0-flash)  │
   │    Temperature = 0.0                    │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 7. Citation Verification & Confidence   │
   │    Extracts [chunk_id] & page evidence  │
   └─────────────────────────────────────────┘
```

---

## 🔬 Deep Dive into Each Stage

### 1. Dense Vector Search (BGE Embeddings)
- **Model:** `BAAI/bge-base-en-v1.5` (768-dimensional dense vector space).
- **Asymmetric Retrieval:** Queries are prefixed with:
  `Represent this sentence for searching relevant passages:`
  while document chunks are embedded directly to optimize passage matching.
- **Index:** `faiss.IndexFlatIP` (Exact Inner Product Search, mathematically equivalent to Cosine Similarity when normalized).

---

### 2. Sparse Lexical Search (BM25)
- **Algorithm:** `rank-bm25` (BM25Okapi).
- **Purpose:** Dense embeddings excel at semantic concepts but can miss exact financial identifiers (e.g., `"EBITDA"`, `"ISIN"`, `"CIN: L28920MH1945PLC004520"`). BM25 guarantees keyword match precision.
- **Boot Optimization:** The 233,706 chunk token index is pre-serialized in [`index_cache/bm25.pkl`](file:///c:/Projects/finsightai/backend/index_cache/bm25.pkl), reducing startup tokenization latency from 13s down to 2.5s.

---

### 3. Cross-Encoder Reranking
- **Model:** `BAAI/bge-reranker-base`.
- Unlike bi-encoders that compute separate embeddings, the Cross-Encoder processes `(Query, Passage)` simultaneously through full multi-head attention layers, computing deep semantic relevance.

---

### 4. Metadata Boosting & Deduplication
To ensure precision across identical financial metrics across different years:
- **Company Match Boost:** `+0.08`
- **Year Match Boost:** `+0.04`
- **Document Type Boost:** `+0.03`
- **Deduplication:** Filters out passages with $> 0.85$ token overlap to maximize diversity across pages.

---

### 5. Grounded Generation (Google GenAI SDK)
- **Engine:** Google Gemini (`gemini-2.0-flash` or `gemini-1.5-flash`).
- **Deterministic Output:** `temperature = 0.0` ensures the model adheres strictly to the provided context without hallucination.
- **Citation Syntax:** The LLM is instructed to append `[chunk_XXXXX]` after any stated fact.

---

### 6. Citation Verification & Grounding Confidence
The backend runs post-generation validation:
1. **Citation Extraction:** Parses all `[chunk_id]` tags from the response.
2. **Hallucination Detection:** Verifies each cited chunk exists in the retrieved evidence set.
3. **Confidence Score:** Calculates a 0.0–1.0 score based on retrieval similarity, citation coverage, and reranker scores.
