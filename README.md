# 🔍 FinSight AI - Indian Financial Document Analyzer

> **Phase 1: Retrieval Pipeline**

A semantic search system for Indian financial documents (SEBI DRHP/RHP, Annual Reports).

## 📋 What This Does

This system finds **relevant evidence** from financial documents based on natural language queries.

```
Query: "What are the risk factors?"
       ↓
   [Retrieval Pipeline]
       ↓
Result: Top 5 most relevant paragraphs from the document
```

**Phase 1 returns EVIDENCE, not answers.** The LLM generation layer comes in Phase 2.

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- A financial PDF document (DRHP, RHP, or Annual Report)

### 2. Setup

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
# Copy environment template
copy .env.example .env    # Windows
# OR
cp .env.example .env      # Mac/Linux

# Edit .env and set your PDF path
# PDF_PATH=data/your_document.pdf
```

### 4. Add Your PDF

Place your financial PDF in the `data/` folder:
```
backend/
  data/
    sample.pdf  ← Your PDF here
```

### 5. Run the Server

```bash
# Start the server
uvicorn main:app --reload

# Server runs at: http://localhost:8000
# Swagger UI at:  http://localhost:8000/docs
```

### 6. Test It

Open http://localhost:8000/docs in your browser and try:

1. **GET /health** - Check if service is running
2. **POST /retrieve** - Search with a query like:
   ```json
   {"query": "What are the risk factors?"}
   ```

---

## 📁 Project Structure

```
backend/
├── main.py                  # FastAPI server (entry point)
├── core/                    # Core RAG engine & storage
│   ├── retriever_pipeline.py # Embedding & FAISS logic
│   ├── corpus_manager.py     # Document state management
│   ├── lookup_index.py       # Precomputed inverted index
│   └── ...
├── query/                   # Query understanding & planning
│   ├── query_orchestrator.py # parse → plan → execute
│   ├── query_understanding.py
│   └── search_plan_builder.py
├── generation/              # LLM generation layer
│   ├── openai_client.py
│   └── prompt_builder.py
├── ingestion/               # CLI tools for data loading
│   └── ingest.py            # Manual ingestion script
├── tests/                   # Unit & evaluation suites
├── data/                    # Raw PDF source files
├── index_cache/             # Persisted FAISS index & metadata
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔌 API Endpoints

### GET /health

Check if the service is running.

**Response:**
```json
{
  "status": "ok",
  "indexed": true,
  "num_chunks": 245
}
```

### POST /retrieve

Find relevant evidence for a query.

**Request:**
```json
{
  "query": "What are the risk factors?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What are the risk factors?",
  "top_k": 5,
  "results": [
    {
      "chunk_id": "chunk_12",
      "score": 0.8732,
      "snippet": "The Company faces several risk factors including regulatory changes..."
    },
    {
      "chunk_id": "chunk_47",
      "score": 0.8156,
      "snippet": "Market volatility poses significant risks to our revenue streams..."
    }
  ]
}
```

---

## 🎤 Faculty Demo Script

### What to Show

1. **Start the Server**
   ```bash
   uvicorn main:app --reload
   ```
   Wait for: "✅ Server ready! Indexed X chunks"

2. **Open Swagger UI**
   - Go to http://localhost:8000/docs
   - Show the interactive documentation

3. **Test Health Endpoint**
   - Click GET /health → Try it out → Execute
   - Show: `{"status": "ok", "indexed": true}`

4. **Demo Retrieval**
   - Click POST /retrieve → Try it out
   - Enter query: `"What are the risk factors?"`
   - Execute and show the results

### What to Say

> "This is FinSight AI, a semantic search system for financial documents.
> 
> In Phase 1, we've built the **retrieval pipeline** - the most critical component of any RAG system.
> 
> Let me show you how it works:
> 1. We load a SEBI DRHP document
> 2. Split it into chunks of 500 characters
> 3. Convert each chunk into a 384-dimensional embedding vector
> 4. Store these in a FAISS vector database
> 
> When a user asks a question, we:
> 1. Convert the question into an embedding
> 2. Find the top-5 most similar chunks
> 3. Return these as **evidence**
> 
> This is the **R** in RAG - Retrieval. In Phase 2, we'll add the **G** - Generation using an LLM."

### 5 Demo Queries to Try

1. `"What are the risk factors?"`
2. `"Tell me about the company's revenue"`
3. `"Who are the promoters and their background?"`
4. `"What is the issue size and price band?"`
5. `"What are the objects of the issue?"`

---

## 📝 Viva Notes (Key Concepts)

### 1. What are Embeddings?

**Simple Answer:**
> Embeddings are numerical representations of text that capture meaning. Similar texts have similar embeddings.

**Technical Answer:**
> An embedding is a dense vector (array of numbers) that represents text in a high-dimensional space. The model `all-MiniLM-L6-v2` produces 384-dimensional vectors. Semantically similar texts are closer in this vector space (measured by cosine similarity).

**Example:**
```
"risk" → [0.2, 0.8, -0.1, ...]
"danger" → [0.21, 0.79, -0.12, ...]  ← Very close (similar meaning)
"profit" → [-0.5, 0.3, 0.9, ...]     ← Far away (different meaning)
```

### 2. Why Use a Vector Database (FAISS)?

**Simple Answer:**
> FAISS is like a super-fast search engine for embeddings. It can find the most similar vectors in milliseconds, even with millions of documents.

**Technical Answer:**
> FAISS (Facebook AI Similarity Search) uses specialized data structures and algorithms (like IVF, HNSW) for approximate nearest neighbor search. For our use case with `IndexFlatIP`, it performs exact inner product search, which equals cosine similarity when vectors are normalized.

**Why not just loop through all vectors?**
> With 1000 chunks, a loop takes 1000 comparisons. With 1 million chunks, FAISS can still find results in milliseconds using indexing structures.

### 3. Why Does Chunking Matter?

**Simple Answer:**
> Documents are too long for embedding models. Chunking breaks them into smaller pieces that can be individually embedded and retrieved.

**Technical Answer:**
> Embedding models have token limits (~512 tokens for most). Also, smaller chunks = more precise retrieval. If a user asks about "risk factors," returning a 500-character chunk about risks is more useful than returning the entire 50-page document.

**Overlap prevents context loss:**
```
Without overlap: "...regulatory risks" | "which may impact revenue..."
With overlap:    "...regulatory risks which may" | "risks which may impact revenue..."
```

### 4. What is Top-K Retrieval?

**Simple Answer:**
> Return the K most similar results instead of just one. This increases the chance of finding relevant information.

**Technical Answer:**
> Top-K retrieval returns the K nearest neighbors in the embedding space. We use K=5 by default. The results are sorted by similarity score (highest first). Multiple results help because:
> - Different chunks may contain different aspects of the answer
> - Allows reranking in later stages
> - Provides redundancy if one result is noisy

### 5. How Does This Become RAG in Phase 2?

**Current System (Phase 1):**
```
Query → Retrieve Top-K Chunks → Return Evidence
```

**RAG System (Phase 2):**
```
Query → Retrieve Top-K Chunks → Combine with Query → Send to LLM → Generate Answer
```

**The prompt to the LLM would look like:**
```
Based on the following evidence from the document:

[Chunk 1]: "The company faces regulatory risks..."
[Chunk 2]: "Market volatility may impact..."

Answer this question: "What are the risk factors?"
```

**Why is retrieval the most important part?**
> If you retrieve wrong chunks, even GPT-4 will give wrong answers. Good retrieval = good RAG. This is why we focus on retrieval in Phase 1.

---

## 🔧 Troubleshooting

### "PDF not found" Error
- Ensure your PDF is in the `data/` folder
- Check the `PDF_PATH` in your `.env` file

### "Model download taking long"
- First run downloads ~80MB model
- This is cached for future runs

### "Out of memory" Error
- For large PDFs (>200 pages), increase chunk size
- Or process fewer pages

### "No results" or poor results
- Try different query phrasings
- Check if the PDF has extractable text (not scanned images)

---

## 🗺️ Roadmap

- [x] **Phase 1:** Retrieval Pipeline (Current)
- [ ] **Phase 2:** LLM Answer Generation
- [ ] **Phase 3:** Multi-document Support
- [ ] **Phase 4:** Frontend UI
- [ ] **Phase 5:** Deployment

---

## 📄 License

This project is for educational purposes (Final Year Project).

---

## 👨‍💻 Author

FinSight AI Team
