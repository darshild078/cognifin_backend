import sys

import core.retriever_pipeline as retriever_pipeline
import core.corpus_manager as corpus_manager
import core.cache_utils as cache_utils
import core.metadata_schema as metadata_schema

sys.modules['retriever_pipeline'] = retriever_pipeline
sys.modules['corpus_manager'] = corpus_manager
sys.modules['cache_utils'] = cache_utils
sys.modules['metadata_schema'] = metadata_schema

import os
import re
import time
import shutil
import logging
import tempfile
from uuid import uuid4
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, EmailStr
from asset_manager import ensure_index_cache

# Auth + Database
from db import users_collection, conversations_collection, ensure_indexes
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from bson import ObjectId

from core.cache_utils import has_leftover_tmp, clean_cache

# Our retrieval pipeline
from core.retriever_pipeline import RetrieverPipeline
from core.metadata_schema import RetrievalResult

# Phase 2: Generation layer
from generation.openai_client import OpenAIClient
from generation.prompt_builder import build_context, build_prompt, extract_citations

# Phase 2.5: Corpus architecture
from core.corpus_manager import CorpusManager

# Stage 6: Query orchestration (parse → plan → execute)
from query.query_orchestrator import retrieve_context, intelligent_retrieve
from query.query_understanding import parse_query
from query.search_plan_builder import build_plan

# Stage 8B: Multi-corpus routing
from core.corpus_router import CorpusRouter

# Conversation persistence
from routers.conversations import (
    router as conversations_router,
    create_conversation as db_create_conversation,
    append_to_conversation as db_append_to_conversation,
)

# Phase 1: Retrieval observability
from core.retrieval_logger import log_no_context_event

# Phase 2: Reranking + refinement
from core.reranker import init_reranker, is_reranker_ready
from core.retrieval_pipeline_v2 import refine_results

# Phase 3: Recall improvement (multi-query, hybrid search, expansion)
from core.bm25_retriever import init_bm25, is_bm25_ready
from query.multi_query import init_multi_query, is_multi_query_enabled
from query.query_expander import get_synonym_count

# Phase 4: Intelligent query understanding
from query.intelligent_parser import init_intelligent_parser, is_intelligent_parsing_enabled
from query.context_assembler import assemble_context
from generation.intent_prompts import build_intent_prompt

# Phase 5: Confidence + citation verification
from core.confidence_scorer import compute_confidence
from core.citation_verifier import verify_citations

# Phase 6: Performance (cache + latency)
from core.response_cache import response_cache
from core.latency_tracker import LatencyTracker, latency_stats

# Phase 7: Query logging
from core.query_logger import log_query, get_query_stats, get_recent_logs

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# FOLLOW-UP QUESTION EXTRACTION
# =============================================================================

def extract_follow_ups(answer: str):
    """
    Split [FOLLOW_UP]: lines from the LLM answer.
    Returns (clean_answer, list_of_follow_up_questions).
    """
    pattern = r'\[FOLLOW_UP\]:\s*(.+?)(?=\n|$)'
    follow_ups = re.findall(pattern, answer)
    clean_answer = re.sub(r'\n*\[FOLLOW_UP\]:\s*.+', '', answer).strip()
    return clean_answer, follow_ups[:3]


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class RetrieveRequest(BaseModel):
    """
    Request schema for the /retrieve endpoint.

    Example:
        {"query": "What are the risk factors?"}
    """
    query: str = Field(
        ...,  # ... means required
        description="The question to search for in the document",
        min_length=3,  # At least 3 characters
        examples=["What are the risk factors?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results to return (default: 5)",
        ge=1,  # Greater than or equal to 1
        le=20  # Less than or equal to 20
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from /upload to include uploaded document in search",
    )


class RetrieveResultItem(BaseModel):
    """
    A single retrieval result.
    """
    chunk_id: str = Field(description="Unique identifier for the chunk")
    score: float = Field(description="Similarity score (0-1, higher is better)")
    snippet: str = Field(description="The text content of the chunk")


class RetrieveResponse(BaseModel):
    """
    Response schema for the /retrieve endpoint.

    Example:
        {
            "query": "What are the risk factors?",
            "top_k": 5,
            "results": [
                {"chunk_id": "chunk_12", "score": 0.87, "snippet": "..."},
                ...
            ]
        }
    """
    query: str = Field(description="The original query")
    top_k: int = Field(description="Number of results returned")
    results: List[RetrieveResultItem] = Field(description="Retrieved chunks")
    filtered_count: int = Field(default=0, description="Number of results filtered out below similarity threshold")


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.
    """
    status: str = Field(description="Service status")
    indexed: bool = Field(description="Whether a document has been indexed")
    num_chunks: int = Field(description="Number of chunks in the index")
    generation_ready: bool = Field(description="Whether OpenAI API is configured")


# --- Auth Models ---

class RegisterRequest(BaseModel):
    """Registration request."""
    name: str = Field(..., min_length=2, description="User display name")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Password")


class AuthResponse(BaseModel):
    """Response for register and login."""
    token: str = Field(description="JWT access token")
    user: dict = Field(description="User info (id, name, email)")


# --- Phase 2: Chat Models ---

class ChatRequest(BaseModel):
    """
    Request schema for the /chat endpoint.

    Example:
        {"question": "What are the risk factors?"}
    """
    question: str = Field(
        ...,
        description="The question to ask about the document",
        min_length=3,
        examples=["What are the risk factors?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of evidence chunks to retrieve (default: 5)",
        ge=1,
        le=20
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from /upload to include uploaded document in search",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID to append to. If omitted, a new conversation is created.",
    )


class EvidenceItem(BaseModel):
    """
    A single piece of evidence used to generate the answer.
    """
    chunk_id: str = Field(description="Chunk identifier")
    snippet: str = Field(description="Text content of the chunk")
    page_number: int = Field(default=0, description="1-based page in the source PDF (0 = unknown)")
    document_label: str = Field(default="", description="Human-readable document label, e.g. TCS_DRHP_2024_v1")
    pdf_url: str = Field(default="", description="Basename of the source PDF")


class ChatResponse(BaseModel):
    """
    Response schema for the /chat endpoint.
    """
    answer: str = Field(description="The generated answer grounded in document evidence")
    citations: List[str] = Field(description="Chunk IDs cited in the answer")
    evidence: List[EvidenceItem] = Field(description="Evidence chunks used to generate the answer")
    conversation_id: Optional[str] = Field(default=None, description="MongoDB conversation ID for persistence")
    metadata: dict = Field(default_factory=dict, description="Pipeline metadata: confidence, latency, intent, sources")
    follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")



# --- Stage 8B: Upload Models ---

class UploadResponse(BaseModel):
    """
    Response schema for the /upload endpoint.
    """
    session_id: str = Field(description="Session ID for subsequent queries")
    chunks: int = Field(description="Number of chunks created from the uploaded document")
    company: str = Field(description="Company name used for the upload")
    year: str = Field(description="Year assigned to the upload")
    document_type: str = Field(description="Document type assigned to the upload")


# =============================================================================
# GLOBAL STATE
# =============================================================================

# We use global instances that persist across requests
# These are initialized when the server starts (see lifespan below)
pipeline: Optional[RetrieverPipeline] = None
llm_client: Optional[OpenAIClient] = None

# Phase 2.5: Corpus manager wraps pipeline for metadata-aware retrieval
corpus_manager: Optional[CorpusManager] = None

# Stage 8B: Corpus router for multi-corpus support
corpus_router: Optional[CorpusRouter] = None


# =============================================================================
# LIFESPAN (Startup/Shutdown Events)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.

    Read-only startup: loads pre-ingested corpus from index_cache/.
    Never ingests documents — use ingest.py or batch_ingest_annual_reports.py.

    Load order:
        1. load_index()
        2. load_registry()
        3. validate_cache_integrity()
        4. init_lookup_index()
    """
    global pipeline, llm_client, corpus_manager, corpus_router

    print("\n" + "="*60)
    print("🚀 FinSight AI - Starting Up (Read-Only Mode)")
    print("="*60)

    start_time = time.time()

    cache_dir = os.getenv("INDEX_CACHE_DIR", "index_cache")

    # Initialize the retrieval pipeline
    pipeline = RetrieverPipeline()

    # Initialize the OpenAI client
    llm_client = OpenAIClient()

    # Wrap pipeline in CorpusManager
    corpus_manager = CorpusManager(pipeline)

    # --- Read-only cache load ---
    if not os.path.exists(cache_dir):
        raise RuntimeError(
            f"Cache directory '{cache_dir}' does not exist. "
            f"Run batch_ingest_annual_reports.py first."
        )

    if has_leftover_tmp(cache_dir):
        clean_cache(cache_dir)

    # Step 1: Load FAISS index
    ensure_index_cache()
    if not pipeline.load_index(cache_dir):
        raise RuntimeError(
            f"Failed to load FAISS index from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 2: Load registry
    if not corpus_manager.load_registry(cache_dir):
        raise RuntimeError(
            f"Failed to load document registry from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 3: Validate integrity
    if not corpus_manager.validate_cache_integrity(pipeline.index.ntotal):
        raise RuntimeError(
            "Cache integrity check failed. "
            "Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 4: Initialize LookupIndex
    corpus_manager.init_lookup_index(cache_dir, pipeline.index.ntotal)

    # Stage 8B: Initialize corpus router with global corpus
    corpus_router = CorpusRouter(corpus_manager)

    # Ensure MongoDB indexes
    ensure_indexes()

    # Phase 2: Initialize cross-encoder reranker
    if init_reranker():
        print("🔄 Reranker loaded: " + os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    else:
        print("⚠️  Reranker disabled or failed to load — using FAISS-only ranking")

    # Phase 3: Build BM25 sparse index from existing chunks
    if init_bm25(pipeline.chunks):
        print(f"🔍 BM25 index built: {len(pipeline.chunks)} documents")
    else:
        print("⚠️  BM25 disabled or failed — using dense search only")

    # Phase 3: Initialize multi-query generator
    if init_multi_query(llm_client):
        print(f"📝 Multi-query enabled (synonyms: {get_synonym_count()} groups)")
    else:
        print("⚠️  Multi-query disabled — using single query")

    # Phase 4: Initialize intelligent query parser
    if init_intelligent_parser(llm_client):
        print("🧠 Intelligent query parser enabled")
    else:
        print("⚠️  Intelligent parsing disabled — using rule-based parser")

    elapsed = time.time() - start_time
    print(f"\n✅ Corpus loaded in {elapsed:.2f}s "
          f"({pipeline.index.ntotal} vectors, "
          f"{len(corpus_manager.documents)} documents)")

    print("\n" + "="*60)
    print("📖 Open http://localhost:8000/docs for Swagger UI")
    print("="*60 + "\n")

    # Yield control to the application
    yield

    # Shutdown
    print("\n👋 Shutting down FinSight AI...")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="FinSight AI",
    description="""
## 🔍 Indian Financial Document Analyzer

**Phase 1: Retrieval Pipeline** + **Phase 2: RAG Generation** + **Stage 8B: Upload Support**

This API provides semantic search and AI-powered Q&A over Indian financial documents (SEBI DRHP/RHP, annual reports).

### How it works:
1. A global corpus of NIFTY annual reports is loaded at startup
2. Users can upload additional PDFs for session-scoped retrieval
3. Query `/retrieve` to find relevant passages (evidence only)
4. Query `/chat` for AI-generated answers grounded in the document

### Endpoints:
- **GET /health** — Service health check
- **POST /upload** — Upload a PDF for session-scoped retrieval
- **POST /retrieve** — Semantic retrieval (returns evidence)
- **POST /chat** — RAG Q&A (grounded answer + citations + evidence)
    """,
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allows frontend to call this API)
# Set ALLOWED_ORIGINS in production to restrict access
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware — required by Authlib for OAuth state (CSRF protection)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("JWT_SECRET", "change-me-to-a-strong-random-secret"),
)

# Google OAuth router
from routers.google_auth import router as google_auth_router

# Conversations router (user-scoped CRUD)
app.include_router(conversations_router)
app.include_router(google_auth_router)

# Serve source PDFs at /pdfs/<filename> so the frontend "Open PDF" link works.
# Falls back gracefully if the data directory doesn't exist.
_PDF_DIR = os.path.join(os.path.dirname(__file__), "data")
if os.path.isdir(_PDF_DIR):
    app.mount("/pdfs", StaticFiles(directory=_PDF_DIR), name="pdfs")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if the service is running and a document is indexed."
)
def health_check():
    """
    Health check endpoint.

    Returns:
        - status: "ok" if service is running
        - indexed: True if a document has been indexed
        - num_chunks: Number of chunks in the index

    Use this to verify the service is running before the demo!
    """
    global corpus_manager, llm_client

    is_indexed = corpus_manager is not None and corpus_manager.is_indexed
    num_chunks = corpus_manager.num_chunks if is_indexed else 0
    gen_ready = llm_client is not None and llm_client.is_configured

    return HealthResponse(
        status="ok",
        indexed=is_indexed,
        num_chunks=num_chunks,
        generation_ready=gen_ready
    )


# =============================================================================
# STAGE 8B: UPLOAD ENDPOINT
# =============================================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload PDF",
    description="Upload a PDF document for session-scoped retrieval.",
)
def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    company_name: str = Form(..., description="Company name for the document"),
    year: Optional[str] = Form(default=None, description="Fiscal year (e.g. '2024')"),
    document_type: Optional[str] = Form(
        default=None, description="Document type (default: 'Annual_Report')"
    ),
):
    """
    Upload a PDF for session-scoped retrieval.

    Creates an in-memory corpus that is NOT persisted to disk and does NOT
    modify the global NIFTY corpus. The returned session_id can be passed
    to /retrieve and /chat to include this document in search results.

    Args:
        file: The PDF file to upload
        company_name: Company name (used for scoping and metadata)
        year: Fiscal year (defaults to '2024')
        document_type: Filing type (defaults to 'Annual_Report')

    Returns:
        session_id, chunk count, and metadata used
    """
    global corpus_router

    if corpus_router is None:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. Please wait for startup to complete.",
        )

    # Apply defaults
    year = year or "2024"
    document_type = document_type or "Annual_Report"

    # Save uploaded file to a temp location
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="finsight_upload_")
        tmp_path = os.path.join(tmp_dir, file.filename or "upload.pdf")

        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create isolated pipeline + corpus (shares embedding model weights)
        session_pipeline = RetrieverPipeline()
        session_corpus = CorpusManager(session_pipeline)

        # Ingest into session corpus (in-memory only)
        num_chunks = session_corpus.add_document(
            pdf_path=tmp_path,
            company=company_name,
            document_type=document_type,
            year=year,
        )

        # Generate session ID and register
        session_id = str(uuid4())
        corpus_router.register_session(session_id, session_corpus)

        return UploadResponse(
            session_id=session_id,
            chunks=num_chunks,
            company=company_name,
            year=year,
            document_type=document_type,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}",
        )
    finally:
        # Clean up temp file
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# RETRIEVE ENDPOINT
# =============================================================================

@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Retrieval"],
    summary="Semantic Retrieval",
    description="Find the most relevant chunks for a given query."
)
def retrieve(request: RetrieveRequest):
    """
    Main retrieval endpoint.

    This is the core of FinSight AI Phase 1:
    1. Takes a natural language query
    2. Embeds it using the same model as the document
    3. Finds the top-K most similar chunks
    4. Returns the chunks as "evidence"

    Args:
        request: Contains the query and optional top_k parameter

    Returns:
        - query: The original query (for reference)
        - top_k: Number of results
        - results: List of chunks with scores and snippets

    Raises:
        503: If no document has been indexed yet
    """
    global corpus_manager, corpus_router

    # Check if corpus is ready
    if corpus_manager is None or not corpus_manager.is_indexed:
        raise HTTPException(
            status_code=503,
            detail="No document indexed. Please ensure PDF_PATH is set correctly and restart the server."
        )

    # Get top_k (use request value or default from env)
    top_k = request.top_k if request.top_k is not None else int(os.getenv("TOP_K", 5))

    try:
        final_k = int(os.getenv("FINAL_K", str(top_k)))
        parsed = None

        if request.session_id is not None:
            # Session path: build plan manually, route through corpus_router
            entities = corpus_manager.list_available_entities()
            companies = entities.get("companies", [])
            parsed = parse_query(request.query, companies)
            plan = build_plan(parsed, top_k)
            results = corpus_router.execute_plan(
                plan,
                embed_query=lambda q: pipeline.embed_query(q),
                session_id=request.session_id,
            )
        else:
            # Global-only path: use standard orchestrator
            results, parsed = retrieve_context(
                raw_query=request.query,
                corpus_manager=corpus_manager,
                embed_query=lambda q: pipeline.embed_query(q),
                default_top_k=top_k,
            )

        # Phase 2: Refinement pipeline (rerank → boost → dedup → enrich)
        results = refine_results(
            results=results,
            query=request.query,
            parsed_query=parsed,
            all_chunks=pipeline.chunks,
            chunk_metadata=corpus_manager.chunk_metadata,
            final_k=final_k,
        )

        # Convert to response format
        result_items = [
            RetrieveResultItem(
                chunk_id=r.chunk_id,
                score=round(r.score, 4),  # Round to 4 decimal places
                snippet=r.snippet
            )
            for r in results
        ]

        return RetrieveResponse(
            query=request.query,
            top_k=len(result_items),
            results=result_items,
            filtered_count=top_k - len(result_items) if len(result_items) < top_k else 0,
        )

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {str(e)}"
        )


# =============================================================================
# PHASE 2: CHAT ENDPOINT (RAG Generation)
# =============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Generation"],
    summary="RAG Chat",
    description="Ask a question and get an AI-generated answer grounded in the document."
)
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Phase 2: RAG Chat endpoint — now with JWT auth and MongoDB persistence.

    RAG pipeline is IDENTICAL to before.  After the answer is generated,
    messages are persisted to MongoDB (either new conversation or appended
    to an existing one identified by conversation_id).
    """
    global corpus_manager, llm_client, corpus_router

    user_id = current_user["user_id"]

    # --- Guard: Check corpus ---
    if corpus_manager is None or not corpus_manager.is_indexed:
        raise HTTPException(
            status_code=503,
            detail="No document indexed. Please ensure PDF_PATH is set correctly and restart the server."
        )

    # --- Guard: Check OpenAI client ---
    if llm_client is None or not llm_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file and restart the server."
        )

    # Initialize tracking and defaults
    tracker = LatencyTracker()
    final_k = int(os.getenv("FINAL_K", str(request.top_k or int(os.getenv("TOP_K", 5)))))
    intent = "lookup"
    confidence = 0.0
    parsed = None

    # Get top_k
    top_k = request.top_k if request.top_k is not None else int(os.getenv("TOP_K", 5))

    # --- Response cache: return instantly for repeated queries ---
    cached = response_cache.get(request.question, request.session_id)
    if cached:
        print(f"⚡ Cache HIT for: '{request.question[:60]}...'")
        cached_metadata = cached.get("metadata", {})
        return ChatResponse(
            answer=cached.get("answer", ""),
            citations=cached.get("citations", []),
            evidence=[EvidenceItem(**e) for e in cached.get("evidence", [])],
            conversation_id=cached.get("conversation_id"),
            metadata={**cached_metadata, "cached": True},
            follow_ups=cached.get("follow_ups", []),
        )

    try:
        if request.session_id is not None:
            # Session path: build plan manually, route through corpus_router
            print(f"\n💬 Chat request (session {request.session_id[:8]}...): "
                  f"'{request.question[:60]}...'")
            with tracker.track("retrieval"):
                entities = corpus_manager.list_available_entities()
                companies = entities.get("companies", [])
                parsed = parse_query(request.question, companies)
                plan = build_plan(parsed, top_k)
                results = corpus_router.execute_plan(
                    plan,
                    embed_query=lambda q: pipeline.embed_query(q),
                    session_id=request.session_id,
                )

            with tracker.track("reranking"):
                results = refine_results(
                    results=results,
                    query=request.question,
                    parsed_query=parsed,
                    all_chunks=pipeline.chunks,
                    chunk_metadata=corpus_manager.chunk_metadata,
                    final_k=final_k,
                )

            context, chunk_ids = build_context(results)

        elif is_intelligent_parsing_enabled():
            # Phase 4: Intelligent retrieval pipeline
            print(f"\n🧠 Chat request (Phase 4): '{request.question[:60]}...'")
            with tracker.track("intelligent_retrieve"):
                step_results, iq = intelligent_retrieve(
                    raw_query=request.question,
                    corpus_manager=corpus_manager,
                    embed_query=lambda q: pipeline.embed_query(q),
                    pipeline=pipeline,
                    default_top_k=top_k,
                )

            intent = iq.intent
            parse_method = iq.parse_method
            print(f"   Intent: {iq.intent} | Complexity: {iq.complexity} | "
                  f"Strategy: {iq.retrieval_strategy} | Parse: {iq.parse_method}")

            with tracker.track("context_assembly"):
                context, chunk_ids = assemble_context(step_results, intent=iq.intent)

            results = []
            for step_r in step_results.values():
                results.extend(step_r)

        else:
            # Global-only path: use standard orchestrator
            print(f"\n💬 Chat request: '{request.question[:60]}...'")
            with tracker.track("retrieval"):
                results, parsed = retrieve_context(
                    raw_query=request.question,
                    corpus_manager=corpus_manager,
                    embed_query=lambda q: pipeline.embed_query(q),
                    default_top_k=top_k,
                )
            
            if parsed:
                intent = parsed.get("intent", "lookup") if isinstance(parsed, dict) else getattr(parsed, "intent", "lookup")
            
            with tracker.track("reranking"):
                results = refine_results(
                    results=results,
                    query=request.question,
                    parsed_query=parsed,
                    all_chunks=pipeline.chunks,
                    chunk_metadata=corpus_manager.chunk_metadata,
                    final_k=final_k,
                )

            context, chunk_ids = build_context(results)

        # STEP 3: Build the prompt (system + user message)
        system_prompt, user_message = build_prompt(context, request.question)

        # STEP 4: Generate answer via OpenAI
        print(f"🤖 Generating answer with {llm_client.model}...")
        raw_answer = llm_client.generate(system_prompt, user_message)

        # STEP 4b: Extract follow-up suggestions from LLM output
        answer, follow_ups = extract_follow_ups(raw_answer)

        # STEP 5: Extract citations from the answer
        citations = extract_citations(answer, chunk_ids)

        # STEP 5b: Compute confidence score
        confidence, conf_label = compute_confidence(results, answer, request.question, citations)

        # STEP 6: Build evidence list (with full source citation info)
        evidence = [
            EvidenceItem(
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                page_number=r.page_number,
                document_label=r.document_label,
                pdf_url=r.pdf_url,
            )
            for r in results
        ]
        # ====== END RAG PIPELINE ======

        total_ms = tracker.get_total_ms()
        breakdown = tracker.get_breakdown()
        latency_stats.record(breakdown)

        top_score = max((r.score for r in results), default=0.0)

        print(
            f"✅ Answer generated ({len(citations)} citations, "
            f"intent={intent}, confidence={confidence:.2f}, "
            f"latency={total_ms:.0f}ms)"
        )

        # --- Persist conversation to MongoDB ---
        now_iso = datetime.now(timezone.utc).isoformat()
        user_msg = {
            "role": "user",
            "content": request.question,
            "metadata": {},
            "timestamp": now_iso,
        }
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "metadata": {
                "citations": citations,
                "evidence": [e.model_dump() for e in evidence],
            },
            "timestamp": now_iso,
        }

        conv_id = request.conversation_id
        try:
            if conv_id:
                # Append to existing conversation
                db_append_to_conversation(conv_id, user_id, user_msg, assistant_msg)
            else:
                # Create new conversation (title = first 60 chars of question)
                title = request.question[:60] + ("..." if len(request.question) > 60 else "")
                conv_id = db_create_conversation(user_id, title, user_msg, assistant_msg)
        except Exception as persist_err:
            # Don't fail the request if persistence fails — log and continue
            print(f"⚠️  Conversation persistence error: {persist_err}")

        pipeline_metadata = {
            "confidence": confidence,
            "confidence_label": conf_label,
            "intent": intent,
            "latency_ms": round(total_ms),
            "latency_breakdown": breakdown,
            "sources_used": len(results),
            "top_score": round(top_score, 3),
            "cached": False,
            "model": llm_client.model,
        }

        response_data = ChatResponse(
            answer=answer,
            citations=citations,
            evidence=evidence,
            conversation_id=conv_id,
            metadata=pipeline_metadata,
            follow_ups=follow_ups,
        )

        # --- Cache the successful response ---
        response_cache.set(
            request.question,
            {
                "answer": answer,
                "citations": citations,
                "evidence": [e.model_dump() for e in evidence],
                "metadata": pipeline_metadata,
                "follow_ups": follow_ups,
            },
            request.session_id,
        )

        return response_data

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


@app.get(
    "/",
    tags=["System"],
    summary="Root",
    description="Redirect to documentation."
)
def root():
    """
    Root endpoint - provides a friendly welcome message.
    """
    return {
        "message": "Welcome to FinSight AI!",
        "docs": "Visit /docs for the interactive API documentation",
        "health": "Visit /health to check service status",
        "diagnostics": "Visit /diagnostics for performance monitoring"
    }


# =============================================================================
# RUN DIRECTLY (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run the server
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
