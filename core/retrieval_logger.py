"""
FinSight AI — Retrieval Logger (Phase 1: Observability)
========================================================
Structured logging for retrieval diagnostics.

Captures the full retrieval pipeline context per request:
    query → embedding → search → filtering → results

Output format: Structured JSON logs (parseable by any log aggregator).
Log level: DEBUG by default, controllable via LOG_LEVEL env var.

Integration points:
    - Called from CorpusManager.search() after score filtering
    - Called from main.py /chat and /retrieve endpoints for request metadata

Author: FinSight AI Team
Phase: 1 (Core Retrieval Fixes — Observability)
"""

import os
import json
import time
import logging
from typing import List, Optional
from dataclasses import asdict

logger = logging.getLogger("finsight.retrieval")


# =============================================================================
# LOG LEVEL CONFIGURATION
# =============================================================================

def configure_retrieval_logging() -> None:
    """
    Configure retrieval logger based on LOG_LEVEL env var.

    Defaults to DEBUG so retrieval diagnostics are always available
    during Phase 1 tuning. Set LOG_LEVEL=INFO in production to reduce noise.
    """
    level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    logger.setLevel(level)

    # Add a handler if none exist (avoid duplicate handlers on reload)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# =============================================================================
# RETRIEVAL EVENT LOGGING
# =============================================================================

def log_retrieval_event(
    query: str,
    results: list,
    filtered_count: int = 0,
    threshold: float = 0.30,
    latency_ms: float = 0.0,
    scope_label: str = "",
    intent: str = "",
    embedding_model: str = "",
) -> None:
    """
    Log a structured retrieval event with full diagnostic context.

    Args:
        query:           The user's query (original or cleaned)
        results:         List of RetrievalResult objects that passed filtering
        filtered_count:  Number of results removed by score threshold
        threshold:       Similarity threshold used for filtering
        latency_ms:      Total retrieval latency (embed + search + filter)
        scope_label:     Scope label from RetrievalScope (e.g. company name)
        intent:          Query intent from ParsedQuery
        embedding_model: Name of the embedding model used
    """
    event = {
        "query": query[:200],  # Truncate very long queries
        "intent": intent,
        "scope": scope_label,
        "result_count": len(results),
        "filtered_count": filtered_count,
        "threshold": threshold,
        "latency_ms": round(latency_ms, 2),
        "embedding_model": embedding_model or os.getenv("EMBEDDING_MODEL", "unknown"),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
                "snippet_preview": r.snippet[:100].replace("\n", " "),
                "document_label": getattr(r, "document_label", ""),
                "page_number": getattr(r, "page_number", 0),
            }
            for r in results
        ],
    }

    logger.debug("RETRIEVAL_EVENT: %s", json.dumps(event, ensure_ascii=False))


def log_no_context_event(
    query: str,
    threshold: float = 0.30,
    scope_label: str = "",
) -> None:
    """
    Log when no results pass the similarity threshold.

    This is a significant event — it means the LLM will receive
    a no-context refusal instead of generating an answer.

    Args:
        query:       The user's query
        threshold:   The threshold that filtered out all results
        scope_label: Scope context
    """
    event = {
        "query": query[:200],
        "scope": scope_label,
        "threshold": threshold,
        "action": "no_context_refusal",
    }

    logger.info("NO_CONTEXT_EVENT: %s", json.dumps(event, ensure_ascii=False))


# =============================================================================
# PHASE 2: RERANK EVENT LOGGING
# =============================================================================

def log_rerank_event(
    query: str,
    pre_rerank_ids: list,
    pre_rerank_scores: list,
    post_rerank_ids: list,
    post_rerank_scores: list,
) -> None:
    """
    Log a Phase 2 reranking comparison event.

    Captures the full before/after state of the refinement pipeline:
    - Pre-rerank: FAISS order and cosine similarity scores
    - Post-rerank: Final order after rerank + boost + dedup + enrich
    - Promotions: chunks that moved up ≥2 positions
    - Demotions: chunks dropped from the final set

    Args:
        query:              The user's query
        pre_rerank_ids:     Chunk IDs in FAISS order (before refinement)
        pre_rerank_scores:  FAISS scores (before refinement)
        post_rerank_ids:    Chunk IDs after full refinement pipeline
        post_rerank_scores: Final scores after refinement
    """
    # Build position maps
    pre_positions = {cid: i for i, cid in enumerate(pre_rerank_ids)}

    # Identify promotions (moved up ≥2 positions) and dropped chunks
    promotions = []
    for i, cid in enumerate(post_rerank_ids):
        old_pos = pre_positions.get(cid)
        if old_pos is not None and old_pos - i >= 2:
            promotions.append({
                "chunk_id": cid,
                "old_position": old_pos,
                "new_position": i,
                "delta": old_pos - i,
            })

    dropped = [cid for cid in pre_rerank_ids if cid not in set(post_rerank_ids)]

    event = {
        "query": query[:200],
        "pre_rerank_count": len(pre_rerank_ids),
        "post_rerank_count": len(post_rerank_ids),
        "pre_rerank_top5": [
            {"id": cid, "score": round(s, 4)}
            for cid, s in zip(pre_rerank_ids[:5], pre_rerank_scores[:5])
        ],
        "post_rerank_top5": [
            {"id": cid, "score": round(s, 4)}
            for cid, s in zip(post_rerank_ids[:5], post_rerank_scores[:5])
        ],
        "promotions": promotions[:5],
        "dropped_count": len(dropped),
        "dropped_ids": dropped[:10],
    }

    logger.debug("RERANK_EVENT: %s", json.dumps(event, ensure_ascii=False))


# =============================================================================
# PHASE 3: MULTI-QUERY + HYBRID EVENT LOGGING
# =============================================================================

def log_multi_query_event(
    original_query: str,
    expanded_query: str,
    query_variants: list,
    per_variant_counts: list,
    bm25_enabled: bool = False,
) -> None:
    """
    Log a Phase 3 multi-query retrieval event.

    Captures query expansion, variant generation, and per-variant
    hit counts for debugging recall improvements.

    Args:
        original_query:     User's raw query
        expanded_query:     Query after synonym expansion
        query_variants:     All query variants (including original)
        per_variant_counts: List of dicts with hit counts per variant
        bm25_enabled:       Whether BM25 hybrid search was active
    """
    event = {
        "original_query": original_query[:200],
        "expanded_query": expanded_query[:200],
        "variant_count": len(query_variants),
        "variants": [v[:100] for v in query_variants],
        "per_variant": per_variant_counts,
        "bm25_enabled": bm25_enabled,
    }

    logger.info("MULTI_QUERY_EVENT: %s", json.dumps(event, ensure_ascii=False))


# =============================================================================
# TIMING CONTEXT MANAGER
# =============================================================================

class RetrievalTimer:
    """
    Simple context manager for measuring retrieval latency.

    Usage:
        with RetrievalTimer() as timer:
            results = do_retrieval()
        print(f"Took {timer.elapsed_ms:.1f}ms")
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


# Initialize logging on import
configure_retrieval_logging()

