"""
FinSight AI — Retrieval Pipeline v2 (Phase 2: Refinement Orchestrator)
=======================================================================
Chains the Phase 2 post-retrieval stages into a single pipeline function.

Pipeline:  rerank → boost → deduplicate → enrich → trim to final_k

This module is the single integration point for Phase 2. Endpoints call
refine_results() and get back the highest-quality chunks for the LLM.

Separation of concerns:
    - corpus_manager.py: FAISS search + scope filtering (unchanged)
    - this module:       post-retrieval quality refinement (new)
    - main.py:           endpoint orchestration (calls both)

Author: FinSight AI Team
Phase: 2 (Retrieval Precision Layer)
"""

import os
import logging
from typing import List, Optional

from core.reranker import rerank, is_reranker_ready
from core.result_refiner import boost_by_metadata, deduplicate, enrich_context
from core.retrieval_logger import log_rerank_event

logger = logging.getLogger(__name__)


def refine_results(
    results: list,
    query: str,
    parsed_query = None,
    all_chunks: list = None,
    chunk_metadata: dict = None,
    final_k: int = 5,
) -> list:
    """
    Phase 2 refinement pipeline: rerank → boost → deduplicate → enrich → trim.

    Each stage is independently feature-flagged and gracefully degrades:
        - Reranker disabled → FAISS order preserved
        - No parsed_query → metadata boosting skipped
        - CONTEXT_WINDOW=0 → enrichment skipped

    Args:
        results:         Raw retrieval results from FAISS (post-threshold)
        query:           Original user query string
        parsed_query:    ParsedQuery from query_understanding (for metadata boost)
        all_chunks:      Full chunk list from RetrieverPipeline (for enrichment)
        chunk_metadata:  Dict mapping vector_id → ChunkMetadata (for enrichment)
        final_k:         Number of results to return to LLM

    Returns:
        Refined, reranked, deduplicated list of top final_k RetrievalResult objects
    """
    if not results:
        return results

    pre_rerank_ids = [r.chunk_id for r in results]
    pre_rerank_scores = [r.score for r in results]

    # ── Stage 1: Cross-encoder reranking ──
    # Phase 6: Reranker optimizations for latency
    skip_threshold = float(os.getenv("SKIP_RERANKER_THRESHOLD", "0.85"))
    max_candidates = int(os.getenv("RERANKER_MAX_CANDIDATES", "8"))
    top_score = results[0].score if results else 0

    if is_reranker_ready() and top_score < skip_threshold:
        # Cap candidates to reduce cross-encoder inference time
        candidates = results[:max_candidates]
        remainder = results[max_candidates:]
        logger.debug(
            "Phase 2 Stage 1: Reranking %d/%d candidates (top_score=%.3f < %.2f)",
            len(candidates), len(results), top_score, skip_threshold,
        )
        reranked = rerank(query, candidates)
        # Append non-reranked remainder (lower priority)
        results = reranked + remainder
    elif is_reranker_ready():
        logger.debug(
            "Phase 2 Stage 1: SKIPPED reranker (top_score=%.3f >= %.2f threshold)",
            top_score, skip_threshold,
        )
    else:
        logger.debug("Phase 2 Stage 1: Reranker not available, using FAISS order")

    # ── Stage 2: Metadata-aware boosting ──
    if parsed_query is not None:
        # Extract metadata from ParsedQuery for boosting
        # ParsedQuery fields: companies (list), years (list), document_types (list), intent (str)
        query_company = None
        query_year = None
        query_doc_type = None
        intent = ""

        if hasattr(parsed_query, "companies"):
            companies = parsed_query.companies
            if isinstance(companies, list) and len(companies) == 1:
                query_company = companies[0]
            elif isinstance(companies, str):
                query_company = companies

        if hasattr(parsed_query, "years"):
            years = parsed_query.years
            if isinstance(years, list) and years:
                query_year = str(years[0])
            elif years:
                query_year = str(years)

        if hasattr(parsed_query, "document_types"):
            doc_types = parsed_query.document_types
            if isinstance(doc_types, list) and doc_types:
                query_doc_type = doc_types[0]
            elif isinstance(doc_types, str):
                query_doc_type = doc_types

        if hasattr(parsed_query, "intent"):
            intent = parsed_query.intent or ""

        logger.debug(
            "Phase 2 Stage 2: Metadata boost (company=%s, year=%s, doc=%s, intent=%s)",
            query_company, query_year, query_doc_type, intent,
        )
        results = boost_by_metadata(
            results,
            query_company=query_company,
            query_year=query_year,
            query_doc_type=query_doc_type,
            intent=intent,
        )
    else:
        logger.debug("Phase 2 Stage 2: No parsed_query, skipping metadata boost")

    # ── Stage 3: Deduplication & diversity ──
    logger.debug("Phase 2 Stage 3: Deduplicating %d results...", len(results))
    results = deduplicate(results)

    # ── Stage 4: Context enrichment ──
    if all_chunks and chunk_metadata:
        results = enrich_context(results, all_chunks, chunk_metadata)
    else:
        logger.debug("Phase 2 Stage 4: No chunks/metadata provided, skipping enrichment")

    # ── Trim to final_k ──
    results = results[:final_k]

    # ── Observability: Log rerank comparison ──
    post_rerank_ids = [r.chunk_id for r in results]
    post_rerank_scores = [r.score for r in results]

    log_rerank_event(
        query=query,
        pre_rerank_ids=pre_rerank_ids,
        pre_rerank_scores=pre_rerank_scores,
        post_rerank_ids=post_rerank_ids,
        post_rerank_scores=post_rerank_scores,
    )

    logger.debug(
        "Phase 2 complete: %d candidates → %d final results",
        len(pre_rerank_ids), len(results),
    )

    return results
