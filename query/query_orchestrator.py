"""
FinSight AI — Query Orchestrator (Stage 6)
==========================================
Pure orchestration layer that connects:
    parse_query  →  build_plan  →  CorpusManager.execute_plan

Phase 2: Returns (results, parsed_query) tuple for metadata-aware boosting.
Phase 3: Adds query expansion, multi-query generation, and hybrid search (BM25).
Phase 4: Adds intelligent LLM parsing → multi-step retrieval → context assembly.

Pipeline (Phase 4):
    raw_query
    → llm_parse_query()          (LLM-based, 6 intents)
    → plan_execution()           (intent → retrieval steps)
    → per step (PARALLEL):
        → expand_query()         (financial synonyms)
        → FAISS + BM25           (hybrid search)
        → RRF merge              (per-step)
        → refine_results()       (rerank + dedup)
    → assemble_context()         (labeled sections)
    → return (step_results, intelligent_query)
"""

import os
import logging
import time
import asyncio
from typing import Callable, List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .query_understanding import parse_query, ParsedQuery
from .search_plan_builder import build_plan
from .search_plan import SearchPlan
from .query_expander import expand_query
from .multi_query import generate_multi_queries, is_multi_query_enabled
from .intelligent_parser import (
    IntelligentQuery, llm_parse_query, is_intelligent_parsing_enabled,
)
from .execution_planner import plan_execution, RetrievalStep
from .context_assembler import assemble_context
from core.corpus_manager import CorpusManager
from core.metadata_schema import RetrievalResult
from core.bm25_retriever import bm25_search, is_bm25_ready
from core.result_merger import merge_results
from core.retrieval_pipeline_v2 import refine_results

logger = logging.getLogger(__name__)

# Thread pool for parallel step execution
_executor = ThreadPoolExecutor(max_workers=4)


def retrieve_context(
    raw_query: str,
    corpus_manager: CorpusManager,
    embed_query: Callable[[str], np.ndarray],
    default_top_k: int = 5,
) -> Tuple[List[RetrievalResult], ParsedQuery]:
    """
    Orchestrate the full Phase 3 retrieval pipeline.

    Flow:
        1. Parse query (extract company, year, intent)
        2. Expand query with financial synonyms
        3. Generate multi-query variants (LLM)
        4. For each variant: FAISS + BM25 search
        5. Merge all result lists via RRF
        6. Return (merged_results, parsed_query)

    Returns:
        Tuple of (results, parsed_query) — results are the merged
        candidate pool for the Phase 2 refinement pipeline.
    """
    # ── Step 1: Parse query (unchanged from Phase 2) ──
    entities = corpus_manager.list_available_entities()
    companies = entities.get("companies", [])
    parsed = parse_query(raw_query, companies)

    # ── Step 2: Expand query with financial synonyms ──
    expanded_query = expand_query(parsed.cleaned_query)
    logger.debug("Phase 3: expanded query: '%s'", expanded_query[:100])

    # ── Step 3: Generate multi-query variants ──
    if is_multi_query_enabled():
        query_variants = generate_multi_queries(expanded_query)
    else:
        query_variants = [expanded_query]

    logger.debug("Phase 3: %d query variants: %s", len(query_variants), query_variants)

    # ── Step 4: Retrieve for each variant (FAISS + BM25) ──
    retrieval_k = int(os.getenv("RETRIEVAL_K", "20"))
    all_result_lists: List[List[RetrievalResult]] = []
    all_labels: List[str] = []

    for i, variant in enumerate(query_variants):
        # 4a: FAISS dense search (via existing plan execution)
        plan = build_plan(parsed, default_top_k)

        # For comparison/temporal intents, build_plan creates per-entity
        # sub-queries with company/year prefixes (e.g., "WIPRO {query}").
        # Preserve those prefixes — only replace the core query portion.
        if parsed.intent in ("comparison", "temporal") and len(plan.sub_queries) > 1:
            for sq in plan.sub_queries:
                # Keep the entity label prefix, replace the query body
                sq.rewritten_query = f"{sq.label} {variant}"
        else:
            for sq in plan.sub_queries:
                sq.rewritten_query = variant

        faiss_results = corpus_manager.execute_plan(plan, embed_query)
        if faiss_results:
            all_result_lists.append(faiss_results)
            all_labels.append(f"dense_q{i}")

        # 4b: BM25 sparse search (if enabled)
        if is_bm25_ready():
            # Collect allowed_ranges from ALL sub-queries (not just the first)
            # so comparison queries search both companies
            all_ranges = []
            for sq in plan.sub_queries:
                ranges, _ = corpus_manager.lookup_index.resolve_ranges(sq.scope)
                if ranges:
                    all_ranges.extend(ranges)
            allowed_ranges = all_ranges if all_ranges else None

            bm25_hits = bm25_search(
                query=variant,
                k=retrieval_k,
                allowed_ranges=allowed_ranges,
            )

            # Convert BM25 hits to RetrievalResult objects
            bm25_results = _bm25_to_retrieval_results(
                bm25_hits, corpus_manager
            )
            if bm25_results:
                all_result_lists.append(bm25_results)
                all_labels.append(f"bm25_q{i}")

    # ── Step 5: Merge all result lists via RRF ──
    if len(all_result_lists) > 1:
        merged = merge_results(
            result_lists=all_result_lists,
            labels=all_labels,
            top_k=retrieval_k,
        )
    elif all_result_lists:
        merged = all_result_lists[0][:retrieval_k]
    else:
        merged = []

    logger.debug(
        "Phase 3 complete: %d variants × %d search types → %d lists → %d merged",
        len(query_variants),
        2 if is_bm25_ready() else 1,
        len(all_result_lists),
        len(merged),
    )

    return merged, parsed


def _bm25_to_retrieval_results(
    bm25_hits: List[Tuple[float, int]],
    corpus_manager: CorpusManager,
) -> List[RetrievalResult]:
    """
    Convert BM25 (score, vector_id) hits to RetrievalResult objects.

    Mirrors the conversion logic in CorpusManager.search() so BM25
    results are compatible with the refinement pipeline.
    """
    results = []
    for score, vector_id in bm25_hits:
        if vector_id >= len(corpus_manager.retriever.chunks):
            continue

        chunk = corpus_manager.retriever.chunks[vector_id]

        # Look up document label from chunk metadata
        meta = corpus_manager.chunk_metadata.get(vector_id)
        doc_label = meta.document_label if meta else "unknown"

        # Resolve PDF path for the /pdfs/ static route (mirrors corpus_manager.search()).
        # pdf_path may be a Colab path, so extract last 2 segments (COMPANY/YEAR.pdf).
        pdf_filename = ""
        for rec in corpus_manager.documents.values():
            if rec.vector_id_start <= vector_id < rec.vector_id_end:
                parts = rec.pdf_path.replace("\\", "/").rstrip("/").split("/")
                if len(parts) >= 2:
                    pdf_filename = f"{parts[-2]}/{parts[-1]}"
                else:
                    pdf_filename = parts[-1] if parts else ""
                break

        result = RetrievalResult(
            chunk_id=chunk.chunk_id,
            score=score,
            snippet=chunk.text,
            document_label=doc_label,
            page_number=chunk.page_number,
            pdf_filename=pdf_filename,
        )
        results.append(result)

    return results


# =============================================================================
# PHASE 4: INTELLIGENT RETRIEVAL
# =============================================================================

def intelligent_retrieve(
    raw_query: str,
    corpus_manager: CorpusManager,
    embed_query: Callable[[str], np.ndarray],
    pipeline=None,
    default_top_k: int = 5,
) -> Tuple[Dict[str, List[RetrievalResult]], IntelligentQuery]:
    """
    Phase 4 intelligent retrieval pipeline.

    Flow:
        1. LLM-parse query → IntelligentQuery (6 intents, metrics, strategy)
        2. Plan execution → List[RetrievalStep]
        3. Execute steps in PARALLEL (each step uses Phase 3 pipeline)
        4. Refine per-step results (Phase 2: rerank + dedup)
        5. Return step_results dict for context assembly

    Performance optimizations:
        - Multi-query DISABLED for multi-step queries (latency control)
        - Parallel step execution via ThreadPoolExecutor
        - Reduced retrieval_k for multi-step (MULTI_STEP_RETRIEVAL_K)
        - Reranker sees fewer candidates per step

    Fallback:
        On ANY failure, falls back to Phase 3 retrieve_context().

    Args:
        raw_query:       User's original question
        corpus_manager:  Loaded CorpusManager with FAISS index
        embed_query:     Embedding function from pipeline
        pipeline:        RetrieverPipeline (for chunks/metadata access)
        default_top_k:   Results per step

    Returns:
        Tuple of:
        - step_results: Dict[label → List[RetrievalResult]]
        - iq: IntelligentQuery with parsing details
    """
    start = time.time()

    # ── Step 1: LLM Parse ──
    entities = corpus_manager.list_available_entities()
    known_companies = entities.get("companies", [])
    iq = llm_parse_query(raw_query, known_companies)

    logger.info(
        "Phase 4: intent=%s, complexity=%s, strategy=%s, "
        "companies=%s, metrics=%s (parse: %s, %.0fms)",
        iq.intent, iq.complexity, iq.retrieval_strategy,
        iq.companies, iq.metrics, iq.parse_method, iq.parse_time_ms,
    )

    # ── Step 2: Plan Execution ──
    steps = plan_execution(iq)
    is_multi_step = len(steps) > 1

    logger.debug("Phase 4: %d retrieval steps planned", len(steps))

    # ── Step 3: Execute Steps (parallel for multi-step) ──
    final_k = int(os.getenv("FINAL_K", str(default_top_k)))

    def _execute_step(step: RetrievalStep) -> Tuple[str, List[RetrievalResult]]:
        """Execute a single retrieval step using Phase 3 pipeline."""
        try:
            step_results = _run_step_retrieval(
                step=step,
                corpus_manager=corpus_manager,
                embed_query=embed_query,
                pipeline=pipeline,
                is_multi_step=is_multi_step,
                final_k=final_k,
            )
            return step.label, step_results
        except Exception as e:
            logger.error("Step '%s' failed: %s", step.label, e)
            return step.label, []

    # Parallel execution for multi-step, sequential for single
    if is_multi_step and len(steps) > 1:
        futures = [_executor.submit(_execute_step, step) for step in steps]
        step_results_dict = {}
        for future in futures:
            label, results = future.result(timeout=30)
            step_results_dict[label] = results
    else:
        step_results_dict = {}
        for step in steps:
            label, results = _execute_step(step)
            step_results_dict[label] = results

    total_chunks = sum(len(r) for r in step_results_dict.values())
    elapsed = time.time() - start

    logger.info(
        "Phase 4 complete: %d steps → %d total chunks in %.2fs",
        len(steps), total_chunks, elapsed,
    )

    return step_results_dict, iq


def _run_step_retrieval(
    step: RetrievalStep,
    corpus_manager: CorpusManager,
    embed_query: Callable[[str], np.ndarray],
    pipeline,
    is_multi_step: bool,
    final_k: int,
) -> List[RetrievalResult]:
    """
    Execute a single retrieval step using the Phase 3 pipeline.

    For multi-step queries:
        - Multi-query generation is DISABLED (latency control)
        - Uses reduced retrieval_k (MULTI_STEP_RETRIEVAL_K)

    For single-step queries:
        - Full Phase 3 pipeline (multi-query + BM25 + RRF)
    """
    retrieval_k = step.retrieval_k

    # ── Step A: Expand query with financial synonyms ──
    expanded = expand_query(step.query)

    # ── Step B: Query variants ──
    # Multi-query DISABLED for multi-step (latency control)
    if is_multi_step or not is_multi_query_enabled():
        query_variants = [expanded]
    else:
        query_variants = generate_multi_queries(expanded)

    # ── Step C: Build scope from step filters ──
    from core.lookup_index import RetrievalScope
    scope = RetrievalScope(
        label=step.label,
        companies=step.companies,
        doc_types=step.doc_types,
        years=step.years,
        top_k=retrieval_k,
    )

    # ── Step D: Retrieve per variant (FAISS + BM25) ──
    all_result_lists: List[List[RetrievalResult]] = []
    all_labels: List[str] = []

    for i, variant in enumerate(query_variants):
        # FAISS dense search
        vector = embed_query(variant)
        faiss_results = corpus_manager.search(scope, vector)
        if faiss_results:
            all_result_lists.append(faiss_results)
            all_labels.append(f"dense_{step.label}_q{i}")

        # BM25 sparse search
        if is_bm25_ready():
            ranges, _ = corpus_manager.lookup_index.resolve_ranges(scope)
            bm25_hits = bm25_search(
                query=variant,
                k=retrieval_k,
                allowed_ranges=ranges if ranges else None,
            )
            bm25_results = _bm25_to_retrieval_results(bm25_hits, corpus_manager)
            if bm25_results:
                all_result_lists.append(bm25_results)
                all_labels.append(f"bm25_{step.label}_q{i}")

    # ── Step E: Merge via RRF ──
    if len(all_result_lists) > 1:
        merged = merge_results(all_result_lists, all_labels, top_k=retrieval_k)
    elif all_result_lists:
        merged = all_result_lists[0][:retrieval_k]
    else:
        merged = []

    # ── Step F: Phase 2 refinement (rerank + boost + dedup) ──
    # Build a ParsedQuery-compatible object for the refiner
    from .query_understanding import ParsedQuery
    parsed_for_refine = ParsedQuery(
        cleaned_query=step.query,
        companies=step.companies,
        years=step.years,
        document_types=step.doc_types,
        intent="single_entity" if len(step.companies) <= 1 else "comparison",
    )

    refined = refine_results(
        results=merged,
        query=step.query,
        parsed_query=parsed_for_refine,
        all_chunks=pipeline.chunks if pipeline else None,
        chunk_metadata=corpus_manager.chunk_metadata,
        final_k=final_k,
    )

    return refined
