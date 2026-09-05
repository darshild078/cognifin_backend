"""
search_plan_builder.py
----------------------
Deterministic planner that converts ParsedQuery → SearchPlan.

Layer  : Between query_understanding (Stage 4) and CorpusManager.execute_plan()
Stage  : 5

    raw_query
    → parse_query()  → ParsedQuery
    → build_plan()   → SearchPlan      ← THIS MODULE
    → execute_plan() → retrieval

Owns:
    build_plan()  — pure functional transformer, same input → same plan

Does NOT own:
    Embedding logic        (RetrieverPipeline)
    FAISS search           (RetrieverPipeline)
    Scope resolution       (LookupIndex)
    Plan execution         (CorpusManager.execute_plan)
    Query parsing          (query_understanding)
    Corpus state           (CorpusManager)

Design:
    This module is a PURE PLANNER.
    No I/O. No global state. No corpus inspection. No vector access.
    It only builds retrieval instructions from linguistic structure.
"""

from __future__ import annotations

from .query_understanding import ParsedQuery
from .search_plan import SearchPlan, SubQuery, MergeStrategy
from core.lookup_index import RetrievalScope


# =============================================================================
# PUBLIC API
# =============================================================================

def build_plan(parsed: ParsedQuery, default_top_k: int = 5) -> SearchPlan:
    """
    Convert a ParsedQuery into a SearchPlan.

    Pure function: deterministic, no I/O, no side effects.
    Same input always produces the same plan.

    Args:
        parsed:         Structured query from parse_query().
        default_top_k:  Retrieval budget per SubQuery and final cap.

    Returns:
        SearchPlan ready for CorpusManager.execute_plan().
    """
    if parsed.intent == "single_entity":
        return _plan_single_entity(parsed, default_top_k)

    if parsed.intent == "comparison":
        return _plan_comparison(parsed, default_top_k)

    if parsed.intent == "temporal":
        return _plan_temporal(parsed, default_top_k)

    return _plan_generic(parsed, default_top_k)


# =============================================================================
# INTENT-SPECIFIC PLANNERS
# =============================================================================

def _plan_single_entity(parsed: ParsedQuery, top_k: int) -> SearchPlan:
    label = parsed.companies[0] if parsed.companies else "generic"

    return SearchPlan(
        sub_queries=[
            SubQuery(
                label=label,
                rewritten_query=parsed.cleaned_query,
                scope=RetrievalScope(
                    label=label,
                    companies=list(parsed.companies),
                    doc_types=list(parsed.document_types),
                    years=list(parsed.years),
                    top_k=top_k,
                ),
            ),
        ],
        merge_strategy=MergeStrategy.SINGLE,
        final_top_k=top_k,
    )


def _plan_comparison(parsed: ParsedQuery, top_k: int) -> SearchPlan:
    if not parsed.companies:
        return _plan_generic(parsed, top_k)

    subs = []
    for company in parsed.companies:
        subs.append(
            SubQuery(
                label=company,
                rewritten_query=f"{company} {parsed.cleaned_query}",
                scope=RetrievalScope(
                    label=company,
                    companies=[company],
                    doc_types=list(parsed.document_types),
                    years=list(parsed.years),
                    top_k=top_k,
                ),
            ),
        )

    return SearchPlan(
        sub_queries=subs,
        merge_strategy=MergeStrategy.INTERLEAVED,
        final_top_k=top_k,
    )


def _plan_temporal(parsed: ParsedQuery, top_k: int) -> SearchPlan:
    subs = []
    for year in parsed.years:
        subs.append(
            SubQuery(
                label=year,
                rewritten_query=f"{parsed.cleaned_query} {year}",
                scope=RetrievalScope(
                    label=year,
                    companies=list(parsed.companies),
                    doc_types=list(parsed.document_types),
                    years=[year],
                    top_k=top_k,
                ),
            ),
        )

    return SearchPlan(
        sub_queries=subs,
        merge_strategy=MergeStrategy.SECTIONED,
        final_top_k=top_k,
    )


def _plan_generic(parsed: ParsedQuery, top_k: int) -> SearchPlan:
    return SearchPlan(
        sub_queries=[
            SubQuery(
                label="generic",
                rewritten_query=parsed.cleaned_query,
                scope=RetrievalScope(
                    label="generic",
                    companies=[],
                    doc_types=list(parsed.document_types),
                    years=[],
                    top_k=top_k,
                ),
            ),
        ],
        merge_strategy=MergeStrategy.SINGLE,
        final_top_k=top_k,
    )
