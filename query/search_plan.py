"""
search_plan.py
--------------
Data structures for multi-scope retrieval planning.

Layer  : Layer 3 — Query Intelligence
Stage  : 3

Owns:
    MergeStrategy  — how results from multiple SubQueries are combined
    SubQuery       — a single scoped retrieval instruction with a query string
    SearchPlan     — an ordered list of SubQueries + merge configuration

Does NOT own:
    Embedding logic         (RetrieverPipeline)
    FAISS search            (RetrieverPipeline)
    Scope resolution        (LookupIndex)
    Plan execution          (CorpusManager.execute_plan)

Design:
    SearchPlan is a pure data container.
    CorpusManager.execute_plan() is the sole executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from core.lookup_index import RetrievalScope


# =============================================================================
# MERGE STRATEGY
# =============================================================================

class MergeStrategy(Enum):
    """
    How results from multiple SubQueries are combined into a final list.

    SINGLE      — Plan has exactly one SubQuery. Return its results directly,
                  trimmed to final_top_k. No merging required.

    INTERLEAVED — Round-robin across all SubQuery result lists until
                  final_top_k is reached or all lists are exhausted.
                  Preserves relative score ordering within each SubQuery.

    SECTIONED   — Concatenate SubQuery results in declaration order.
                  Each result's chunk_id is prefixed with [label]: to
                  preserve provenance. Trim to final_top_k.
    """
    SINGLE      = "single"
    INTERLEAVED = "interleaved"
    SECTIONED   = "sectioned"


# =============================================================================
# SUB QUERY
# =============================================================================

@dataclass
class SubQuery:
    """
    A single scoped retrieval instruction.

    Paired with a RetrievalScope that constrains which FAISS vectors
    are eligible, and a rewritten_query string that is embedded at
    execute_plan() time.

    Fields:
        label            — human-readable tag used in logging and
                           SECTIONED chunk_id prefixes
        rewritten_query  — the query string passed to embed_query()
        scope            — RetrievalScope controlling which documents
                           are eligible for this sub-query
    """
    label:            str
    rewritten_query:  str
    scope:            RetrievalScope


# =============================================================================
# SEARCH PLAN
# =============================================================================

@dataclass
class SearchPlan:
    """
    An ordered execution plan for multi-scope retrieval.

    Consumed exclusively by CorpusManager.execute_plan().

    Fields:
        sub_queries      — ordered list of SubQuery instructions;
                           executed sequentially, one embed+search per entry
        merge_strategy   — how per-SubQuery result lists are combined
        final_top_k      — total result count cap after merging
    """
    sub_queries:      List[SubQuery]
    merge_strategy:   MergeStrategy
    final_top_k:      int
