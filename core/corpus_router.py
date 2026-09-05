"""
FinSight AI — Corpus Router (Stage 8B)
=======================================
Routes retrieval plans across global and session-scoped corpora.

Responsibilities:
    - Hold reference to global (read-only) CorpusManager
    - Maintain session corpora (in-memory, never persisted)
    - Route execute_plan() to correct corpus(es)
    - Merge dual-corpus results preserving FAISS rank order

Does NOT:
    - Perform ingestion
    - Perform I/O or persistence
    - Import FastAPI or HTTP concerns
    - Modify global corpus state

Design:
    When session_id is None, delegate entirely to global_corpus.
    When session_id is present, run each SubQuery against both
    global and session corpora, merge per-SubQuery results by
    score (preserving FAISS ranking authority), then apply the
    plan's merge_strategy exactly as CorpusManager.execute_plan() does.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace as dc_replace
from typing import Callable, Dict, List, Optional

import numpy as np

from .corpus_manager import CorpusManager
from .metadata_schema import RetrievalResult
from query.search_plan import SearchPlan, MergeStrategy

logger = logging.getLogger(__name__)


class CorpusRouter:
    """
    Routes SearchPlan execution across global and session corpora.

    Global corpus is loaded once at startup (read-only).
    Session corpora are created per-upload and live in memory only.
    """

    def __init__(self, global_corpus: CorpusManager):
        self.global_corpus: CorpusManager = global_corpus
        self.session_corpora: Dict[str, CorpusManager] = {}

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def register_session(self, session_id: str, corpus: CorpusManager) -> None:
        """Register an in-memory session corpus."""
        self.session_corpora[session_id] = corpus
        logger.info(
            "Session registered: %s (%d vectors)",
            session_id, corpus.num_chunks,
        )

    def remove_session(self, session_id: str) -> None:
        """Remove a session corpus, freeing its memory."""
        if session_id in self.session_corpora:
            del self.session_corpora[session_id]
            logger.info("Session removed: %s", session_id)

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.session_corpora

    # =========================================================================
    # PLAN EXECUTION
    # =========================================================================

    def execute_plan(
        self,
        plan: SearchPlan,
        embed_query: Callable[[str], np.ndarray],
        session_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute a SearchPlan, routing to the correct corpus(es).

        If session_id is None:
            → delegate entirely to global_corpus.execute_plan()

        If session_id is present:
            → run each SubQuery against BOTH global and session corpora
            → merge per-SubQuery results by score (FAISS order preserved)
            → apply plan.merge_strategy across SubQueries
        """
        # --- Global-only path (no session) ---
        if session_id is None:
            return self.global_corpus.execute_plan(plan, embed_query)

        # --- Dual-corpus path ---
        if session_id not in self.session_corpora:
            raise KeyError(
                f"Session '{session_id}' not found. "
                f"Upload a document first via POST /upload."
            )

        session_corpus = self.session_corpora[session_id]

        # Run each SubQuery against both corpora and merge by score
        per_subquery: List[List[RetrievalResult]] = []

        for sub_query in plan.sub_queries:
            vector = embed_query(sub_query.rewritten_query)

            # Global results
            global_results = self.global_corpus.search(
                sub_query.scope, vector,
            )

            # Session results
            session_results = session_corpus.search(
                sub_query.scope, vector,
            )

            # Merge by score (descending = higher similarity first)
            # This preserves FAISS ranking authority within each source
            merged = _merge_by_score(global_results, session_results)
            # Phase 2: Use RETRIEVAL_K so reranker gets full candidate set
            merge_limit = int(os.getenv("RETRIEVAL_K", str(sub_query.scope.top_k)))
            merged = merged[:merge_limit]

            per_subquery.append(merged)

            logger.debug(
                "execute_plan [%s]: global=%d, session=%d, merged=%d",
                sub_query.label,
                len(global_results),
                len(session_results),
                len(merged),
            )

        # Apply merge strategy across SubQueries
        return _apply_merge_strategy(plan, per_subquery)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _merge_by_score(
    a: List[RetrievalResult],
    b: List[RetrievalResult],
) -> List[RetrievalResult]:
    """
    Merge two result lists by score (descending).

    Both inputs are already sorted by FAISS distance (descending similarity).
    Two-pointer merge preserves relative order within each source.
    """
    merged: List[RetrievalResult] = []
    i, j = 0, 0

    while i < len(a) and j < len(b):
        if a[i].score >= b[j].score:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1

    # Append remainder
    merged.extend(a[i:])
    merged.extend(b[j:])

    return merged


def _apply_merge_strategy(
    plan: SearchPlan,
    per_subquery: List[List[RetrievalResult]],
) -> List[RetrievalResult]:
    """
    Apply the plan's merge strategy across SubQuery result lists.

    Phase 2: Uses RETRIEVAL_K as merge limit (not final_top_k) so the
    reranker receives the full candidate set. Final trim to FINAL_K
    happens in refine_results().
    """
    merge_limit = int(os.getenv("RETRIEVAL_K", str(plan.final_top_k)))

    if plan.merge_strategy == MergeStrategy.SINGLE:
        return per_subquery[0][:merge_limit]

    if plan.merge_strategy == MergeStrategy.INTERLEAVED:
        merged: List[RetrievalResult] = []
        round_idx = 0
        while len(merged) < merge_limit:
            added_this_round = False
            for results in per_subquery:
                if round_idx < len(results):
                    merged.append(results[round_idx])
                    added_this_round = True
                    if len(merged) >= merge_limit:
                        break
            if not added_this_round:
                break
            round_idx += 1
        return merged

    if plan.merge_strategy == MergeStrategy.SECTIONED:
        merged = []
        for sub_query, results in zip(plan.sub_queries, per_subquery):
            for r in results:
                if len(merged) >= merge_limit:
                    break
                labeled = dc_replace(
                    r,
                    chunk_id=f"[{sub_query.label}]:{r.chunk_id}",
                )
                merged.append(labeled)
            if len(merged) >= merge_limit:
                break
        return merged

    logger.error(
        "Unhandled MergeStrategy '%s' — returning [].",
        plan.merge_strategy,
    )
    return []
