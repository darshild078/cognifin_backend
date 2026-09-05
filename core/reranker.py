"""
FinSight AI — Cross-Encoder Reranker (Phase 2: Retrieval Precision)
====================================================================
Reranks FAISS retrieval candidates using a cross-encoder model for 
higher precision results.

Architecture:
    Bi-encoder (FAISS) retrieves Top-N candidates (fast, approximate).
    Cross-encoder (this module) rescores each (query, chunk) pair
    for precise semantic relevance (slower, accurate).

Model: BAAI/bge-reranker-base (110M params, ~200ms for 20 pairs on CPU)
    - Same BGE family as our embedding model (bge-base-en-v1.5)
    - Trained with compatible objectives for two-stage retrieval
    - Loaded via sentence-transformers CrossEncoder class (no new deps)

Feature Flag: RERANKER_ENABLED (env var) — set to 'false' for Phase 1 fallback.

Author: FinSight AI Team
Phase: 2 (Retrieval Precision Layer)
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global singleton — initialized once at startup via init_reranker()
_reranker_model = None
_reranker_ready = False


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_reranker() -> bool:
    """
    Load the cross-encoder reranker model.

    Called once during application startup (main.py lifespan).
    Model is stored as a module-level singleton for thread-safe inference.

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _reranker_model, _reranker_ready

    if not os.getenv("RERANKER_ENABLED", "true").lower() == "true":
        logger.info("Reranker disabled (RERANKER_ENABLED != true)")
        return False

    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s ...", model_name)
        _reranker_model = CrossEncoder(model_name)
        _reranker_ready = True
        logger.info("Reranker loaded successfully: %s", model_name)
        return True

    except Exception as e:
        logger.error("Failed to load reranker model '%s': %s", model_name, e)
        _reranker_ready = False
        return False


def is_reranker_ready() -> bool:
    """Check if the reranker is loaded and ready for inference."""
    return _reranker_ready


# =============================================================================
# RERANKING
# =============================================================================

def rerank(
    query: str,
    results: list,
    top_k: Optional[int] = None,
) -> list:
    """
    Rerank retrieval results using the cross-encoder model.

    Takes (query, chunk_snippet) pairs, scores them with the cross-encoder,
    and returns results sorted by cross-encoder score (descending).

    If the reranker is not loaded or fails, returns results unchanged
    (graceful fallback to FAISS-only ranking).

    Args:
        query:   The user's original query string
        results: List of RetrievalResult objects from FAISS search
        top_k:   Optional limit on returned results (None = return all)

    Returns:
        Reranked list of RetrievalResult objects with updated scores.
        Original FAISS scores are preserved in a new attribute if possible.
    """
    if not _reranker_ready or _reranker_model is None:
        logger.debug("Reranker not ready — returning results unchanged")
        return results[:top_k] if top_k else results

    if not results:
        return results

    try:
        # Build (query, document) pairs for cross-encoder scoring
        pairs = [(query, r.snippet) for r in results]

        # Batch inference — all pairs scored in one forward pass
        scores = _reranker_model.predict(pairs)

        # Normalize scores to [0, 1] range using sigmoid
        # bge-reranker outputs raw logits; sigmoid converts to probabilities
        import math
        normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]

        # Attach reranker scores and sort
        scored_results = list(zip(results, normalized_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build reranked output
        reranked = []
        for result, reranker_score in scored_results:
            # Create a copy with the reranker score replacing the FAISS score.
            # We use dataclasses.replace to avoid mutating the original.
            from dataclasses import replace as dc_replace
            updated = dc_replace(result, score=round(reranker_score, 4))
            reranked.append(updated)

        if top_k:
            reranked = reranked[:top_k]

        logger.debug(
            "Reranked %d results → top score: %.4f, bottom score: %.4f",
            len(reranked),
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )

        return reranked

    except Exception as e:
        logger.error("Reranking failed: %s — returning FAISS order", e)
        return results[:top_k] if top_k else results
