"""
FinSight AI — BM25 Retriever (Phase 3: Hybrid Search)
=======================================================
Sparse keyword-based retrieval using BM25 scoring.

Complements FAISS dense search by catching:
- Exact financial terms ("Section 134", "Clause 49")
- Numeric values ("₹45,230 crore")
- Abbreviations that embeddings may not handle well ("NPA", "CASA")

Index is built at startup from pipeline.chunks (no separate file).
Search supports scope filtering via allowed_ranges (same as FAISS).

Built on rank_bm25 (pure Python, no external server).

Author: FinSight AI Team
Phase: 3 (Recall Improvement Layer)
"""

import os
import re
import logging
import time
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Global singleton — initialized once at startup
_bm25_index = None
_tokenized_corpus = None
_bm25_ready = False


# =============================================================================
# TOKENIZATION
# =============================================================================

def _tokenize(text: str) -> List[str]:
    """
    Simple tokenization for BM25 indexing and querying.

    Approach: lowercase + split on non-alphanumeric characters.
    Preserves numbers (important for financial data).
    No stemming or lemmatization — keeps it simple and fast.

    Args:
        text: Raw text string

    Returns:
        List of lowercase tokens
    """
    # Lowercase and split on non-alphanumeric (keep digits for "45230", "2024")
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return tokens


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_bm25(chunks: list, cache_dir: Optional[str] = None) -> bool:
    """
    Build or load BM25 index from disk cache for fast startup.

    If bm25.pkl exists in cache_dir and matches chunk count, loads in ~0.3s.
    Otherwise tokenizes and builds, saving to cache_dir for subsequent startups.

    Args:
        chunks: List of Chunk objects from RetrieverPipeline
        cache_dir: Optional directory containing saved index files

    Returns:
        True if index is ready, False otherwise
    """
    global _bm25_index, _tokenized_corpus, _bm25_ready

    if not os.getenv("BM25_ENABLED", "true").lower() == "true":
        logger.info("BM25 disabled (BM25_ENABLED != true)")
        return False

    if not chunks:
        logger.warning("BM25: No chunks provided, skipping index build")
        return False

    cache_dir = cache_dir or os.getenv("INDEX_CACHE_DIR", "index_cache")
    bm25_cache_path = os.path.join(cache_dir, "bm25.pkl")

    import pickle
    from rank_bm25 import BM25Okapi

    # Fast path: Load precomputed BM25 index from disk
    if os.path.exists(bm25_cache_path):
        try:
            start_time = time.time()
            with open(bm25_cache_path, "rb") as f:
                _bm25_index = pickle.load(f)
            if hasattr(_bm25_index, "corpus_size") and _bm25_index.corpus_size == len(chunks):
                _bm25_ready = True
                elapsed = time.time() - start_time
                logger.info(f"📂 BM25 index loaded from disk: {len(chunks)} documents in {elapsed:.2f}s")
                return True
        except Exception as e:
            logger.warning(f"Failed to load cached BM25 index: {e} — rebuilding...")

    try:
        start_time = time.time()
        logger.info("Building BM25 index from %d chunks...", len(chunks))

        # Tokenize all chunks
        _tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]

        # Build BM25 index
        _bm25_index = BM25Okapi(_tokenized_corpus)
        _bm25_ready = True

        elapsed = time.time() - start_time
        logger.info(
            "BM25 index built: %d documents, %.1fs",
            len(chunks), elapsed,
        )

        # Save to disk cache for future instant boots
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(bm25_cache_path, "wb") as f:
                pickle.dump(_bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Saved BM25 index to {bm25_cache_path}")
        except Exception as save_err:
            logger.warning(f"Could not persist BM25 cache: {save_err}")

        return True

    except ImportError:
        logger.error(
            "rank_bm25 not installed. Run: pip install rank-bm25"
        )
        return False
    except Exception as e:
        logger.error("BM25 index build failed: %s", e)
        _bm25_ready = False
        return False


def is_bm25_ready() -> bool:
    """Check if BM25 index is ready for search."""
    return _bm25_ready


# =============================================================================
# SEARCH
# =============================================================================

def bm25_search(
    query: str,
    k: int = 20,
    allowed_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[float, int]]:
    """
    Search the BM25 index for relevant chunks.

    Returns results in the same format as FAISS search_scoped():
    List of (score, vector_id) tuples, sorted by score descending.

    Args:
        query:           Search query string
        k:               Number of results to return
        allowed_ranges:  Optional list of (start, end) vector_id ranges
                         for scope filtering (same as FAISS)

    Returns:
        List of (bm25_score, vector_id) tuples, highest scores first.
        Returns empty list if BM25 is not ready.
    """
    if not _bm25_ready or _bm25_index is None:
        return []

    try:
        # Tokenize query
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Get scores for all documents
        scores = _bm25_index.get_scores(query_tokens)

        # Build (score, vector_id) pairs
        scored = [(float(scores[i]), i) for i in range(len(scores))]

        # Apply scope filtering if provided
        if allowed_ranges:
            def _in_scope(vid: int) -> bool:
                for start, end in allowed_ranges:
                    if start <= vid < end:
                        return True
                return False

            scored = [(s, vid) for s, vid in scored if _in_scope(vid)]

        # Filter out zero scores (no keyword overlap)
        scored = [(s, vid) for s, vid in scored if s > 0.0]

        # Sort by score descending and take top-k
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[:k]

    except Exception as e:
        logger.error("BM25 search failed: %s", e)
        return []
