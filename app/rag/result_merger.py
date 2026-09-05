"""
FinSight AI — Result Merger (Phase 3: Recall Improvement)
==========================================================
Merges results from multiple queries and search systems using
Reciprocal Rank Fusion (RRF).

Why RRF:
    FAISS cosine scores (0-1) and BM25 scores (0-∞) are on incompatible
    scales. RRF uses RANK positions only — scale-invariant, proven in IR.

Formula:
    RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
                     i
    where k=60 (standard constant)

Chunks appearing in MULTIPLE result lists get higher RRF scores,
naturally promoting results that are relevant from multiple angles.

Author: FinSight AI Team
Phase: 3 (Recall Improvement Layer)
"""

import os
import logging
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import replace as dc_replace

logger = logging.getLogger(__name__)

# Known document type tokens to split on
_DOC_TYPE_TOKENS = [
    "_Annual_Report_", "_DRHP_", "_Quarterly_Report_",
    "_Balance_Sheet_", "_Profit_Loss_", "_Cash_Flow_",
]


def _extract_company(document_label: str) -> str:
    """
    Extract company name from document_label.

    Format: "COMPANY_DocType_Year_vN" → returns "COMPANY"
    Examples:
        "ADANIPORTS_Annual_Report_2023_v1" → "ADANIPORTS"
        "WIPRO_Annual_Report_2024_v1"      → "WIPRO"
        "TCS_DRHP_2025_v1"                 → "TCS"

    Falls back to full label if pattern doesn't match.
    """
    for token in _DOC_TYPE_TOKENS:
        if token in document_label:
            return document_label.split(token)[0]
    # Fallback: use first segment before underscore
    parts = document_label.split("_")
    return parts[0] if parts else document_label


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def merge_results(
    result_lists: List[List],
    labels: Optional[List[str]] = None,
    top_k: int = 20,
) -> list:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list may come from:
    - Different queries (multi-query variants)
    - Different search systems (FAISS dense, BM25 sparse)
    - Or both (multi-query × hybrid = N lists)

    Chunks appearing in multiple lists receive higher RRF scores,
    reflecting agreement across retrieval approaches.

    Args:
        result_lists: List of result lists, each containing RetrievalResult objects
        labels:       Optional labels for each list (for logging)
        top_k:        Number of results to return after merging

    Returns:
        Merged, deduplicated list of RetrievalResult with RRF scores,
        sorted by RRF score descending.
    """
    if not result_lists:
        return []

    # Flatten and handle single-list case
    non_empty = [rl for rl in result_lists if rl]
    if not non_empty:
        return []

    if len(non_empty) == 1:
        return non_empty[0][:top_k]

    rrf_k = int(os.getenv("RRF_K", "60"))
    max_doc_frac = float(os.getenv("MAX_DOC_CONCENTRATION", "0.4"))

    # ── Phase 1: Compute RRF scores per chunk_id ──
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, object] = {}  # chunk_id → best RetrievalResult object
    source_counts: Dict[str, int] = {}  # chunk_id → number of lists it appears in

    for list_idx, results in enumerate(non_empty):
        for rank, result in enumerate(results):
            cid = result.chunk_id
            rrf_score = 1.0 / (rrf_k + rank + 1)  # rank is 0-based

            # Accumulate RRF score across lists
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            # Track source count
            source_counts[cid] = source_counts.get(cid, 0) + 1

            # Keep the result object with the highest original score
            if cid not in chunk_map or result.score > chunk_map[cid].score:
                chunk_map[cid] = result

    # ── Phase 2: Sort by RRF score descending ──
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # ── Phase 3: Company-level diversity enforcement ──
    # Use company name (not full document label) as the diversity key.
    # This prevents one company from dominating all result slots in
    # comparison queries (e.g., 5 ADANIPORTS chunks, 0 WIPRO chunks).
    max_per_company = max(1, int(top_k * max_doc_frac))
    company_counts: Counter = Counter()
    merged = []

    for cid, rrf_score in sorted_chunks:
        if len(merged) >= top_k:
            break

        result = chunk_map[cid]
        doc_label = result.document_label or "unknown"

        # Extract company name from document_label
        # Format: "COMPANY_DocType_Year_vN" → take first segment before known types
        company_key = _extract_company(doc_label)

        if company_counts[company_key] >= max_per_company:
            continue

        # Replace score with RRF score for downstream use
        updated = dc_replace(result, score=round(rrf_score, 6))
        merged.append(updated)
        company_counts[company_key] += 1

    # ── Logging ──
    list_labels = labels or [f"list_{i}" for i in range(len(non_empty))]
    list_sizes = [len(rl) for rl in non_empty]
    multi_source = sum(1 for c in source_counts.values() if c > 1)

    logger.debug(
        "merge_results: %d lists %s → %d unique chunks → %d merged "
        "(multi-source=%d, rrf_k=%d)",
        len(non_empty), list_sizes, len(rrf_scores), len(merged),
        multi_source, rrf_k,
    )

    return merged
