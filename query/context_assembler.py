"""
FinSight AI — Context Assembler (Phase 4)
==========================================
Combines multi-step retrieval results into a structured context
string for the LLM.

Uses labeled sections (Option A) for comparison/trend queries
to help the LLM produce structured responses.

Author: FinSight AI Team
Phase: 4 (Intelligent Query Understanding)
"""

import logging
from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)

# Max total chunks across all steps (prevents context overflow)
MAX_TOTAL_CHUNKS = 10


# =============================================================================
# PUBLIC API
# =============================================================================

def assemble_context(
    step_results: Dict[str, list],
    intent: str = "lookup",
) -> Tuple[str, List[str]]:
    """
    Assemble multi-step retrieval results into structured context.

    For comparison/trend intents, produces labeled sections:
        --- WIPRO ---
        [chunk_5012]: WIPRO revenue...
        
        --- ADANIPORTS ---
        [chunk_142]: ADANIPORTS revenue...

    For other intents, produces flat relevance-sorted context.

    Args:
        step_results: Dict mapping step_label → List[RetrievalResult]
        intent: Query intent from IntelligentQuery

    Returns:
        Tuple of (context_string, chunk_ids)
    """
    if not step_results:
        return "", []

    if intent in ("compare", "trend") and len(step_results) > 1:
        return _assemble_labeled(step_results)
    else:
        return _assemble_flat(step_results)


# =============================================================================
# ASSEMBLY STRATEGIES
# =============================================================================

def _assemble_labeled(
    step_results: Dict[str, list],
) -> Tuple[str, List[str]]:
    """
    Labeled sections — one per step (company or year).

    Distributes chunks evenly across steps to prevent one entity
    from dominating the context.
    """
    chunks_per_step = max(2, MAX_TOTAL_CHUNKS // len(step_results))
    sections = []
    all_chunk_ids = []
    seen_chunks: Set[str] = set()

    for label, results in step_results.items():
        section_parts = [f"--- {label} ---"]
        added = 0

        for r in results:
            if added >= chunks_per_step:
                break
            if r.chunk_id in seen_chunks:
                continue

            section_parts.append(f"[{r.chunk_id}]: {r.snippet}")
            all_chunk_ids.append(r.chunk_id)
            seen_chunks.add(r.chunk_id)
            added += 1

        if added > 0:
            sections.append("\n".join(section_parts))

    context = "\n\n".join(sections)

    logger.debug(
        "assemble_labeled: %d sections, %d total chunks",
        len(sections), len(all_chunk_ids),
    )
    return context, all_chunk_ids


def _assemble_flat(
    step_results: Dict[str, list],
) -> Tuple[str, List[str]]:
    """
    Flat context — all results merged by score, deduplicated.
    Used for lookup, summarize, explain, list intents.
    """
    all_results = []
    for results in step_results.values():
        all_results.extend(results)

    # Sort by score descending
    all_results.sort(key=lambda r: r.score, reverse=True)

    # Deduplicate and cap
    parts = []
    chunk_ids = []
    seen: Set[str] = set()

    for r in all_results:
        if len(parts) >= MAX_TOTAL_CHUNKS:
            break
        if r.chunk_id in seen:
            continue

        parts.append(f"[{r.chunk_id}]: {r.snippet}")
        chunk_ids.append(r.chunk_id)
        seen.add(r.chunk_id)

    context = "\n\n".join(parts)

    logger.debug("assemble_flat: %d chunks from %d steps", len(chunk_ids), len(step_results))
    return context, chunk_ids
