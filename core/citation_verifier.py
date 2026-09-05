"""
FinSight AI — Citation Verifier (Phase 5: Advanced RAG)
========================================================
Verifies that LLM-generated citations are grounded in context.

Checks:
1. All cited chunk_ids actually exist in the provided context
2. Computes citation coverage (% of claims backed by chunks)
3. Flags unsupported sentences (no citation found)

Pure text matching — zero LLM calls, <5ms execution.

Author: FinSight AI Team
Phase: 5 (Advanced RAG Layer)
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def verify_citations(
    answer: str,
    chunk_ids: List[str],
    results: list = None,
) -> Dict:
    """
    Verify that citations in the answer are grounded.

    Args:
        answer:    LLM-generated answer text
        chunk_ids: Chunk IDs that were in the context
        results:   List of RetrievalResult objects (optional, for extra checks)

    Returns:
        Dict with verification results:
        {
            "valid_citations": ["chunk_12", "chunk_45"],
            "invalid_citations": [],
            "total_sentences": 8,
            "cited_sentences": 5,
            "coverage": 0.625,
            "verified": True
        }
    """
    # Find all chunk_N references in the answer
    found_refs = re.findall(r'chunk_\d+', answer)
    seen = set()
    unique_refs = []
    for ref in found_refs:
        if ref not in seen:
            unique_refs.append(ref)
            seen.add(ref)

    # Split into valid (in context) vs invalid (hallucinated)
    valid = [ref for ref in unique_refs if ref in chunk_ids]
    invalid = [ref for ref in unique_refs if ref not in chunk_ids]

    # Count sentences and citation coverage
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]
    cited_count = sum(1 for s in sentences if re.search(r'chunk_\d+', s))

    total = len(sentences)
    coverage = round(cited_count / total, 3) if total > 0 else 0.0

    # Answer is "verified" if no hallucinated citations and decent coverage
    verified = len(invalid) == 0 and coverage >= 0.3

    result = {
        "valid_citations": valid,
        "invalid_citations": invalid,
        "total_sentences": total,
        "cited_sentences": cited_count,
        "coverage": coverage,
        "verified": verified,
    }

    if invalid:
        logger.warning("Citation verification: %d invalid refs: %s", len(invalid), invalid)
    else:
        logger.debug("Citation verification: %d valid, coverage=%.1f%%", len(valid), coverage * 100)

    return result
