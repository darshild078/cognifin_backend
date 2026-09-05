"""
FinSight AI — Confidence Scorer (Phase 5: Advanced RAG)
========================================================
Heuristic-based confidence scoring for RAG responses.

Assigns a 0.0-1.0 confidence score WITHOUT any extra LLM call.
Uses retrieval scores, context coverage, and citation density.

Runs in <5ms — pure computation on existing data.

Author: FinSight AI Team
Phase: 5 (Advanced RAG Layer)
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def compute_confidence(
    results: list,
    answer: str,
    query: str,
    citations: List[str],
) -> Tuple[float, str]:
    """
    Compute confidence score for a RAG response.

    Signals (weighted average):
        1. Top retrieval score (0.30) — how relevant is the best chunk?
        2. Score consistency (0.15) — are top chunks all relevant?
        3. Context coverage (0.25) — do retrieved chunks cover query terms?
        4. Citation density (0.15) — does the answer cite sources?
        5. Result availability (0.15) — were any chunks found at all?

    Args:
        results:   List of RetrievalResult objects used for context
        answer:    Generated LLM answer text
        query:     Original user query
        citations: List of cited chunk_ids found in the answer

    Returns:
        Tuple of (score: float 0-1, label: str)
        Labels: "high_confidence", "medium_confidence", "low_confidence"
    """
    if not results:
        return 0.0, "low_confidence"

    # ── Signal 1: Top retrieval score (weight: 0.30) ──
    scores = [r.score for r in results if r.score is not None]
    top_score = max(scores) if scores else 0.0
    # Normalize to 0-1 (reranker scores are already 0-1, FAISS cosine is 0-1)
    sig_top = min(1.0, max(0.0, top_score))

    # ── Signal 2: Score consistency (weight: 0.15) ──
    # If top-5 scores are tightly clustered → consistent context → higher confidence
    if len(scores) >= 2:
        score_range = max(scores) - min(scores)
        # Smaller range = more consistent = higher confidence
        sig_consistency = max(0.0, 1.0 - score_range)
    else:
        sig_consistency = 0.5

    # ── Signal 3: Context coverage (weight: 0.25) ──
    # What fraction of query terms appear in the retrieved context?
    query_terms = set(re.findall(r'[a-z0-9]+', query.lower()))
    # Remove stop words
    stop_words = {"what", "is", "the", "of", "in", "for", "and", "or", "a", "an",
                  "to", "how", "why", "when", "where", "does", "did", "do", "has",
                  "have", "been", "was", "were", "are", "this", "that", "with"}
    query_terms -= stop_words

    if query_terms:
        context_text = " ".join(r.snippet.lower() for r in results)
        matched = sum(1 for t in query_terms if t in context_text)
        sig_coverage = matched / len(query_terms)
    else:
        sig_coverage = 0.5

    # ── Signal 4: Citation density (weight: 0.15) ──
    # What fraction of answer sentences cite a source?
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 20]
    if sentences:
        cited_sentences = sum(1 for s in sentences if re.search(r'chunk_\d+', s))
        sig_citations = cited_sentences / len(sentences)
    else:
        sig_citations = 0.0

    # ── Signal 5: Result availability (weight: 0.15) ──
    sig_availability = 1.0 if len(results) >= 3 else len(results) / 3.0

    # ── Weighted average ──
    confidence = (
        0.30 * sig_top +
        0.15 * sig_consistency +
        0.25 * sig_coverage +
        0.15 * sig_citations +
        0.15 * sig_availability
    )

    confidence = round(min(1.0, max(0.0, confidence)), 3)

    # ── Label ──
    if confidence >= 0.75:
        label = "high_confidence"
    elif confidence >= 0.45:
        label = "medium_confidence"
    else:
        label = "low_confidence"

    logger.debug(
        "Confidence: %.3f (%s) — top=%.2f, consist=%.2f, cover=%.2f, cite=%.2f, avail=%.2f",
        confidence, label, sig_top, sig_consistency, sig_coverage, sig_citations, sig_availability,
    )

    return confidence, label
