"""
FinSight AI — Query Expander (Phase 3: Recall Improvement)
============================================================
Financial domain-aware query expansion using synonym dictionaries.

Expands user terms into related financial vocabulary so embeddings
can match document language that uses different terminology.

Example:
    "revenue" → query becomes "revenue sales turnover total income"

This runs BEFORE embedding, modifying the query text itself.
No API calls, no model inference — pure dictionary lookup.

Author: FinSight AI Team
Phase: 3 (Recall Improvement Layer)
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


# =============================================================================
# FINANCIAL SYNONYM DICTIONARY
# =============================================================================

# Each key is a trigger term. When found in the query, its synonyms are
# appended to the query text. Case-insensitive matching.
# ~20 groups covering core financial terminology.

FINANCIAL_SYNONYMS = {
    "revenue": ["sales", "turnover", "total income", "gross revenue"],
    "profit": ["net income", "earnings", "net profit", "PAT", "profit after tax"],
    "debt": ["borrowings", "liabilities", "loans", "leverage", "indebtedness"],
    "margin": ["profitability", "margin ratio", "profit margin"],
    "growth": ["increase", "expansion", "rise", "CAGR", "year-over-year"],
    "dividend": ["payout", "distribution", "dividend per share", "DPS"],
    "assets": ["total assets", "asset base", "holdings"],
    "equity": ["shareholders equity", "net worth", "book value"],
    "cash flow": ["operating cash flow", "free cash flow", "cash generation", "OCF", "FCF"],
    "expense": ["cost", "expenditure", "spending", "OPEX", "operating expense"],
    "capex": ["capital expenditure", "capital spending", "investment outlay"],
    "EPS": ["earnings per share", "EPS diluted", "basic EPS"],
    "ROE": ["return on equity"],
    "ROA": ["return on assets"],
    "EBITDA": ["operating profit", "EBITDA margin", "operating income"],
    "NPA": ["non-performing assets", "bad loans", "gross NPA", "net NPA"],
    "AUM": ["assets under management"],
    "NII": ["net interest income", "interest income"],
    "CASA": ["current account savings account", "CASA ratio", "low-cost deposits"],
    "risk": ["risks", "risk factors", "challenges", "threats", "concerns"],
}

# Pre-compiled regex patterns for each trigger term (word-boundary matching)
_SYNONYM_PATTERNS = {
    term: re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    for term in FINANCIAL_SYNONYMS
}


# =============================================================================
# PUBLIC API
# =============================================================================

def expand_query(query: str) -> str:
    """
    Expand a query with financial synonyms.

    For each financial term found in the query, append its synonyms
    to broaden the embedding search surface.

    Args:
        query: Original user query string

    Returns:
        Expanded query string with synonyms appended.
        If no synonyms match, returns the original query unchanged.

    Example:
        >>> expand_query("What is RELIANCE revenue?")
        "What is RELIANCE revenue? [sales turnover total income gross revenue]"
    """
    expansions: List[str] = []
    matched_terms: List[str] = []

    for term, pattern in _SYNONYM_PATTERNS.items():
        if pattern.search(query):
            matched_terms.append(term)
            expansions.extend(FINANCIAL_SYNONYMS[term])

    if not expansions:
        return query

    # Deduplicate expansions while preserving order
    seen: Set[str] = set()
    unique_expansions = []
    for exp in expansions:
        exp_lower = exp.lower()
        # Don't add synonyms that are already in the query
        if exp_lower not in seen and exp_lower not in query.lower():
            seen.add(exp_lower)
            unique_expansions.append(exp)

    if not unique_expansions:
        return query

    # Append synonyms in brackets so the model can distinguish them
    expanded = f"{query} [{' '.join(unique_expansions)}]"

    logger.debug(
        "expand_query: matched terms=%s, added %d synonyms",
        matched_terms, len(unique_expansions),
    )

    return expanded


def get_synonym_count() -> int:
    """Return total number of synonym groups for observability."""
    return len(FINANCIAL_SYNONYMS)
