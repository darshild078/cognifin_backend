"""
FinSight AI — Multi-Query Generator (Phase 3: Recall Improvement)
==================================================================
Generates semantically diverse query variants using GPT-4o-mini.

Why multi-query?
    A single query matches one phrasing. Financial documents use varied
    terminology. Generating 2 additional search variants dramatically
    improves recall without sacrificing precision (the reranker filters noise).

Example:
    Input:  "What is RELIANCE revenue?"
    Output: [
        "What is RELIANCE revenue?",                     (original)
        "RELIANCE total sales and turnover FY2025",      (variant 1)
        "annual income and gross revenue for RELIANCE",  (variant 2)
    ]

Feature Flag: MULTI_QUERY_ENABLED — set to 'false' for Phase 2 behavior.

Author: FinSight AI Team
Phase: 3 (Recall Improvement Layer)
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Store reference to OpenAI client (set in init_multi_query)
_openai_client = None


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

_MULTI_QUERY_PROMPT = """You are a financial document search assistant.
Given a user query about financial documents (annual reports, DRHP, balance sheets),
generate {variant_count} alternative search queries that would help find relevant
information. The alternatives should use different wording and financial terminology.

Rules:
- Keep the SAME intent, company name, and time period as the original query
- Use DIFFERENT financial terminology and phrasing
- Be specific and concrete — avoid vague generalizations
- Each query should be a standalone search query (not a question about the original)
- Output exactly {variant_count} queries, one per line, no numbering or bullets

User query: {query}"""


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_multi_query(openai_client) -> bool:
    """
    Initialize multi-query generator with the existing OpenAI client.

    Called once during application startup. Reuses the OpenAI client
    already configured for answer generation — no new API key needed.

    Args:
        openai_client: OpenAIClient instance from generation/openai_client.py

    Returns:
        True if initialization successful, False otherwise
    """
    global _openai_client

    if not os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true":
        logger.info("Multi-query disabled (MULTI_QUERY_ENABLED != true)")
        return False

    if openai_client is None or not openai_client.is_configured:
        logger.warning("Multi-query: OpenAI client not configured, disabling")
        return False

    _openai_client = openai_client
    logger.info("Multi-query generator initialized (model: %s)", openai_client.model)
    return True


# =============================================================================
# QUERY GENERATION
# =============================================================================

def generate_multi_queries(
    query: str,
    count: Optional[int] = None,
) -> List[str]:
    """
    Generate multiple semantically diverse query variants.

    Always includes the original query as the first element.
    If LLM generation fails, returns only the original query (graceful fallback).

    Args:
        query: Original user query string
        count: Total number of queries (including original). Default from env.

    Returns:
        List of query strings [original, variant_1, variant_2, ...]
        Minimum length: 1 (original only, on failure)
    """
    total_count = count or int(os.getenv("MULTI_QUERY_COUNT", "3"))
    variant_count = total_count - 1  # Subtract 1 for original

    # Always include the original
    queries = [query]

    # If multi-query is disabled or no variants requested, return original only
    if variant_count <= 0 or _openai_client is None:
        return queries

    try:
        # Build the prompt
        prompt = _MULTI_QUERY_PROMPT.format(
            variant_count=variant_count,
            query=query,
        )

        # Call GPT-4o-mini for variant generation
        # Use temperature=0.7 for some creativity in query reformulation
        response = _openai_client.client.chat.completions.create(
            model=_openai_client.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        # Parse response — each line is a variant
        raw_output = response.choices[0].message.content.strip()
        variants = [
            line.strip()
            for line in raw_output.split("\n")
            if line.strip() and len(line.strip()) > 10  # Skip empty/tiny lines
        ]

        # Take only the requested number of variants
        variants = variants[:variant_count]

        if variants:
            queries.extend(variants)
            logger.debug(
                "generate_multi_queries: %d variants for '%s': %s",
                len(variants), query[:60], variants,
            )
        else:
            logger.warning("generate_multi_queries: LLM returned no valid variants")

    except Exception as e:
        logger.error(
            "generate_multi_queries failed: %s — using original query only", e
        )

    return queries


def is_multi_query_enabled() -> bool:
    """Check if multi-query generation is available and enabled."""
    return (
        os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true"
        and _openai_client is not None
    )
