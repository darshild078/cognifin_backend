"""
CogniFin AI — Multi-Query Generator (Phase 3: Recall Improvement)
==================================================================
Generates semantically diverse query variants using the unified LLMClient.
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger("cognifin.rag.multiquery")

# Store reference to LLM client (set in init_multi_query)
_llm_client = None


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

def init_multi_query(llm_client) -> bool:
    """
    Initialize multi-query generator with the existing LLM client.
    """
    global _llm_client

    if not os.getenv("MULTI_QUERY_ENABLED", "false").lower() == "true":
        logger.info("Multi-query disabled (MULTI_QUERY_ENABLED != true)")
        return False

    if llm_client is None or not llm_client.is_configured:
        logger.warning("Multi-query: LLM client not configured, disabling")
        return False

    _llm_client = llm_client
    logger.info("Multi-query generator initialized (model: %s)", llm_client.model)
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
    """
    total_count = count or int(os.getenv("MULTI_QUERY_COUNT", "3"))
    variant_count = total_count - 1

    queries = [query]

    if variant_count <= 0 or _llm_client is None:
        return queries

    try:
        prompt = _MULTI_QUERY_PROMPT.format(
            variant_count=variant_count,
            query=query,
        )

        raw_output = _llm_client.generate_text(prompt, temperature=0.7)
        variants = [
            line.strip()
            for line in raw_output.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

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
        os.getenv("MULTI_QUERY_ENABLED", "false").lower() == "true"
        and _llm_client is not None
    )

