"""
FinSight AI — Intelligent Query Parser (Phase 4)
==================================================
LLM-based query parsing using GPT-4o-mini with JSON mode.

Replaces rule-based parsing with deep understanding:
- 6 intent types (lookup, compare, trend, summarize, explain, list)
- Metric extraction (revenue, profit, margin, etc.)
- Complexity classification (simple, multi_step, complex)
- Retrieval strategy hints (single, per_entity, per_year)

Fallback: On ANY failure, wraps Phase 3 parse_query() result
in an IntelligentQuery object — zero downtime risk.

Author: FinSight AI Team
Phase: 4 (Intelligent Query Understanding)
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Store reference to OpenAI client
_openai_client = None


# =============================================================================
# INTELLIGENT QUERY SCHEMA
# =============================================================================

@dataclass
class IntelligentQuery:
    """
    Full LLM-parsed representation of user intent.

    Superset of ParsedQuery — includes metrics, constraints,
    retrieval strategy hints, and expected output format.
    """
    # ── Core Fields ──
    intent: str                    # lookup|compare|trend|summarize|explain|list
    complexity: str                # simple|multi_step|complex

    # ── Entities ──
    companies: List[str] = field(default_factory=list)
    years: List[str] = field(default_factory=list)
    document_types: List[str] = field(default_factory=list)

    # ── Semantic Fields ──
    metrics: List[str] = field(default_factory=list)

    # ── Query Versions ──
    original_query: str = ""
    cleaned_query: str = ""

    # ── Execution Hints ──
    retrieval_strategy: str = "single"   # single|per_entity|per_year
    expected_output: str = "narrative"    # table|narrative|bullet_points|number

    # ── Metadata ──
    parse_method: str = "llm"            # "llm" or "fallback"
    parse_time_ms: float = 0.0

    def to_parsed_query(self):
        """
        Convert to Phase 3 ParsedQuery for backward compatibility.

        Maps Phase 4 intents back to Phase 3 intent vocabulary.
        """
        from .query_understanding import ParsedQuery

        intent_map = {
            "compare": "comparison",
            "trend": "temporal",
            "lookup": "single_entity",
            "summarize": "generic",
            "explain": "single_entity",
            "list": "single_entity",
        }

        return ParsedQuery(
            cleaned_query=self.cleaned_query,
            companies=self.companies,
            years=self.years,
            document_types=self.document_types,
            intent=intent_map.get(self.intent, "generic"),
        )


# =============================================================================
# LLM PARSING PROMPT
# =============================================================================

_PARSE_PROMPT = """You are a financial query parser for FinSight AI. Parse the user query into a structured JSON object.

Available companies in the corpus: {companies}

Output EXACTLY this JSON structure (no markdown, no explanation):
{{
  "intent": "<lookup|compare|trend|summarize|explain|list>",
  "complexity": "<simple|multi_step|complex>",
  "companies": ["<company1>", "<company2>"],
  "years": ["<year1>", "<year2>"],
  "metrics": ["<metric1>", "<metric2>"],
  "retrieval_strategy": "<single|per_entity|per_year>",
  "expected_output": "<narrative|table|bullet_points|number>",
  "cleaned_query": "<core question without company/year tokens>"
}}

Intent rules:
- lookup: Simple fact question about one entity (e.g., "What is X revenue?")
- compare: Comparing 2+ entities (e.g., "Compare A and B", "A vs B")
- trend: Time-based analysis (e.g., "revenue growth 2021-2024")
- summarize: Broad overview request (e.g., "Summarize annual report")
- explain: Causal/reasoning question (e.g., "Why did debt increase?")
- list: Enumeration request (e.g., "List all risk factors")

Strategy rules:
- single: One retrieval call (lookup, summarize, explain, list)
- per_entity: One call per company (compare intent)
- per_year: One call per year (trend intent)

Complexity rules:
- simple: 1 entity, 1 metric, ≤1 year → single retrieval
- multi_step: 2+ entities OR 2+ years → parallel retrieval
- complex: 3+ entities AND 3+ years → capped multi-step

IMPORTANT:
- Only use company names from the available list above
- years must be 4-digit strings like "2023"
- metrics should be financial terms found in the query
- If unsure, default to intent="lookup", complexity="simple"

User query: "{query}"
"""


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_intelligent_parser(openai_client) -> bool:
    """Initialize the intelligent parser with the existing OpenAI client."""
    global _openai_client

    if not os.getenv("INTELLIGENT_PARSING_ENABLED", "true").lower() == "true":
        logger.info("Intelligent parsing disabled")
        return False

    if openai_client is None or not openai_client.is_configured:
        logger.warning("Intelligent parser: OpenAI not configured, will use fallback")
        return False

    _openai_client = openai_client
    logger.info("Intelligent query parser initialized (model: %s)", openai_client.model)
    return True


def is_intelligent_parsing_enabled() -> bool:
    """Check if LLM-based parsing is available."""
    return (
        os.getenv("INTELLIGENT_PARSING_ENABLED", "true").lower() == "true"
        and _openai_client is not None
    )


# =============================================================================
# LLM PARSING
# =============================================================================

def llm_parse_query(
    query: str,
    known_companies: List[str],
) -> IntelligentQuery:
    """
    Parse a user query using GPT-4o-mini with JSON mode.

    On ANY failure, falls back to wrapping Phase 3 parse_query() result.

    Args:
        query: Raw user query
        known_companies: Available companies from corpus registry

    Returns:
        IntelligentQuery — always returns a valid object (never raises)
    """
    start = time.time()

    # ── Try LLM parsing ──
    if is_intelligent_parsing_enabled():
        try:
            iq = _llm_parse(query, known_companies)
            iq.parse_time_ms = round((time.time() - start) * 1000, 1)
            iq.parse_method = "llm"
            logger.debug(
                "LLM parse: intent=%s, complexity=%s, companies=%s, metrics=%s (%.0fms)",
                iq.intent, iq.complexity, iq.companies, iq.metrics, iq.parse_time_ms,
            )
            return iq
        except Exception as e:
            logger.warning("LLM parsing failed (%s), falling back to Phase 3", e)

    # ── Fallback: wrap Phase 3 parse_query() ──
    return _fallback_parse(query, known_companies, start)


def _llm_parse(query: str, known_companies: List[str]) -> IntelligentQuery:
    """Internal: Call GPT-4o-mini for structured parsing."""
    companies_str = ", ".join(known_companies[:50])  # Cap to avoid token overflow
    prompt = _PARSE_PROMPT.format(companies=companies_str, query=query)

    response = _openai_client.client.chat.completions.create(
        model=_openai_client.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    # Validate and sanitize
    valid_intents = {"lookup", "compare", "trend", "summarize", "explain", "list"}
    intent = data.get("intent", "lookup")
    if intent not in valid_intents:
        intent = "lookup"

    valid_complexity = {"simple", "multi_step", "complex"}
    complexity = data.get("complexity", "simple")
    if complexity not in valid_complexity:
        complexity = "simple"

    valid_strategies = {"single", "per_entity", "per_year"}
    strategy = data.get("retrieval_strategy", "single")
    if strategy not in valid_strategies:
        strategy = "single"

    # Validate companies against known list
    raw_companies = data.get("companies", [])
    companies = [c for c in raw_companies if c in known_companies]

    # Validate years
    raw_years = data.get("years", [])
    years = [str(y) for y in raw_years if str(y).isdigit() and 2000 <= int(str(y)) <= 2035]

    return IntelligentQuery(
        intent=intent,
        complexity=complexity,
        companies=companies,
        years=years,
        document_types=[],
        metrics=data.get("metrics", []),
        original_query=query,
        cleaned_query=data.get("cleaned_query", query),
        retrieval_strategy=strategy,
        expected_output=data.get("expected_output", "narrative"),
    )


def _fallback_parse(
    query: str,
    known_companies: List[str],
    start_time: float,
) -> IntelligentQuery:
    """Wrap Phase 3 parse_query() result in an IntelligentQuery."""
    from .query_understanding import parse_query

    parsed = parse_query(query, known_companies)

    # Map Phase 3 intents → Phase 4 intents
    intent_map = {
        "single_entity": "lookup",
        "comparison": "compare",
        "temporal": "trend",
        "generic": "lookup",
    }

    # Infer complexity
    complexity = "simple"
    if len(parsed.companies) >= 2 or len(parsed.years) >= 2:
        complexity = "multi_step"

    # Infer strategy
    strategy = "single"
    if parsed.intent == "comparison":
        strategy = "per_entity"
    elif parsed.intent == "temporal":
        strategy = "per_year"

    elapsed = round((time.time() - start_time) * 1000, 1)

    return IntelligentQuery(
        intent=intent_map.get(parsed.intent, "lookup"),
        complexity=complexity,
        companies=list(parsed.companies),
        years=list(parsed.years),
        document_types=list(parsed.document_types),
        metrics=[],
        original_query=query,
        cleaned_query=parsed.cleaned_query,
        retrieval_strategy=strategy,
        expected_output="narrative",
        parse_method="fallback",
        parse_time_ms=elapsed,
    )
