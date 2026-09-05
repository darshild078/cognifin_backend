"""
FinSight AI — Execution Planner (Phase 4)
==========================================
Converts IntelligentQuery → list of RetrievalStep objects.

Pure function — no I/O, no LLM calls, no embedding access.
Maps intent × complexity → concrete retrieval steps with scope filters.

Performance constraints:
- Max steps capped by MAX_RETRIEVAL_STEPS (default: 4)
- Multi-step queries use MULTI_STEP_RETRIEVAL_K (default: 10)
- Single queries use full RETRIEVAL_K (default: 20)

Author: FinSight AI Team
Phase: 4 (Intelligent Query Understanding)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

from .intelligent_parser import IntelligentQuery

logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVAL STEP SCHEMA
# =============================================================================

@dataclass
class RetrievalStep:
    """
    A single retrieval instruction in an execution plan.

    Each step becomes one call to the Phase 3 retrieval pipeline.
    """
    label: str                              # Human-readable tag ("WIPRO", "2023")
    query: str                              # Search query for this step
    companies: List[str] = field(default_factory=list)  # Scope filter
    years: List[str] = field(default_factory=list)      # Scope filter
    doc_types: List[str] = field(default_factory=list)   # Scope filter
    retrieval_k: int = 20                   # How many candidates to retrieve


# =============================================================================
# EXECUTION PLANNER
# =============================================================================

def plan_execution(iq: IntelligentQuery) -> List[RetrievalStep]:
    """
    Convert an IntelligentQuery into a list of RetrievalStep objects.

    Planning rules by intent:
        lookup    → 1 step (scoped by company+year)
        compare   → N steps (one per company)
        trend     → N steps (one per year)
        summarize → 1 step (broad scope)
        explain   → 1 step (focused search)
        list      → 1 step (section-focused)

    Performance:
        - Multi-step queries use reduced retrieval_k
        - Max steps capped to prevent latency explosion

    Args:
        iq: IntelligentQuery from llm_parse_query()

    Returns:
        List of RetrievalStep objects for execution
    """
    max_steps = int(os.getenv("MAX_RETRIEVAL_STEPS", "4"))
    full_k = int(os.getenv("RETRIEVAL_K", "20"))
    multi_k = int(os.getenv("MULTI_STEP_RETRIEVAL_K", "10"))

    if iq.intent == "compare" and len(iq.companies) >= 2:
        return _plan_compare(iq, max_steps, multi_k)

    if iq.intent == "trend" and len(iq.years) >= 2:
        return _plan_trend(iq, max_steps, multi_k)

    # All other intents: single step
    return _plan_single(iq, full_k)


# =============================================================================
# INTENT-SPECIFIC PLANNERS
# =============================================================================

def _plan_compare(
    iq: IntelligentQuery,
    max_steps: int,
    retrieval_k: int,
) -> List[RetrievalStep]:
    """
    One retrieval step per company.

    "Compare WIPRO and ADANIPORTS revenue" →
        Step 1: query="revenue", scope=[WIPRO]
        Step 2: query="revenue", scope=[ADANIPORTS]
    """
    steps = []
    # Use metrics in the query if available, otherwise use cleaned_query
    search_query = " ".join(iq.metrics) if iq.metrics else iq.cleaned_query

    for company in iq.companies[:max_steps]:
        steps.append(RetrievalStep(
            label=company,
            query=f"{company} {search_query}",
            companies=[company],
            years=list(iq.years),
            doc_types=list(iq.document_types),
            retrieval_k=retrieval_k,
        ))

    logger.debug(
        "plan_compare: %d steps for companies=%s, query='%s'",
        len(steps), iq.companies[:max_steps], search_query[:60],
    )
    return steps


def _plan_trend(
    iq: IntelligentQuery,
    max_steps: int,
    retrieval_k: int,
) -> List[RetrievalStep]:
    """
    One retrieval step per year.

    "RELIANCE revenue 2022-2024" →
        Step 1: query="RELIANCE revenue", scope=[2022]
        Step 2: query="RELIANCE revenue", scope=[2023]
        Step 3: query="RELIANCE revenue", scope=[2024]
    """
    steps = []
    search_query = " ".join(iq.metrics) if iq.metrics else iq.cleaned_query
    company_prefix = iq.companies[0] if iq.companies else ""

    for year in iq.years[:max_steps]:
        steps.append(RetrievalStep(
            label=year,
            query=f"{company_prefix} {search_query} {year}".strip(),
            companies=list(iq.companies),
            years=[year],
            doc_types=list(iq.document_types),
            retrieval_k=retrieval_k,
        ))

    logger.debug(
        "plan_trend: %d steps for years=%s, company=%s",
        len(steps), iq.years[:max_steps], company_prefix,
    )
    return steps


def _plan_single(iq: IntelligentQuery, retrieval_k: int) -> List[RetrievalStep]:
    """Single retrieval step for simple queries."""
    company_prefix = iq.companies[0] if iq.companies else ""
    search_query = iq.cleaned_query

    return [RetrievalStep(
        label=company_prefix or "general",
        query=f"{company_prefix} {search_query}".strip(),
        companies=list(iq.companies),
        years=list(iq.years),
        doc_types=list(iq.document_types),
        retrieval_k=retrieval_k,
    )]
