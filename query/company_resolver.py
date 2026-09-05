"""
FinSight AI - Company Resolver
================================
Maps user queries to canonical company identifiers.

Phase 2.5: Stub that always returns ["DEMO_COMPANY"].
Phase 3:   Will use keyword matching → NER → canonical lookup.

Design decisions:
- Returns a LIST (not string) to support multi-entity queries
  like "Compare TCS and Infosys" without interface changes.
- Uses EntityRecord for future alias resolution.

Author: FinSight AI Team
Phase: 2.5 (Corpus Architecture)
"""

from typing import List
from metadata_schema import EntityRecord


# =============================================================================
# DEFAULT ENTITY (Phase 2.5)
# =============================================================================

# Single demo entity — Phase 3 replaces this with a real registry
DEFAULT_ENTITY = EntityRecord(
    entity_id="DEMO_COMPANY",
    entity_aliases=["demo_company"],
    parent_entity=None,
)


# =============================================================================
# RESOLVER
# =============================================================================

def resolve_company(query: str, entity_registry: List[EntityRecord] = None) -> List[str]:
    """
    Resolve which companies a query is about.
    
    Phase 2.5 behavior:
        Always returns ["DEMO_COMPANY"] regardless of query.
        This ensures all chunks pass any company filter.
    
    Phase 3 behavior (not implemented):
        1. Keyword-match query against entity_aliases
        2. If no match, run NER on the query
        3. Map extracted names to canonical entity_ids
        4. Return list of matched entity_ids
    
    Args:
        query: The user's natural language question
        entity_registry: List of known entities (Phase 3)
    
    Returns:
        List of entity_id strings. Always at least one element.
        
    Examples:
        Phase 2.5:
            resolve_company("What are the risk factors?") → ["DEMO_COMPANY"]
        
        Phase 3 (future):
            resolve_company("Compare TCS and Infosys risks") → ["TCS", "INFY"]
            resolve_company("Tell me about Reliance") → ["RELIANCE_INDUSTRIES"]
    """
    # Phase 2.5: always return the demo entity
    # Phase 3 will add real resolution logic here
    return [DEFAULT_ENTITY.entity_id]
