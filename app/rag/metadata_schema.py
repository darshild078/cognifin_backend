"""
FinSight AI - Metadata Schema
================================
Data structures for the corpus-based architecture.

This module defines:
- ChunkMetadata: metadata attached to each chunk (company, authority, etc.)
- SearchFilters: query-time filter criteria
- DocumentRecord: registry entry for each ingested document
- EntityRecord: canonical company identity with aliases

Phase 2.5: All structures exist, defaults produce identical Phase-2 behavior.

Author: FinSight AI Team
Phase: 2.5 (Corpus Architecture)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from datetime import datetime


# =============================================================================
# CHUNK METADATA
# =============================================================================

@dataclass
class ChunkMetadata:
    """
    Metadata attached to every chunk in the corpus.
    
    Identity Architecture:
        - vector_id: FAISS index position (the real unique key)
        - display_chunk_id: human-readable citation anchor ("chunk_42")
        - document_label: source context for prompt display
    
    These are intentionally separated:
        - vector_id prevents collisions across documents
        - display_chunk_id stays short for LLM citation parsing
        - document_label provides context without corrupting the anchor
    """
    # --- Identity ---
    vector_id: int                  # FAISS position (globally unique)
    display_chunk_id: str           # e.g. "chunk_42" (citation anchor)
    document_label: str             # e.g. "TCS_DRHP_2024_v1"
    
    # --- Source Context ---
    company: str                    # e.g. "TCS", "demo_company"
    document_type: str              # e.g. "DRHP", "Annual_Report"
    year: str                       # e.g. "2024", "2024_Q3"
    source_type: str                # "pdf", "html", "user_upload"
    
    # --- Authority & Classification ---
    authority: int                  # 100=official, 50=user, 30=derived
    section_hint: Optional[str]     # topic: "risk_factors", "financials", None
    content_form: str               # structure: "narrative", "tabular", "list", "heading", "unknown"
    contains_numeric: bool          # True if chunk has financial numbers
    
    # --- Lifecycle ---
    version: int                    # document revision (for replacements)
    is_active: bool                 # False = superseded by newer version
    temporal_scope: str             # "current" or "historical"
    source_class: str               # "official", "user", "derived"
    page_number: int = 1            # 1-based page number in source PDF



# =============================================================================
# SEARCH FILTERS
# =============================================================================

@dataclass
class SearchFilters:
    """
    Query-time filter criteria for corpus search.
    
    All fields default to None/permissive values, meaning:
    - No filtering applied → identical to Phase-2 behavior
    
    Fields accepting list[str] support multi-entity queries:
    - company=["TCS", "INFY"] → retrieve from both
    
    Temporal Precedence Rules:
        - temporal_scope="current" + no year → latest document only
        - year specified → explicit period query
        - temporal_scope=None → include historical (comparisons)
    """
    company: Optional[Union[str, List[str]]] = None
    document_type: Optional[Union[str, List[str]]] = None
    year: Optional[Union[str, List[str]]] = None
    source_type: Optional[str] = None
    active_only: bool = True
    temporal_scope: Optional[str] = "current"       # "current", "historical", None
    min_authority: Optional[int] = None
    source_class: Optional[str] = None              # "official", "user", "derived"
    require_numeric: bool = False

    def is_empty(self) -> bool:
        """
        Check if any non-default filters are active.
        
        Returns True when no filtering will occur — this means
        retrieval behavior is identical to Phase 2.
        """
        return (
            self.company is None
            and self.document_type is None
            and self.year is None
            and self.source_type is None
            and self.active_only is True
            and self.temporal_scope == "current"
            and self.min_authority is None
            and self.source_class is None
            and self.require_numeric is False
        )


# =============================================================================
# DOCUMENT RECORD
# =============================================================================

@dataclass
class DocumentRecord:
    """
    Registry entry for a single ingested document.
    
    The document_id is a composite key:
        "{company}_{document_type}_{year}_v{version}"
    
    vector_id_start and vector_id_end track where this document's
    chunks live in the FAISS index, enabling document-level operations
    (e.g., marking all chunks as superseded).
    
    Stage 2: source_path/source_url added for rebuild capability.
    At least one must be set for new documents.
    """
    document_id: str                    # e.g. "demo_company_DRHP_2024_v1"
    pdf_path: str                       # original file path
    chunk_count: int                    # number of chunks from this document
    vector_id_start: int                # first FAISS position
    vector_id_end: int                  # last FAISS position (exclusive)
    metadata: ChunkMetadata             # shared metadata template
    indexed_at: str = field(            # ISO timestamp
        default_factory=lambda: datetime.now().isoformat()
    )
    # Stage 2: Persistent source references for corpus rebuild
    source_path: Optional[str] = None   # CLI ingestion source (local path)
    source_url: Optional[str] = None    # future crawler source (URL)
    ingestion_timestamp: str = field(   # when this document was ingested
        default_factory=lambda: datetime.now().isoformat()
    )

    # Stage 3 addition — backward compatible: old registry files deserialize safely
    status: str = "active"   # "active" | "inactive"


# =============================================================================
# ENTITY RECORD
# =============================================================================

@dataclass
class EntityRecord:
    """
    Canonical company identity with aliases.
    
    Phase 2.5: Single hardcoded entity.
    Phase 3: Resolver maps aliases → entity_id.
    
    Example:
        entity_id = "RELIANCE_INDUSTRIES"
        entity_aliases = ["Reliance", "RIL", "Reliance Industries"]
        parent_entity = None
    """
    entity_id: str                              # e.g. "DEMO_COMPANY"
    entity_aliases: List[str] = field(          # alternate names
        default_factory=list
    )
    parent_entity: Optional[str] = None         # for subsidiary relationships


# =============================================================================
# RETRIEVAL RESULT
# =============================================================================

@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result.

    Attributes:
        chunk_id:       Which chunk was retrieved
        score:          Similarity score (higher = more similar)
        snippet:        The text content of the chunk
        page_number:    1-based page number in the source PDF (0 = unknown)
        document_label: Human-readable document identifier, e.g. "TCS_DRHP_2024_v1"
        pdf_filename:   Basename of the source PDF file, e.g. "TCS_DRHP_2024.pdf"
        pdf_url : url of the pdf in huggingface bucket
    """
    chunk_id: str
    score: float
    snippet: str
    page_number: int = 0
    document_label: str = ""
    pdf_filename: str = ""
    pdf_url: str = ""


# =============================================================================
# FACTORY FUNCTIONS (Convenience)
# =============================================================================

def create_default_metadata(
    vector_id: int,
    display_chunk_id: str,
    company: str = "demo_company",
    document_type: str = "DRHP",
    year: str = "2024",
) -> ChunkMetadata:
    """
    Create a ChunkMetadata with Phase-2.5 defaults.
    
    All classification fields default to "unknown"/None/False,
    producing identical behavior to Phase 2.
    
    Args:
        vector_id: FAISS index position
        display_chunk_id: e.g. "chunk_0"
        company: company name (default: "demo_company")
        document_type: filing type (default: "DRHP")
        year: fiscal year (default: "2024")
    
    Returns:
        ChunkMetadata with safe defaults
    """
    document_label = f"{company}_{document_type}_{year}_v1"
    
    return ChunkMetadata(
        vector_id=vector_id,
        display_chunk_id=display_chunk_id,
        document_label=document_label,
        company=company,
        document_type=document_type,
        year=year,
        source_type="pdf",
        authority=100,
        section_hint=None,
        content_form="unknown",
        contains_numeric=False,
        version=1,
        is_active=True,
        temporal_scope="current",
        source_class="official",
    )
