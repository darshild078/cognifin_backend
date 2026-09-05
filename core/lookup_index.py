"""
lookup_index.py
---------------
Precomputed inverted index enabling O(1) scoped vector retrieval.

Layer  : Layer 3 — Query Intelligence (retrieval contract component)
Stage  : 3

Owns:
    doc_to_range      — immutable vector ranges per document
    company_to_docs   — inverted: company → active doc_ids
    doctype_to_docs   — inverted: doc_type → active doc_ids
    year_to_docs      — inverted: year (str) → active doc_ids

Does NOT own:
    FAISS index            (RetrieverPipeline)
    document_registry      (CorpusManager)
    chunk_metadata         (CorpusManager)
    embedding logic        (RetrieverPipeline)
    config fingerprint     (injected by CorpusManager, never computed here)

Persistence invariants:
    [P1] Stored separately from document_registry.json.
         Validated against registry AND faiss_ntotal at startup.
         If mismatched → rebuild from registry only. FAISS never touched.
    [P2] doc_to_range entries are immutable once written.
         add_document() raises ImmutableRangeError if range already exists.
         Replacement: mark old doc inactive + new doc_id.

Range semantics (match DocumentRecord exactly):
    vector_id_start — inclusive first FAISS position
    vector_id_end   — EXCLUSIVE last FAISS position
    valid IDs       — [start, end)
    total vectors   — end - start
"""

import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .cache_utils import atomic_write_json

if TYPE_CHECKING:
    from .metadata_schema import DocumentRecord

logger = logging.getLogger(__name__)

LOOKUP_INDEX_VERSION    = "1.0"
LOOKUP_INDEX_FILENAME   = "lookup_index.json"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ImmutableRangeError(Exception):
    """
    Raised when add_document() is called for a doc_id that already has a
    registered range. Ranges are permanent FAISS index pointers.
    To replace: mark old doc inactive, re-ingest under a new doc_id.
    """


class LookupIndexCorruptError(Exception):
    """Raised when LookupIndex cannot be loaded or rebuilt."""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


# =============================================================================
# RETRIEVAL SCOPE
# =============================================================================

@dataclass
class RetrievalScope:
    """
    A single scoped retrieval instruction.

    Passed to resolve_ranges() to produce allowed vector ranges.
    Forward-compatible: SearchPlanBuilder (Step 5) will map SubQuery →
    RetrievalScope before calling resolve_ranges().

    Field semantics:
        companies  — empty list = no company filter (search full corpus)
        doc_types  — empty list = no doc_type filter
        years      — str, must match ChunkMetadata.year ("2024", "2024_Q3")
                     empty list = no year filter
        top_k      — retrieval budget consumed by search_scoped()
        label      — human-readable tag for logging and result tagging
    """
    label:     str
    companies: List[str] = field(default_factory=list)
    doc_types: List[str] = field(default_factory=list)
    years:     List[str] = field(default_factory=list)
    top_k:     int = 5


# =============================================================================
# LOOKUP INDEX
# =============================================================================

@dataclass
class LookupIndex:
    """
    Precomputed lookup maps for scoped vector retrieval.

    Four maps:
        doc_to_range     — immutable: doc_id → (start_id, end_id)
                           end_id is EXCLUSIVE — matches DocumentRecord.vector_id_end
        company_to_docs  — inverted: company → {active doc_ids}
        doctype_to_docs  — inverted: doc_type → {active doc_ids}
        year_to_docs     — inverted: year (str) → {active doc_ids}

    Active-only rule:
        Inverted maps contain only active documents.
        doc_to_range contains ALL documents ever ingested (immutable record).
    """

    doc_to_range:    Dict[str, Tuple[int, int]] = field(default_factory=dict)
    company_to_docs: Dict[str, Set[str]]        = field(
                         default_factory=lambda: defaultdict(set))
    doctype_to_docs: Dict[str, Set[str]]        = field(
                         default_factory=lambda: defaultdict(set))
    year_to_docs:    Dict[str, Set[str]]        = field(
                         default_factory=lambda: defaultdict(set))

    # =========================================================================
    # MUTATION — ingestion path only
    # =========================================================================

    def add_document(
        self,
        doc_id:   str,
        company:  str,
        doc_type: str,
        year:     str,
        start_id: int,
        end_id:   int,   # EXCLUSIVE — matches DocumentRecord.vector_id_end
    ) -> None:
        """
        Register a new document's vector range.
        Called once per document by CorpusManager.add_document(),
        after FAISS append and DocumentRecord creation.

        Raises:
            ImmutableRangeError : doc_id already has a registered range.
            ValueError          : invalid range bounds.
        """
        # [P2] Immutability guard
        if doc_id in self.doc_to_range:
            existing = self.doc_to_range[doc_id]
            raise ImmutableRangeError(
                f"Range for '{doc_id}' is immutable and cannot be reassigned.\n"
                f"  Existing : {existing}\n"
                f"  Attempted: ({start_id}, {end_id})\n"
                f"  To replace this document: mark it inactive in document_registry "
                f"and re-ingest under a new doc_id."
            )

        if start_id < 0 or end_id < 0:
            raise ValueError(
                f"Vector IDs must be non-negative. "
                f"Got start_id={start_id}, end_id={end_id} for '{doc_id}'."
            )
        if start_id >= end_id:
            raise ValueError(
                f"start_id ({start_id}) must be < end_id ({end_id}) "
                f"for '{doc_id}'. Range must contain at least one vector."
            )

        self.doc_to_range[doc_id] = (start_id, end_id)
        self.company_to_docs[company].add(doc_id)
        self.doctype_to_docs[doc_type].add(doc_id)
        self.year_to_docs[year].add(doc_id)

        logger.debug(
            "LookupIndex.add_document: '%s' range=[%d, %d) "
            "company='%s' doc_type='%s' year='%s'",
            doc_id, start_id, end_id, company, doc_type, year,
        )

    def deactivate_document(
        self,
        doc_id:   str,
        company:  str,
        doc_type: str,
        year:     str,
    ) -> None:
        """
        Remove a document from all inverted indexes so it is excluded from
        future retrieval scopes.

        doc_to_range entry is preserved permanently:
            - Immutable historical record.
            - [P2] guard depends on this entry remaining present.

        Safe to call multiple times (discard is idempotent).
        """
        self.company_to_docs[company].discard(doc_id)
        self.doctype_to_docs[doc_type].discard(doc_id)
        self.year_to_docs[year].discard(doc_id)

        logger.info(
            "LookupIndex.deactivate_document: '%s' removed from inverted indexes "
            "(range preserved at %s).",
            doc_id, self.doc_to_range.get(doc_id, "NOT FOUND"),
        )

    # =========================================================================
    # ACTIVE STATE ENFORCEMENT — called after every deserialization
    # =========================================================================

    def _enforce_active_state(
        self,
        registry: Dict[str, "DocumentRecord"],
    ) -> None:
        """
        Correction pass: remove stale entries from all three inverted maps.

        Runs unconditionally after every load from disk, before validation.
        This is not a validation failure — it is a deterministic correction.
        The registry's status field is authoritative; the file is advisory.

        Removes any doc_id from an inverted map if:
            - The doc_id is not present in the registry (orphan), OR
            - The doc_id's status is not "active"
        """
        for map_attr in ("company_to_docs", "doctype_to_docs", "year_to_docs"):
            dim_map: Dict[str, Set[str]] = getattr(self, map_attr)
            for key in list(dim_map.keys()):
                stale: Set[str] = set()
                for doc_id in dim_map[key]:
                    record = registry.get(doc_id)
                    if record is None:
                        stale.add(doc_id)
                    elif getattr(record, "status", "active") != "active":
                        stale.add(doc_id)
                if stale:
                    dim_map[key] -= stale
                    logger.info(
                        "_enforce_active_state: removed %s from %s['%s']",
                        stale, map_attr, key,
                    )

    # =========================================================================
    # VALIDATION — startup only
    # =========================================================================

    def validate_against_registry(
        self,
        registry:     Dict[str, "DocumentRecord"],
        faiss_ntotal: int,
    ) -> Tuple[bool, List[str]]:
        """
        Cross-validate LookupIndex against document_registry AND FAISS.

        Four checks:
            [V1] No orphaned ranges: every doc_id in doc_to_range must
                 exist in document_registry.
            [V2] No missing ranges: every active DocumentRecord must have
                 a range in doc_to_range.
            [V3] Inverted index consistency: active docs appear in all
                 three maps under their correct keys.
            [V4] FAISS boundary: no range end exceeds faiss_ntotal.
                 Prevents search_scoped from querying non-existent positions.

        Returns:
            (is_valid: bool, errors: List[str])
        """
        errors: List[str] = []

        # [V1] Orphaned ranges
        for doc_id in self.doc_to_range:
            if doc_id not in registry:
                errors.append(
                    f"[V1:ORPHAN_RANGE] doc_to_range has '{doc_id}' "
                    f"absent from document_registry."
                )

        # [V4] FAISS boundary — check all ranges, not just active
        for doc_id, (start, end) in self.doc_to_range.items():
            if end > faiss_ntotal:
                errors.append(
                    f"[V4:FAISS_BOUNDARY] Range for '{doc_id}' ends at {end} "
                    f"but faiss_ntotal={faiss_ntotal}. "
                    f"Partial commit or index mismatch detected."
                )

        for doc_id, record in registry.items():
            is_active = getattr(record, "status", "active") == "active"

            if is_active:
                # [V2] Active doc must have a range
                if doc_id not in self.doc_to_range:
                    errors.append(
                        f"[V2:MISSING_RANGE] Active doc '{doc_id}' "
                        f"has no range in LookupIndex."
                    )
                    continue  # Cannot check V3 without a range

                # [V3] Inverted index consistency
                company  = record.metadata.company
                doc_type = record.metadata.document_type
                year     = record.metadata.year

                if doc_id not in self.company_to_docs.get(company, set()):
                    errors.append(
                        f"[V3:INDEX_GAP] Active doc '{doc_id}' missing from "
                        f"company_to_docs['{company}']."
                    )
                if doc_id not in self.doctype_to_docs.get(doc_type, set()):
                    errors.append(
                        f"[V3:INDEX_GAP] Active doc '{doc_id}' missing from "
                        f"doctype_to_docs['{doc_type}']."
                    )
                if doc_id not in self.year_to_docs.get(year, set()):
                    errors.append(
                        f"[V3:INDEX_GAP] Active doc '{doc_id}' missing from "
                        f"year_to_docs['{year}']."
                    )

        return len(errors) == 0, errors

    # =========================================================================
    # REBUILD — startup fallback only
    # =========================================================================

    @classmethod
    def rebuild_from_registry(
        cls,
        registry:     Dict[str, "DocumentRecord"],
        faiss_ntotal: int,
    ) -> "LookupIndex":
        """
        Reconstruct LookupIndex from DocumentRecord entries.
        FAISS is never written to — only faiss_ntotal is used as a boundary.

        Skips documents where:
            - vector_id_start or vector_id_end is invalid
            - vector_id_end > faiss_ntotal (partial commit — vectors absent)

        Restores doc_to_range for ALL valid documents (active + inactive).
        Restores inverted maps for ACTIVE documents only.
        """
        instance = cls()
        rebuilt  = 0
        skipped  = 0

        for doc_id, record in registry.items():
            start = record.vector_id_start
            end   = record.vector_id_end

            # Skip invalid ranges
            if start is None or end is None or start >= end:
                logger.warning(
                    "LookupIndex rebuild: skipping '%s' — "
                    "invalid range [%s, %s).", doc_id, start, end,
                )
                skipped += 1
                continue

            # Skip partially committed documents
            if end > faiss_ntotal:
                logger.warning(
                    "LookupIndex rebuild: skipping '%s' — "
                    "range end %d exceeds faiss_ntotal %d. "
                    "Partial commit detected.",
                    doc_id, end, faiss_ntotal,
                )
                skipped += 1
                continue

            # Restore immutable range for ALL valid docs (active + inactive)
            # Preserves [P2] guard for any future re-ingestion attempt.
            instance.doc_to_range[doc_id] = (start, end)

            # Restore inverted maps for ACTIVE docs only
            is_active = getattr(record, "status", "active") == "active"
            if is_active:
                instance.company_to_docs[record.metadata.company].add(doc_id)
                instance.doctype_to_docs[record.metadata.document_type].add(doc_id)
                instance.year_to_docs[record.metadata.year].add(doc_id)

            rebuilt += 1

        logger.info(
            "LookupIndex rebuilt from registry: "
            "%d processed, %d skipped (invalid or partial range).",
            rebuilt, skipped,
        )
        return instance

    # =========================================================================
    # STARTUP ENTRY POINT
    # =========================================================================

    @classmethod
    def load_or_rebuild(
        cls,
        save_dir:           str,
        registry:           Dict[str, "DocumentRecord"],
        faiss_ntotal:       int,
        config_fingerprint: str,
    ) -> "LookupIndex":
        """
        Canonical startup loader. Called by CorpusManager.init_lookup_index().

        Sequence:
            1. If file exists  → load, check fingerprint, enforce active state,
                                 validate against registry + faiss_ntotal.
            2. If valid        → return corrected in-memory instance.
            3. If invalid      → log errors, rebuild, save, return.
            4. If file absent  → build from registry, save, return.

        FAISS is never written to regardless of outcome.
        """
        path = os.path.join(save_dir, LOOKUP_INDEX_FILENAME)

        if os.path.exists(path):
            try:
                instance = cls._load_from_path(path, config_fingerprint)

                # Correction pass — runs before validation intentionally
                # so V3 reflects registry-authoritative active state
                instance._enforce_active_state(registry)

                is_valid, errors = instance.validate_against_registry(
                    registry, faiss_ntotal
                )

                if is_valid:
                    logger.info(
                        "LookupIndex: validation passed (%d documents).",
                        len(instance.doc_to_range),
                    )
                    return instance

                logger.warning(
                    "LookupIndex: validation failed with %d error(s). "
                    "Rebuilding from registry (FAISS untouched).",
                    len(errors),
                )
                for err in errors:
                    logger.warning("  %s", err)

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "LookupIndex: failed to load from '%s' (%s). "
                    "Rebuilding from registry.", path, exc,
                )
        else:
            logger.info(
                "LookupIndex: no file at '%s'. Building from registry.", path
            )

        # Rebuild path — registry is the sole source of truth
        instance = cls.rebuild_from_registry(registry, faiss_ntotal)
        instance.save(save_dir, config_fingerprint)
        return instance

    # =========================================================================
    # QUERY HELPERS — read-only, called by CorpusManager.search()
    # =========================================================================

    def resolve_ranges(
        self,
        scope: RetrievalScope,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """
        Hierarchical filter narrowing: company (anchor) → year → doc_type.

        Operates on a single RetrievalScope (one sub-query).
        Called once per SubQuery inside CorpusManager.search().

        Rules:
            Within a dimension  : UNION  (year "2022" OR "2023")
            Across dimensions   : AND    (company AND year AND doc_type)
            Graceful fallback   : if a refinement would produce an empty set,
                                  skip that dimension with a warning.

        Returns:
            (allowed_ranges, total_allowed_vectors)
            allowed_ranges        : sorted List[Tuple[int,int]] for bisect
            total_allowed_vectors : sum of (end - start) across all ranges,
                                    used as density cap in search_scoped()
        """

        # Stage 1 — company anchor
        if scope.companies:
            candidate_docs: Set[str] = set()
            for c in scope.companies:
                candidate_docs |= self.company_to_docs.get(c, set())
        else:
            # No company filter: start from full active corpus
            candidate_docs = {
                doc_id
                for docs in self.company_to_docs.values()
                for doc_id in docs
            }

        # Stage 2 — narrow by year (union within dimension)
        if scope.years:
            year_pool: Set[str] = set()
            for y in scope.years:
                year_pool |= self.year_to_docs.get(y, set())
            refined = candidate_docs & year_pool
            if refined:
                candidate_docs = refined
            else:
                logger.warning(
                    "resolve_ranges [%s]: year filter %s yielded empty "
                    "intersection — year filter skipped.",
                    scope.label, scope.years,
                )

        # Stage 3 — narrow by doc_type (union within dimension)
        if scope.doc_types:
            type_pool: Set[str] = set()
            for dt in scope.doc_types:
                type_pool |= self.doctype_to_docs.get(dt, set())
            refined = candidate_docs & type_pool
            if refined:
                candidate_docs = refined
            else:
                logger.warning(
                    "resolve_ranges [%s]: doc_type filter %s yielded empty "
                    "intersection — doc_type filter skipped.",
                    scope.label, scope.doc_types,
                )

        # Resolve doc_ids → vector ranges
        allowed_ranges = [
            self.doc_to_range[doc_id]
            for doc_id in candidate_docs
            if doc_id in self.doc_to_range
        ]

        # Sort for O(log R) bisect membership in search_scoped()
        allowed_ranges.sort()

        # end is exclusive: total = end - start
        total_allowed = sum(end - start for start, end in allowed_ranges)

        logger.debug(
            "resolve_ranges [%s]: %d docs → %d ranges → %d total vectors",
            scope.label, len(candidate_docs), len(allowed_ranges), total_allowed,
        )
        return allowed_ranges, total_allowed

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, save_dir: str, config_fingerprint: str) -> None:
        """
        Persist LookupIndex to lookup_index.json.

        Uses atomic_write_json — same crash-safe pattern as save_registry().
        Called as Phase 3 of the three-phase commit, AFTER pipeline.save_index().
        config_fingerprint is injected by CorpusManager (which owns the pipeline
        import) to avoid a circular dependency here.
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, LOOKUP_INDEX_FILENAME)
        data = self.to_dict()
        data["config_fingerprint"] = config_fingerprint
        atomic_write_json(path, data)
        logger.info(
            "LookupIndex saved: %s (%d documents)", path, len(self.doc_to_range)
        )

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (without config_fingerprint)."""
        return {
            "version": LOOKUP_INDEX_VERSION,
            "doc_to_range": {
                doc_id: list(rng)
                for doc_id, rng in self.doc_to_range.items()
            },
            "company_to_docs": {
                company: sorted(docs)
                for company, docs in self.company_to_docs.items()
                if docs
            },
            "doctype_to_docs": {
                dt: sorted(docs)
                for dt, docs in self.doctype_to_docs.items()
                if docs
            },
            "year_to_docs": {
                year: sorted(docs)
                for year, docs in self.year_to_docs.items()
                if docs
            },
        }

    @classmethod
    def _load_from_path(cls, path: str, config_fingerprint: str) -> "LookupIndex":
        """
        Load from JSON and verify config fingerprint.
        Raises ValueError on fingerprint mismatch — caller triggers rebuild.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version")
        if version != LOOKUP_INDEX_VERSION:
            logger.warning(
                "LookupIndex: version mismatch (file='%s', expected='%s').",
                version, LOOKUP_INDEX_VERSION,
            )

        # Fingerprint check — mismatch means FAISS was rebuilt with new config
        stored_fp = data.get("config_fingerprint")
        if stored_fp != config_fingerprint:
            raise ValueError(
                f"Config fingerprint mismatch: "
                f"stored='{stored_fp}', current='{config_fingerprint}'. "
                f"FAISS index was rebuilt — LookupIndex ranges are stale."
            )

        instance = cls()
        instance.doc_to_range = {
            doc_id: tuple(rng)
            for doc_id, rng in data.get("doc_to_range", {}).items()
        }
        instance.company_to_docs = defaultdict(set, {
            company: set(docs)
            for company, docs in data.get("company_to_docs", {}).items()
        })
        instance.doctype_to_docs = defaultdict(set, {
            dt: set(docs)
            for dt, docs in data.get("doctype_to_docs", {}).items()
        })
        instance.year_to_docs = defaultdict(set, {
            year: set(docs)
            for year, docs in data.get("year_to_docs", {}).items()
        })

        logger.info(
            "LookupIndex loaded from disk: %s (%d documents)",
            path, len(instance.doc_to_range),
        )
        return instance
