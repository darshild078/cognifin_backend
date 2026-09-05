"""
FinSight AI - Corpus Manager
================================
Orchestration layer between the API and the retrieval engine.

Responsibilities:
- Document registry: track ingested documents with full metadata
- Identity mapping: vector_id → ChunkMetadata
- Search orchestration: delegate FAISS search, post-filter, return results
- Entity resolution: call resolver, use result as filter
- Corpus introspection: list companies, doc types, years
- Document lifecycle: version management, active/superseded marking

Design principle:
    CorpusManager is a THIN WRAPPER. It delegates all heavy work
    (embedding, FAISS search) to RetrieverPipeline. It only adds
    metadata tracking and post-filtering on top.

Phase 2.5: Works with a single document. All filters default to no-op.
Phase 3:   add_document() called multiple times → multi-company corpus.

Author: FinSight AI Team
Phase: 2.5 (Corpus Architecture)
"""

import os
import json
import logging
import pickle
from dataclasses import asdict, replace as dc_replace
from typing import List, Dict, Optional, Callable
import numpy as np
from .retriever_pipeline import RetrieverPipeline
from .metadata_schema import RetrievalResult
from .cache_utils import atomic_write_bytes, atomic_write_json
from .retrieval_logger import log_retrieval_event, RetrievalTimer

from .metadata_schema import (
    ChunkMetadata,
    DocumentRecord,
    create_default_metadata,
)

from .lookup_index import LookupIndex, ImmutableRangeError, RetrievalScope
from query.search_plan import SearchPlan, MergeStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# CORPUS MANAGER
# =============================================================================

class CorpusManager:
    """
    Central orchestrator for the corpus-based RAG architecture.
    
    This class sits between main.py and RetrieverPipeline:
    
        main.py → CorpusManager → RetrieverPipeline → FAISS
    
    It adds three capabilities that RetrieverPipeline doesn't have:
    1. Metadata tracking (which company, doc type, authority, etc.)
    2. Post-filter search results by metadata
    3. Document registry (what's been ingested, versions, lifecycle)
    
    Usage:
        corpus = CorpusManager(retriever_pipeline)
        corpus.add_document("data/sample.pdf", company="TCS", ...)
        results = corpus.search("What are the risk factors?")
    """
    
    def __init__(self, retriever: RetrieverPipeline):
        """
        Initialize the Corpus Manager.
        
        Args:
            retriever: An already-initialized RetrieverPipeline instance.
                      CorpusManager does NOT create its own — it wraps the existing one.
        """
        self.retriever = retriever
        
        # Document registry: document_id → DocumentRecord
        self.documents: Dict[str, DocumentRecord] = {}
        
        # Metadata mapping: vector_id (int) → ChunkMetadata
        # This is the core identity resolution table.
        # FAISS returns vector positions → we look up metadata here.
        self.chunk_metadata: Dict[int, ChunkMetadata] = {}
        
        # Track total vectors indexed (for vector_id_start calculation)
        self._total_vectors: int = 0
        
        # Stage 3: Precomputed inverted index for scoped retrieval (Step 1)
        self.lookup_index: LookupIndex = LookupIndex()
        
        print("📚 Corpus Manager initialized")
    
    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================
    
    def add_document(
        self,
        pdf_path: str,
        company: str = "demo_company",
        document_type: str = "DRHP",
        year: str = "2024",
        source_type: str = "pdf",
        authority: int = 100,
        source_class: str = "official",
    ) -> int:
        """
        Ingest a document into the corpus with metadata.
        
        Two execution paths:
          Cold start (no index): delegates to RetrieverPipeline.index_document()
          Append (index exists): uses prepare_document → embed_texts → append_vectors
        
        Stage 2 additions:
          - Duplicate protection: aborts if company+type+year already exists
          - Append-only: FAISS position is the sole ordering authority
          - Source tracking: stores pdf_path for rebuild capability
        
        Args:
            pdf_path: Path to the PDF file
            company: Company identifier
            document_type: Filing type (DRHP, Annual_Report, etc.)
            year: Fiscal year/period
            source_type: How document was obtained (pdf, html, user_upload)
            authority: Trust level (100=official, 50=user, 30=derived)
            source_class: official, user, or derived
        
        Returns:
            Number of chunks indexed from this document
            
        Raises:
            ValueError: If company+type+year already exists (duplicate protection)
        """
        # --- Stage 2: True duplicate protection ---
        # A company + document_type + year combination is unique.
        # Versioning belongs to replacement workflows (future stage).
        for doc_id, rec in self.documents.items():
            if (rec.metadata.company == company and
                    rec.metadata.document_type == document_type and
                    rec.metadata.year == year):
                raise ValueError(
                    f"Document for {company}/{document_type}/{year} already "
                    f"exists as '{doc_id}'. Stage 2 does not support "
                    f"replacement or versioning. "
                    f"Document replacement will be added in a future stage."
                )
        
        # Build the document label and ID
        version = self._get_next_version(company, document_type, year)
        document_label = f"{company}_{document_type}_{year}_v{version}"
        document_id = document_label
        
        logger.info("Adding document to corpus: %s", document_id)
        
        # --- Choose execution path ---
        if self.retriever.index is not None:
            # APPEND PATH: index already exists
            # 1. Prepare document (load → normalize → chunk) — returns (texts, page_numbers)
            texts, page_numbers = self.retriever.prepare_document(pdf_path)

            # 2. Embed texts
            embeddings = self.retriever.embed_texts(texts)

            # 3. Read start position from FAISS (sole ordering authority)
            vector_id_start = self.retriever.index.ntotal

            # 4. Append vectors (memory only) — pass page_numbers
            self.retriever.append_vectors(embeddings, texts, page_numbers)

            # 5. Read end position from FAISS AFTER append
            vector_id_end = self.retriever.index.ntotal
            num_chunks = vector_id_end - vector_id_start
        else:
            # COLD START PATH: no index exists yet
            vector_id_start = 0

            # Delegate to RetrieverPipeline (unchanged Phase 1 logic)
            num_chunks = self.retriever.index_document(pdf_path)

            # Read end position from FAISS AFTER index creation
            vector_id_end = self.retriever.index.ntotal
            # For cold-start, page_numbers come from the chunks already stored
            page_numbers = [
                self.retriever.chunks[i].page_number
                for i in range(vector_id_start, vector_id_end)
            ]

        # --- Create metadata for each chunk ---
        for i in range(vector_id_start, vector_id_end):
            # Chunk index within this document
            local_index = i - vector_id_start
            display_chunk_id = f"chunk_{local_index}"

            metadata = create_default_metadata(
                vector_id=i,
                display_chunk_id=display_chunk_id,
                company=company,
                document_type=document_type,
                year=year,
            )
            # Override non-default fields
            metadata.document_label = document_label
            metadata.source_type = source_type
            metadata.authority = authority
            metadata.source_class = source_class
            metadata.version = version
            # Store page number for this chunk
            metadata.page_number = page_numbers[local_index] if local_index < len(page_numbers) else 1

            self.chunk_metadata[i] = metadata

        
        # --- Register the document ---
        # Use the first chunk's metadata as the template
        template_meta = self.chunk_metadata[vector_id_start]
        
        record = DocumentRecord(
            document_id=document_id,
            pdf_path=pdf_path,
            chunk_count=num_chunks,
            vector_id_start=vector_id_start,
            vector_id_end=vector_id_end,
            metadata=template_meta,
            source_path=os.path.abspath(pdf_path),
        )
        self.documents[document_id] = record
        
        # Update total vector count
        self._total_vectors = vector_id_end

        # Stage 3: Register range in LookupIndex AFTER FAISS append
        # and AFTER DocumentRecord creation. vector_id_start/end are
        # already FAISS-authoritative at this point (read from index.ntotal).
        self.lookup_index.add_document(
            doc_id=document_id,
            company=company,
            doc_type=document_type,
            year=year,
            start_id=vector_id_start,
            end_id=vector_id_end,   # exclusive — matches DocumentRecord
        )
        
        logger.info(
            "Corpus updated: %s (%d chunks, vectors %d-%d)",
            document_id, num_chunks, vector_id_start, vector_id_end
        )
        
        return num_chunks
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        scope: RetrievalScope,
        query_vector: np.ndarray,
    ) -> List[RetrievalResult]:
        allowed_ranges, total_allowed = self.lookup_index.resolve_ranges(scope)

        if not allowed_ranges:
            logger.debug(
                "search [%s]: empty scope — returning [].",
                scope.label,
            )
            return []

        # Phase 2: Use RETRIEVAL_K for expanded candidate pool (reranker needs more candidates)
        retrieval_k = int(os.getenv("RETRIEVAL_K", str(scope.top_k * 3)))
        candidate_k = retrieval_k

        raw = self.retriever.search_scoped(
            query_vector,
            allowed_ranges,
            candidate_k,
            total_allowed,
        )

        results: List[RetrievalResult] = []

        for distance, vector_id in raw:
            # Phase 2: Remove top_k cap here — let all candidates through
            # to the reranker. Final trimming happens in refine_results().
            if len(results) >= retrieval_k:
                break

            if vector_id >= len(self.retriever.chunks):
                raise RuntimeError(
                    f"search [{scope.label}]: vector_id {vector_id} out of bounds "
                    f"(chunks len={len(self.retriever.chunks)}). "
                    f"Index/metadata mismatch — cache corruption."
                )

            meta = self.chunk_metadata.get(vector_id)
            if meta is None:
                logger.warning(
                    "search [%s]: vector_id %d missing from chunk_metadata — skipping.",
                    scope.label, vector_id,
                )
                continue

            chunk = self.retriever.chunks[vector_id]

            # Resolve PDF path for the static /pdfs/ route.
            # pdf_path may be a Colab path (e.g. /content/drive/MyDrive/data/ADANIPORTS/2023.pdf)
            # so we extract the last 2 segments (COMPANY/YEAR.pdf) which matches
            # the local data/ directory structure and the /pdfs/ static mount.
            # Resolve PDF URL
            pdf_url = ""

            for rec in self.documents.values():
                if rec.vector_id_start <= vector_id < rec.vector_id_end:

                    parts = rec.pdf_path.replace("\\", "/").rstrip("/").split("/")

                    asset_mode = os.getenv("ASSET_MODE", "local")

                    if asset_mode == "local":
                        if len(parts) >= 2:
                            pdf_url = f"/pdfs/{parts[-2]}/{parts[-1]}"
                        else:
                            pdf_url = f"/pdfs/{parts[-1]}"

                    else:
                        hf_base = os.getenv("HF_PDF_BASE_URL", "").rstrip("/")

                        if len(parts) >= 2:
                            pdf_url = f"{hf_base}/{parts[-2]}/{parts[-1]}"
                        else:
                            pdf_url = f"{hf_base}/{parts[-1]}"

                    break

            results.append(RetrievalResult(
                chunk_id=f"chunk_{vector_id}",
                score=distance,
                snippet=chunk.text,
                page_number=getattr(meta, "page_number", 0),
                document_label=meta.document_label,
                pdf_url=pdf_url
            ))

        # --- Phase 1: Score threshold filtering ---
        # Remove results below the similarity threshold to prevent
        # low-quality chunks from reaching the LLM.
        # Future (Phase 2): Implement top-k expansion before filtering —
        # retrieve top_k × N candidates from FAISS, apply threshold,
        # then trim to top_k. This prevents threshold filtering from
        # returning fewer results than expected.
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
        pre_filter_count = len(results)
        results = [r for r in results if r.score >= threshold]
        filtered_count = pre_filter_count - len(results)

        if filtered_count > 0:
            logger.debug(
                "search [%s]: filtered %d/%d results below threshold %.2f",
                scope.label, filtered_count, pre_filter_count, threshold,
            )

        # --- Phase 1: Retrieval observability ---
        log_retrieval_event(
            query="",  # query text is logged at the endpoint level
            results=results,
            filtered_count=filtered_count,
            threshold=threshold,
            scope_label=scope.label,
        )

        return results
    
    # =========================================================================
    # PLAN EXECUTION (Stage 3)
    # =========================================================================

    def execute_plan(
        self,
        plan: SearchPlan,
        embed_query: Callable[[str], np.ndarray],
    ) -> List[RetrievalResult]:
        per_subquery: List[List[RetrievalResult]] = []

        # Phase 2: Use RETRIEVAL_K as merge limit to preserve candidates
        # for the reranker. Final trimming to FINAL_K happens in refine_results().
        merge_limit = int(os.getenv("RETRIEVAL_K", str(plan.final_top_k)))

        for sub_query in plan.sub_queries:
            vector  = embed_query(sub_query.rewritten_query)
            results = self.search(sub_query.scope, vector)
            per_subquery.append(results)
            logger.debug(
                "execute_plan [%s]: %d result(s) retrieved.",
                sub_query.label, len(results),
            )

        if plan.merge_strategy == MergeStrategy.SINGLE:
            return per_subquery[0][:merge_limit]

        elif plan.merge_strategy == MergeStrategy.INTERLEAVED:
            merged: List[RetrievalResult] = []
            round_idx = 0
            while len(merged) < merge_limit:
                added_this_round = False
                for results in per_subquery:
                    if round_idx < len(results):
                        merged.append(results[round_idx])
                        added_this_round = True
                        if len(merged) >= merge_limit:
                            break
                if not added_this_round:
                    break
                round_idx += 1
            return merged

        elif plan.merge_strategy == MergeStrategy.SECTIONED:
            merged = []
            for sub_query, results in zip(plan.sub_queries, per_subquery):
                for r in results:
                    if len(merged) >= merge_limit:
                        break
                    labeled = dc_replace(
                        r,
                        chunk_id=f"[{sub_query.label}]:{r.chunk_id}",
                    )
                    merged.append(labeled)
                if len(merged) >= merge_limit:
                    break
            return merged

        logger.error(
            "execute_plan: unhandled MergeStrategy '%s' — returning [].",
            plan.merge_strategy,
        )
        return []

    # =========================================================================
    # CORPUS INTROSPECTION
    # =========================================================================
    
    def list_available_entities(self) -> dict:
        """
        List what's in the corpus.
        
        Returns a summary of registered documents:
        - companies: list of unique company names
        - document_types: list of unique filing types
        - years: list of unique fiscal periods
        - total_chunks: total chunks across all documents
        - documents: list of document_ids
        
        Returns:
            Dictionary with corpus summary
        """
        companies = set()
        doc_types = set()
        years = set()
        
        for doc in self.documents.values():
            companies.add(doc.metadata.company)
            doc_types.add(doc.metadata.document_type)
            years.add(doc.metadata.year)
        
        return {
            "companies": sorted(companies),
            "document_types": sorted(doc_types),
            "years": sorted(years),
            "total_chunks": self._total_vectors,
            "documents": list(self.documents.keys()),
        }
    
    @property
    def is_indexed(self) -> bool:
        """Whether any document has been successfully indexed."""
        return self.retriever.is_indexed
    
    @property
    def num_chunks(self) -> int:
        """Total number of chunks in the corpus."""
        return self._total_vectors
    
    # =========================================================================
    # PERSISTENCE (Stage 1)
    # =========================================================================
    
    def save_registry(self, save_dir: str) -> None:
        """
        Persist the document registry and chunk metadata to disk.
        
        Uses atomic writes (write → tmp → fsync → rename) so a crash
        mid-save never leaves corrupt registry files.
        
        Saved files:
          - document_registry.json : Human-readable records + ingestion order
          - chunk_metadata.pkl     : Pickled metadata mapping
        
        Args:
            save_dir: Directory to save into (created if missing)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Build ingestion order (preserves dict insertion order)
        ingestion_order = list(self.documents.keys())
        
        # Serialize document records to JSON
        registry_data = {
            "document_ingestion_order": ingestion_order,
            "documents": {
                doc_id: {
                    "document_id": rec.document_id,
                    "pdf_path": rec.pdf_path,
                    "chunk_count": rec.chunk_count,
                    "vector_id_start": rec.vector_id_start,
                    "vector_id_end": rec.vector_id_end,
                    "metadata": asdict(rec.metadata),
                    "indexed_at": rec.indexed_at,
                    # Stage 2: source references for rebuild
                    "source_path": rec.source_path,
                    "source_url": rec.source_url,
                    "ingestion_timestamp": rec.ingestion_timestamp,
                    "status":                rec.status,           # Stage 3
                }
                for doc_id, rec in self.documents.items()
            },
            "total_vectors": self._total_vectors,
        }
        
        atomic_write_json(
            os.path.join(save_dir, "document_registry.json"), registry_data
        )
        
        # Serialize chunk metadata (dataclass objects → pickle)
        metadata_bytes = pickle.dumps(self.chunk_metadata)
        atomic_write_bytes(
            os.path.join(save_dir, "chunk_metadata.pkl"), metadata_bytes
        )
        
        print(f"💾 Registry saved: {len(self.documents)} document(s), "
              f"{self._total_vectors} vectors")
    
    def load_registry(self, save_dir: str) -> bool:
        """
        Load the document registry and chunk metadata from disk.
        
        Validates:
          - File existence
          - Ingestion order matches document keys
        
        Args:
            save_dir: Directory containing saved registry files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        registry_path = os.path.join(save_dir, "document_registry.json")
        metadata_path = os.path.join(save_dir, "chunk_metadata.pkl")
        
        if not all(os.path.exists(p) for p in [registry_path, metadata_path]):
            return False
        
        try:
            # Load document registry from JSON
            with open(registry_path, "r") as f:
                registry_data = json.load(f)
            
            # Reconstruct DocumentRecord objects in ingestion order
            ingestion_order = registry_data.get(
                "document_ingestion_order", []
            )
            
            self.documents = {}
            for doc_id in ingestion_order:
                data = registry_data["documents"].get(doc_id)
                if data is None:
                    print(f"⚠️  Document '{doc_id}' in ingestion order "
                          f"but missing from registry")
                    return False

                # --- Backward-compatible metadata loading ---
                # Colab-generated registries may not have a 'metadata' sub-object.
                # Reconstruct ChunkMetadata from document_id if missing.
                if "metadata" in data:
                    meta = ChunkMetadata(**data["metadata"])
                else:
                    # Parse document_id: "{company}_{doctype}_{year}_v{version}"
                    # e.g. "ADANIENT_Annual_Report_2023_v1"
                    parts = data["document_id"].rsplit("_v", 1)
                    version = int(parts[1]) if len(parts) == 2 else 1
                    # Split the prefix by known doc types
                    prefix = parts[0]  # e.g. "ADANIENT_Annual_Report_2023"
                    # Find year (last segment)
                    segments = prefix.rsplit("_", 1)
                    year = segments[1] if len(segments) == 2 else "2024"
                    # Find company and doc_type
                    remaining = segments[0]  # e.g. "ADANIENT_Annual_Report"
                    # Try known doc types
                    company = remaining
                    doc_type = "Annual_Report"
                    for known_type in ["Annual_Report", "DRHP", "Quarterly_Report",
                                       "Balance_Sheet", "Profit_Loss", "Cash_Flow"]:
                        if f"_{known_type}" in remaining:
                            idx = remaining.index(f"_{known_type}")
                            company = remaining[:idx]
                            doc_type = known_type
                            break

                    meta = ChunkMetadata(
                        vector_id=data["vector_id_start"],
                        display_chunk_id=f"chunk_{data['vector_id_start']}",
                        document_label=data["document_id"],
                        company=company,
                        document_type=doc_type,
                        year=year,
                        source_type="pdf",
                        authority=100,
                        section_hint=None,
                        content_form="unknown",
                        contains_numeric=False,
                        version=version,
                        is_active=True,
                        temporal_scope="current",
                        source_class="official",
                        page_number=1,
                    )

                record = DocumentRecord(
                    document_id=data["document_id"],
                    pdf_path=data["pdf_path"],
                    chunk_count=data["chunk_count"],
                    vector_id_start=data["vector_id_start"],
                    vector_id_end=data["vector_id_end"],
                    metadata=meta,
                    indexed_at=data["indexed_at"],
                    # Stage 2 fields (backward-compat defaults)
                    source_path=data.get("source_path"),
                    source_url=data.get("source_url"),
                    ingestion_timestamp=data.get(
                        "ingestion_timestamp", data["indexed_at"]
                    ),
                    status=data.get("status", "active"),
                )
                self.documents[doc_id] = record
            
            # Validate all documents in registry were in order list
            registry_keys = set(registry_data["documents"].keys())
            order_keys = set(ingestion_order)
            if registry_keys != order_keys:
                print("⚠️  Ingestion order / registry key mismatch")
                return False
            
            self._total_vectors = registry_data["total_vectors"]
            
            # Load chunk metadata from pickle (with cross-environment remapping)
            # On Colab, ChunkMetadata may have been pickled as __main__.ChunkMetadata
            # or metadata_schema.ChunkMetadata — remap to core.metadata_schema
            _META_MODULE_ALIASES = {"__main__", "metadata_schema"}

            class _MetadataUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == "ChunkMetadata" and module in _META_MODULE_ALIASES:
                        return ChunkMetadata
                    return super().find_class(module, name)

            with open(metadata_path, "rb") as f:
                self.chunk_metadata = _MetadataUnpickler(f).load()
            
            print(f"📂 Registry loaded: {len(self.documents)} document(s), "
                  f"{self._total_vectors} vectors")
            return True
            
        except Exception as e:
            import traceback
            print(f"⚠️  Failed to load registry: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    
    def init_lookup_index(self, save_dir: str, faiss_ntotal: int) -> None:
        """
        Initialize LookupIndex after validate_cache_integrity() passes.

        Called during startup by both main.py and ingest.py.
        Must be called AFTER load_registry() and validate_cache_integrity()
        so that self.documents is populated and faiss_ntotal is confirmed.

        Passes faiss_ntotal for [V4] boundary validation and [P1] rebuild.
        """
        from .retriever_pipeline import RetrieverPipeline
        config_fingerprint = RetrieverPipeline.compute_config_fingerprint()
        self.lookup_index = LookupIndex.load_or_rebuild(
            save_dir=save_dir,
            registry=self.documents,
            faiss_ntotal=faiss_ntotal,
            config_fingerprint=config_fingerprint,
        )
        print(f"🔍 LookupIndex ready: {len(self.lookup_index.doc_to_range)} document(s)")

    def save_lookup_index(self, save_dir: str) -> None:
        """
        Phase 3 commit: persist LookupIndex after FAISS has been written.

        MUST be called after pipeline.save_index() — never before.
        Called by ingest.py and main.py as the final step of the three-phase
        commit sequence.
        """
        from .retriever_pipeline import RetrieverPipeline
        os.makedirs(save_dir, exist_ok=True)
        config_fingerprint = RetrieverPipeline.compute_config_fingerprint()
        self.lookup_index.save(save_dir, config_fingerprint)
        print(f"💾 LookupIndex saved: {len(self.lookup_index.doc_to_range)} document(s)")
    
    def validate_cache_integrity(self, faiss_count: int) -> bool:
        """
        Structural validation of the loaded cache.
        
        Checks:
          1. FAISS vector count == len(chunk_metadata)
          2. Sum of DocumentRecord.chunk_count == _total_vectors
          3. Every vector_id in [0, _total_vectors) has metadata
          4. Every document_id in documents exists with valid ranges
        
        Args:
            faiss_count: Number of vectors in the FAISS index (index.ntotal)
            
        Returns:
            True if all checks pass, False otherwise
        """
        # Check 1: FAISS count == metadata count
        metadata_count = len(self.chunk_metadata)
        if faiss_count != metadata_count:
            print(f"⚠️  FAISS vectors ({faiss_count}) != "
                  f"metadata entries ({metadata_count})")
            return False
        
        # Check 2: Registry chunk totals == _total_vectors
        registry_total = sum(
            rec.chunk_count for rec in self.documents.values()
        )
        if registry_total != self._total_vectors:
            print(f"⚠️  Registry chunk total ({registry_total}) != "
                  f"stored total ({self._total_vectors})")
            return False
        
        # Check 3: Every vector_id has metadata
        for vid in range(self._total_vectors):
            if vid not in self.chunk_metadata:
                print(f"⚠️  Vector ID {vid} missing from metadata")
                return False
        
        # Check 4: Document ranges are contiguous and valid
        for doc_id, rec in self.documents.items():
            if rec.vector_id_end - rec.vector_id_start != rec.chunk_count:
                print(f"⚠️  Document '{doc_id}' range mismatch: "
                      f"{rec.vector_id_start}-{rec.vector_id_end} "
                      f"!= {rec.chunk_count} chunks")
                return False
        
        print(f"✅ Cache integrity verified: {faiss_count} vectors, "
              f"{len(self.documents)} document(s)")
        return True

    # =========================================================================
    # INTERNAL: Versioning
    # =========================================================================
    
    def _get_next_version(self, company: str, document_type: str, year: str) -> int:
        """
        Determine the next version number for a document.
        
        If TCS_DRHP_2024_v1 exists, returns 2.
        If no previous version exists, returns 1.
        """
        version = 1
        while True:
            doc_id = f"{company}_{document_type}_{year}_v{version}"
            if doc_id not in self.documents:
                return version
            version += 1
