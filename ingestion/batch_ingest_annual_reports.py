"""
FinSight AI - Batch Ingestion: NIFTY 50 Annual Reports
========================================================
Deterministic batch ingestion script for annual reports.

Walks backend/data/<SYMBOL>/<YEAR>.pdf and ingests each PDF
via the existing CorpusManager pipeline.

Usage:
    cd backend
    python -m ingestion.batch_ingest_annual_reports

Author: FinSight AI Team
Stage: 7 (Batch Ingestion)
Phase 1: Fixed imports, added progress logging
"""

import os
import re
import sys
import time

from pathlib import Path

# Add backend root to path so package imports work when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.retriever_pipeline import RetrieverPipeline
from core.corpus_manager import CorpusManager
from core.lookup_index import ImmutableRangeError
from core.cache_utils import has_leftover_tmp, clean_cache


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CACHE_DIR = os.getenv("INDEX_CACHE_DIR", "index_cache")
YEAR_PDF_RE = re.compile(r"^\d{4}\.pdf$")


def main():
    # ── Step 1: Validate data directory ──────────────────────────────
    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # ── Step 2: Init pipeline + corpus manager ───────────────────────
    pipeline = RetrieverPipeline()
    corpus = CorpusManager(pipeline)

    # ── Step 3: Load existing index + registry (if any) ──────────────
    if os.path.exists(CACHE_DIR):
        if has_leftover_tmp(CACHE_DIR):
            clean_cache(CACHE_DIR)

        index_loaded = pipeline.load_index(CACHE_DIR)

        if index_loaded:
            registry_loaded = corpus.load_registry(CACHE_DIR)

            if registry_loaded:
                integrity_ok = corpus.validate_cache_integrity(
                    pipeline.index.ntotal
                )
                if integrity_ok:
                    corpus.init_lookup_index(CACHE_DIR, pipeline.index.ntotal)
                else:
                    print("Cache integrity check failed. Cannot proceed.")
                    sys.exit(1)
            else:
                print("Failed to load registry. Cannot proceed.")
                sys.exit(1)

    # ── Step 4: Discover company folders (sorted alphabetically) ─────
    companies = sorted(
        entry
        for entry in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, entry))
    )

    # Count total PDFs for progress tracking
    total_pdfs = 0
    for company in companies:
        company_dir = os.path.join(data_dir, company)
        total_pdfs += sum(1 for f in os.listdir(company_dir) if YEAR_PDF_RE.match(f))

    print(f"\n📊 Found {total_pdfs} PDFs across {len(companies)} companies\n")

    # ── Step 5: Ingest each PDF ──────────────────────────────────────
    ingested = 0
    skipped = 0
    batch_start = time.time()

    for company in companies:
        company_dir = os.path.join(data_dir, company)

        # Collect only valid year PDFs, sorted numerically
        pdfs = sorted(
            f
            for f in os.listdir(company_dir)
            if YEAR_PDF_RE.match(f)
        )

        for pdf_name in pdfs:
            year = pdf_name.replace(".pdf", "")
            pdf_path = os.path.join(company_dir, pdf_name)

            doc_start = time.time()
            print(f"[{ingested + skipped + 1}/{total_pdfs}] Ingesting {company} - {year}...", end=" ")

            try:
                corpus.add_document(
                    pdf_path=pdf_path,
                    company=company,
                    document_type="Annual_Report",
                    year=year,
                )
            except (ValueError, ImmutableRangeError) as e:
                print(f"SKIPPED (duplicate)")
                skipped += 1
                continue
            except Exception as e:
                print(f"\n❌ Error ingesting {company}/{year}: {e}")
                sys.exit(1)

            # Three-phase commit after each successful ingestion
            corpus.save_registry(CACHE_DIR)
            pipeline.save_index(CACHE_DIR, pdf_path)
            corpus.save_lookup_index(CACHE_DIR)

            doc_elapsed = time.time() - doc_start
            ingested += 1

            # Progress estimate
            elapsed_total = time.time() - batch_start
            avg_per_doc = elapsed_total / ingested
            remaining = (total_pdfs - ingested - skipped) * avg_per_doc
            remaining_min = remaining / 60

            print(f"✅ ({doc_elapsed:.1f}s) | ETA: {remaining_min:.0f}min")

    elapsed = time.time() - batch_start
    print(f"\n✅ Batch ingestion complete: {ingested} ingested, {skipped} skipped in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()

