"""
FinSight AI - Document Ingestion CLI
=====================================
Offline tool for adding new documents to the corpus
without starting the server.

Usage:
    python ingest.py --file data/report.pdf --company TCS --type Annual_Report --year 2024

The ingestion follows a three-phase commit:
    1. Save registry first   (metadata phase)
    2. Save index last       (commit point)
    3. Save LookupIndex      (post-commit, after FAISS is safe on disk)

If a crash occurs between the two saves, startup validation
detects the mismatch and triggers a safe rebuild.

Author: FinSight AI Team
Stage: 2 (Manual Ingestion)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add backend root to path so package imports work when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment before anything else
from dotenv import load_dotenv
load_dotenv()

from core.retriever_pipeline import RetrieverPipeline
from core.corpus_manager import CorpusManager
from core.cache_utils import has_leftover_tmp, clean_cache


def parse_args():
    """Parse CLI arguments for document ingestion."""
    parser = argparse.ArgumentParser(
        description="FinSight AI — Ingest a document into the vector corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --file data/report.pdf
  python ingest.py --file docs/tcs_annual.pdf --company TCS --type Annual_Report --year 2024
  python ingest.py --file docs/reliance_drhp.pdf --company Reliance --type DRHP --year 2025
        """,
    )
    
    parser.add_argument(
        "--file", required=True,
        help="Path to the PDF file to ingest",
    )
    parser.add_argument(
        "--company",
        default=os.getenv("DEFAULT_COMPANY", "demo_company"),
        help="Company identifier (default: from .env or 'demo_company')",
    )
    parser.add_argument(
        "--type",
        default=os.getenv("DEFAULT_DOC_TYPE", "DRHP"),
        help="Document type: DRHP, Annual_Report, etc. (default: from .env)",
    )
    parser.add_argument(
        "--year",
        default=os.getenv("DEFAULT_YEAR", "2024"),
        help="Fiscal year/period (default: from .env or '2024')",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("INDEX_CACHE_DIR", "index_cache"),
        help="Cache directory (default: from .env or 'index_cache')",
    )
    
    return parser.parse_args()


def main():
    """
    Main ingestion entry point.
    
    Flow:
        1. Parse CLI args
        2. Validate PDF exists
        3. Init RetrieverPipeline + CorpusManager
        4. Load existing index + registry (if exists)
        5. Validate cache integrity BEFORE append
        6. Ingest document (duplicate check inside add_document)
        7. Three-phase commit: save registry → save index
    """
    args = parse_args()
    
    pdf_path = args.file
    company = args.company
    doc_type = args.type
    year = args.year
    cache_dir = args.cache_dir
    
    # ── Step 1: Validate PDF exists ──────────────────────────────────
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.lower().endswith(".pdf"):
        print(f"⚠️  Warning: '{pdf_path}' does not have a .pdf extension")
    
    print("=" * 60)
    print("FinSight AI — Document Ingestion")
    print("=" * 60)
    print(f"  File:    {pdf_path}")
    print(f"  Company: {company}")
    print(f"  Type:    {doc_type}")
    print(f"  Year:    {year}")
    print(f"  Cache:   {cache_dir}")
    print()
    
    # ── Step 2: Init pipeline + corpus manager ───────────────────────
    print("🤖 Initializing pipeline...")
    pipeline = RetrieverPipeline()
    corpus = CorpusManager(pipeline)
    
    # ── Step 3: Load existing index + registry ───────────────────────
    existing_index = False
    
    if os.path.exists(cache_dir):
        # Check for leftover tmp files (crashed previous save)
        if has_leftover_tmp(cache_dir):
            print("⚠️  Found leftover temporary files — cleaning cache")
            clean_cache(cache_dir)
        
        # Try loading existing index
        print("📂 Loading existing index...")
        index_loaded = pipeline.load_index(cache_dir)
        
        if index_loaded:
            registry_loaded = corpus.load_registry(cache_dir)
            
            if registry_loaded:
                # ── Step 4: Validate integrity BEFORE append ─────────
                integrity_ok = corpus.validate_cache_integrity(
                    pipeline.index.ntotal
                )
                
                if integrity_ok:
                    existing_index = True
                    print(f"✅ Loaded existing corpus: "
                          f"{pipeline.index.ntotal} vectors, "
                          f"{len(corpus.documents)} documents")
                    
                    # Stage 3: Initialize LookupIndex after integrity confirmed
                    corpus.init_lookup_index(cache_dir, pipeline.index.ntotal)
                else:
                    print("⚠️  Cache integrity check failed")
                    print("   Cannot safely append to a corrupted cache.")
                    print("   Please restart the server to trigger a clean rebuild,")
                    print("   then re-run this ingestion command.")
                    sys.exit(1)
            else:
                print("⚠️  Failed to load registry")
                print("   Cannot safely append without registry.")
                print("   Please restart the server to trigger a clean rebuild.")
                sys.exit(1)
        else:
            print("📭 No valid existing index found — will create fresh corpus")
    else:
        print("📭 No cache directory found — will create fresh corpus")
    
    # ── Step 5: Ingest document ──────────────────────────────────────
    doc_id_preview = f"{company}_{doc_type}_{year}"
    print(f"📄 Ingesting: {doc_id_preview}")
    print(f"   Loading and chunking PDF...")
    
    import time
    t_start = time.time()
    
    try:
        num_chunks = corpus.add_document(
            pdf_path=pdf_path,
            company=company,
            document_type=doc_type,
            year=year,
        )
    except ValueError as e:
        # Duplicate protection or empty PDF
        print(f"\n❌ Ingestion aborted: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during ingestion: {e}")
        sys.exit(1)
    
    t_elapsed = time.time() - t_start
    print(f"   ✅ Indexed {num_chunks} chunks in {t_elapsed:.1f}s")
    
    # ── Step 6: Three-phase commit ─────────────────────────────────────
    print("\n💾 Saving corpus (three-phase commit)...")
    
    # Phase 1: Save registry FIRST
    print("   Phase 1: Saving registry...")
    corpus.save_registry(cache_dir)
    
    # Phase 2: Save index (commit point)
    print("   Phase 2: Saving index (commit point)...")
    pipeline.save_index(cache_dir, pdf_path)

    # Phase 3: Save LookupIndex (post-commit — after FAISS is safe on disk)
    print("   Phase 3: Saving LookupIndex...")
    corpus.save_lookup_index(cache_dir)

    print("\n✅ Ingestion complete!")
    
    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"  Document:     {company}_{doc_type}_{year}")
    print(f"  Chunks added: {num_chunks}")
    print(f"  Total vectors: {pipeline.index.ntotal}")
    print(f"  Total documents: {len(corpus.documents)}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Timestamp:    {datetime.now().isoformat()}")
    
    # List all documents in corpus
    print("\n📚 Corpus contents:")
    for doc_id, rec in corpus.documents.items():
        print(f"  • {doc_id} ({rec.chunk_count} chunks, "
              f"vectors {rec.vector_id_start}-{rec.vector_id_end})")
    
    print("\n🚀 Server will load the updated corpus on next restart.")


if __name__ == "__main__":
    main()
