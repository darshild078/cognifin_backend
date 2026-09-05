"""
FinSight AI — Retrieval Evaluation Framework (Stage 10)
========================================================
Deterministic, read-only evaluation harness that validates retrieval
correctness and isolation guarantees against the global corpus.

Does NOT:
    - Import FastAPI or OpenAI
    - Mutate any corpus or index_cache
    - Persist anything to disk
    - Modify any existing backend module

Usage:
    python evaluation.py               # Global corpus tests only
    python evaluation.py --with-session # Include session corpus test

Exit codes:
    0  — All tests passed
    1  — One or more tests failed

Author: FinSight AI Team
Stage: 10 (Retrieval Evaluation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import re
import time

# Ensure UTF-8 output on Windows (prevents UnicodeEncodeError with emojis)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Core pipeline imports (no FastAPI, no OpenAI)
from core.retriever_pipeline import RetrieverPipeline
from core.corpus_manager import CorpusManager
from core.corpus_router import CorpusRouter
from core.metadata_schema import ChunkMetadata, RetrievalResult
from query.query_understanding import parse_query
from query.search_plan_builder import build_plan
from core.cache_utils import has_leftover_tmp, clean_cache


# =============================================================================
# EVALUATION CASE DEFINITION
# =============================================================================

@dataclass
class EvalCase:
    """A single evaluation test case."""
    name: str
    query: str
    expected_companies: List[str]
    expected_years: List[str]
    intent: str


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        name="single_entity_revenue",
        query="TCS total revenue 2024",
        expected_companies=["TCS"],
        expected_years=["2024"],
        intent="single_entity",
    ),
    EvalCase(
        name="comparison_two_companies",
        query="Compare TCS and RELIANCE revenue 2024",
        expected_companies=["TCS", "RELIANCE"],
        expected_years=["2024"],
        intent="comparison",
    ),
    EvalCase(
        name="temporal_test",
        query="TCS revenue 2023",
        expected_companies=["TCS"],
        expected_years=["2023"],
        intent="single_entity",
    ),
]


# =============================================================================
# EVALUATION RESULT TRACKING
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseReport:
    """Aggregated report for one evaluation case."""
    case_name: str
    checks: List[CheckResult] = field(default_factory=list)
    precision_at_5: Optional[float] = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# =============================================================================
# METADATA EXTRACTION HELPERS
# =============================================================================

def extract_vector_id(chunk_id: str) -> Optional[int]:
    """
    Extract vector_id from a chunk_id string.

    Handles both plain and labeled formats:
        "chunk_42"          → 42
        "[TCS]:chunk_42"    → 42
    """
    match = re.search(r"chunk_(\d+)", chunk_id)
    if match:
        return int(match.group(1))
    return None


def build_metadata_map(
    results: List[RetrievalResult],
    chunk_metadata: Dict[int, ChunkMetadata],
) -> Dict[str, Optional[ChunkMetadata]]:
    """
    Map each RetrievalResult.chunk_id → its ChunkMetadata.

    Returns a dict keyed by chunk_id.  Value is None if the
    vector_id could not be resolved (should not happen in practice).
    """
    meta_map: Dict[str, Optional[ChunkMetadata]] = {}
    for r in results:
        vid = extract_vector_id(r.chunk_id)
        if vid is not None:
            meta_map[r.chunk_id] = chunk_metadata.get(vid)
        else:
            meta_map[r.chunk_id] = None
    return meta_map


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_company_isolation(
    results: List[RetrievalResult],
    meta_map: Dict[str, Optional[ChunkMetadata]],
    expected_companies: List[str],
) -> CheckResult:
    """
    All result.metadata.company must be in expected_companies.
    If an unexpected company is found → FAIL.
    """
    unexpected: List[str] = []
    expected_set = {c.upper() for c in expected_companies}

    for r in results:
        meta = meta_map.get(r.chunk_id)
        if meta is None:
            continue
        if meta.company.upper() not in expected_set:
            unexpected.append(meta.company)

    if unexpected:
        contaminants = sorted(set(unexpected))
        return CheckResult(
            name="Company isolation",
            passed=False,
            detail=f"Contamination: found {', '.join(contaminants)}",
        )
    return CheckResult(name="Company isolation", passed=True, detail="OK")


def validate_year_isolation(
    results: List[RetrievalResult],
    meta_map: Dict[str, Optional[ChunkMetadata]],
    expected_years: List[str],
) -> CheckResult:
    """
    If expected_years is non-empty, all result.metadata.year must match.
    """
    if not expected_years:
        return CheckResult(name="Year isolation", passed=True, detail="OK (no year filter)")

    unexpected: List[str] = []
    expected_set = set(expected_years)

    for r in results:
        meta = meta_map.get(r.chunk_id)
        if meta is None:
            continue
        if meta.year not in expected_set:
            unexpected.append(meta.year)

    if unexpected:
        bad_years = sorted(set(unexpected))
        return CheckResult(
            name="Year isolation",
            passed=False,
            detail=f"Unexpected years: {', '.join(bad_years)}",
        )
    return CheckResult(name="Year isolation", passed=True, detail="OK")


def validate_company_coverage(
    results: List[RetrievalResult],
    meta_map: Dict[str, Optional[ChunkMetadata]],
    expected_companies: List[str],
) -> CheckResult:
    """
    For comparison queries: each expected company must appear at least once.
    """
    found_companies: set = set()
    for r in results:
        meta = meta_map.get(r.chunk_id)
        if meta is not None:
            found_companies.add(meta.company.upper())

    expected_set = {c.upper() for c in expected_companies}
    missing = expected_set - found_companies

    if missing:
        return CheckResult(
            name="Company coverage",
            passed=False,
            detail=f"Missing: {', '.join(sorted(missing))}",
        )
    return CheckResult(name="Company coverage", passed=True, detail="OK")


def precision_at_k(
    results: List[RetrievalResult],
    meta_map: Dict[str, Optional[ChunkMetadata]],
    expected_companies: List[str],
    k: int = 5,
) -> float:
    """
    Return float ratio of top-k results belonging to expected companies.
    """
    top_k = results[:k]
    if not top_k:
        return 0.0

    expected_set = {c.upper() for c in expected_companies}
    relevant = 0
    for r in top_k:
        meta = meta_map.get(r.chunk_id)
        if meta is not None and meta.company.upper() in expected_set:
            relevant += 1

    return relevant / len(top_k)


# =============================================================================
# CORPUS LOADING (mirrors main.py lifespan)
# =============================================================================

def load_corpus() -> Tuple[RetrieverPipeline, CorpusManager, CorpusRouter]:
    """
    Load the global corpus from index_cache exactly as main.py lifespan does.

    Load order:
        1. load_index()
        2. load_registry()
        3. validate_cache_integrity()
        4. init_lookup_index()
        5. CorpusRouter(corpus_manager)

    Returns:
        (pipeline, corpus_manager, corpus_router)

    Raises:
        RuntimeError if any step fails.
    """
    cache_dir = os.getenv("INDEX_CACHE_DIR", "index_cache")

    # Initialize pipeline (loads embedding model)
    pipeline = RetrieverPipeline()

    # Wrap in CorpusManager
    corpus_manager = CorpusManager(pipeline)

    # Validate cache directory
    if not os.path.exists(cache_dir):
        raise RuntimeError(
            f"Cache directory '{cache_dir}' does not exist. "
            f"Run batch_ingest_annual_reports.py first."
        )

    if has_leftover_tmp(cache_dir):
        # Do NOT clean — this is read-only evaluation
        raise RuntimeError(
            f"Leftover tmp files in '{cache_dir}'. "
            f"Resolve cache corruption before running evaluation."
        )

    # Step 1: Load FAISS index
    if not pipeline.load_index(cache_dir):
        raise RuntimeError(
            f"Failed to load FAISS index from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 2: Load registry
    if not corpus_manager.load_registry(cache_dir):
        raise RuntimeError(
            f"Failed to load document registry from '{cache_dir}'. "
            f"Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 3: Validate integrity
    if not corpus_manager.validate_cache_integrity(pipeline.index.ntotal):
        raise RuntimeError(
            "Cache integrity check failed. "
            "Run batch_ingest_annual_reports.py to rebuild."
        )

    # Step 4: Initialize LookupIndex
    corpus_manager.init_lookup_index(cache_dir, pipeline.index.ntotal)

    # Step 5: Build router
    router = CorpusRouter(corpus_manager)

    return pipeline, corpus_manager, router


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def run_eval_case(
    case: EvalCase,
    pipeline: RetrieverPipeline,
    corpus_manager: CorpusManager,
    router: CorpusRouter,
    session_id: Optional[str] = None,
) -> CaseReport:
    """
    Execute a single evaluation case through the full pipeline:
        parse_query → build_plan → corpus_router.execute_plan

    Returns a CaseReport with all validation checks.
    """
    report = CaseReport(case_name=case.name)

    # --- Step 1: Parse query ---
    entities = corpus_manager.list_available_entities()
    companies = entities.get("companies", [])
    parsed = parse_query(case.query, companies)

    # --- Step 2: Build plan ---
    plan = build_plan(parsed, default_top_k=5)

    # --- Step 3: Execute plan via router ---
    results = router.execute_plan(
        plan,
        embed_query=lambda q: pipeline.embed_texts([q]),
        session_id=session_id,
    )

    # --- Step 4: Build metadata map ---
    meta_map = build_metadata_map(results, corpus_manager.chunk_metadata)

    # --- Step 5: Run validations ---

    # Company isolation
    report.checks.append(
        validate_company_isolation(results, meta_map, case.expected_companies)
    )

    # Year isolation
    report.checks.append(
        validate_year_isolation(results, meta_map, case.expected_years)
    )

    # Company coverage (meaningful for comparison intent)
    if case.intent == "comparison" or len(case.expected_companies) > 1:
        report.checks.append(
            validate_company_coverage(results, meta_map, case.expected_companies)
        )

    # Precision@5
    p5 = precision_at_k(results, meta_map, case.expected_companies, k=5)
    report.precision_at_5 = p5

    if p5 < 0.6:
        report.checks.append(
            CheckResult(
                name="Precision@5",
                passed=False,
                detail=f"{p5:.2f} (below 0.6 threshold)",
            )
        )

    return report


# =============================================================================
# SESSION TEST (--with-session)
# =============================================================================

def run_session_test(
    pipeline: RetrieverPipeline,
    corpus_manager: CorpusManager,
    router: CorpusRouter,
) -> Optional[CaseReport]:
    """
    Create a temporary session corpus from data/sample.pdf (company=TESTCO),
    run a cross-corpus query, validate, then remove the session.

    Returns CaseReport or None if test PDF not available.
    """
    test_pdf = os.path.join("data", "sample.pdf")
    if not os.path.exists(test_pdf):
        print("  ⚠️  Skipping session test: data/sample.pdf not found")
        return None

    session_id = "__eval_session__"

    try:
        # Create isolated pipeline + corpus (mirrors main.py /upload)
        session_pipeline = RetrieverPipeline()
        session_corpus = CorpusManager(session_pipeline)

        # Ingest into session corpus (in-memory only)
        num_chunks = session_corpus.add_document(
            pdf_path=test_pdf,
            company="TESTCO",
            document_type="Annual_Report",
            year="2024",
        )
        print(f"  📄 Session corpus created: {num_chunks} chunks (TESTCO)")

        # Register session
        router.register_session(session_id, session_corpus)

        # Build the cross-corpus eval case
        case = EvalCase(
            name="session_cross_corpus",
            query="Compare TESTCO and TCS revenue 2024",
            expected_companies=["TESTCO", "TCS"],
            expected_years=["2024"],
            intent="comparison",
        )

        # We need TESTCO in the known companies for parse_query
        # Merge global companies with session company
        entities = corpus_manager.list_available_entities()
        global_companies = entities.get("companies", [])
        all_companies = list(set(global_companies + ["TESTCO"]))

        # --- Execute manually so we can inject the augmented company list ---
        report = CaseReport(case_name=case.name)

        parsed = parse_query(case.query, all_companies)
        plan = build_plan(parsed, default_top_k=5)
        results = router.execute_plan(
            plan,
            embed_query=lambda q: pipeline.embed_texts([q]),
            session_id=session_id,
        )

        # Build combined metadata map from both corpora
        combined_meta: Dict[int, ChunkMetadata] = {}
        combined_meta.update(corpus_manager.chunk_metadata)
        combined_meta.update(session_corpus.chunk_metadata)

        meta_map = build_metadata_map(results, combined_meta)

        # Validate
        report.checks.append(
            validate_company_isolation(results, meta_map, case.expected_companies)
        )
        report.checks.append(
            validate_year_isolation(results, meta_map, case.expected_years)
        )
        report.checks.append(
            validate_company_coverage(results, meta_map, case.expected_companies)
        )

        p5 = precision_at_k(results, meta_map, case.expected_companies, k=5)
        report.precision_at_5 = p5
        if p5 < 0.6:
            report.checks.append(
                CheckResult(
                    name="Precision@5",
                    passed=False,
                    detail=f"{p5:.2f} (below 0.6 threshold)",
                )
            )

        return report

    finally:
        # Always clean up session — no persistence
        router.remove_session(session_id)
        print("  🗑️  Session removed (no data persisted)")


# =============================================================================
# REPORT PRINTER
# =============================================================================

def print_report(reports: List[CaseReport]) -> bool:
    """
    Print structured evaluation report.

    Returns True if all cases passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("  FinSight AI Retrieval Evaluation Report")
    print("=" * 60)

    all_passed = True

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        if not report.passed:
            all_passed = False

        print(f"\n  [{status}] {report.case_name}")

        for check in report.checks:
            indicator = "  ✅" if check.passed else "  ❌"
            print(f"    {indicator} {check.name}: {check.detail}")

        if report.precision_at_5 is not None:
            print(f"    📊 Precision@5: {report.precision_at_5:.2f}")

    print("\n" + "=" * 60)
    overall = "PASS ✅" if all_passed else "FAIL ❌"
    print(f"  OVERALL RESULT: {overall}")
    print("=" * 60 + "\n")

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinSight AI — Retrieval Evaluation Framework (Stage 10)"
    )
    parser.add_argument(
        "--with-session",
        action="store_true",
        help="Include session corpus cross-routing test",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔬 FinSight AI — Retrieval Evaluation Framework")
    print("=" * 60)

    # --- Load corpus ---
    print("\n📂 Loading global corpus...")
    start = time.time()
    try:
        pipeline, corpus_manager, router = load_corpus()
    except RuntimeError as e:
        print(f"\n❌ Startup failed: {e}")
        return 1

    elapsed = time.time() - start
    entities = corpus_manager.list_available_entities()
    print(f"   Loaded in {elapsed:.2f}s — "
          f"{corpus_manager.num_chunks} vectors, "
          f"{len(entities.get('companies', []))} companies, "
          f"{len(entities.get('years', []))} years")

    # --- Run evaluation cases ---
    print(f"\n🧪 Running {len(EVAL_CASES)} evaluation cases...")
    reports: List[CaseReport] = []

    for case in EVAL_CASES:
        print(f"\n  → {case.name}: \"{case.query}\"")
        report = run_eval_case(case, pipeline, corpus_manager, router)
        reports.append(report)

    # --- Optional session test ---
    if args.with_session:
        print("\n🔗 Running session cross-corpus test...")
        session_report = run_session_test(pipeline, corpus_manager, router)
        if session_report is not None:
            reports.append(session_report)

    # --- Print report ---
    all_passed = print_report(reports)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
