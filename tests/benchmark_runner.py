"""
FinSight AI — Benchmark Runner (Phase 7: Evaluation)
======================================================
Automated benchmark suite for validation.

Runs 10 pre-defined test queries against the /chat endpoint,
measures latency, and validates intent detection and confidence.

Usage:
    python -m tests.benchmark_runner

Requires: Backend running at http://127.0.0.1:8000

Author: FinSight AI Team
Phase: 7 (Evaluation & Monitoring)
"""

import time
import json
import requests
from typing import List, Dict, Optional


# =============================================================================
# BENCHMARK QUERIES
# =============================================================================

BENCHMARK_QUERIES = [
    {
        "query": "What is ADANIPORTS revenue?",
        "expected_intent": "lookup",
        "latency_target_ms": 3000,
    },
    {
        "query": "Compare WIPRO and ADANIPORTS",
        "expected_intent": "compare",
        "latency_target_ms": 4000,
    },
    {
        "query": "ADANIPORTS revenue trend",
        "expected_intent": "trend",
        "latency_target_ms": 4000,
    },
    {
        "query": "Summarize ADANIPORTS annual report",
        "expected_intent": "summarize",
        "latency_target_ms": 3000,
    },
    {
        "query": "Why did WIPRO debt increase?",
        "expected_intent": "explain",
        "latency_target_ms": 3000,
    },
    {
        "query": "List all risk factors for ADANIPORTS",
        "expected_intent": "list",
        "latency_target_ms": 3000,
    },
    {
        "query": "WIPRO vs ADANIPORTS profit margins",
        "expected_intent": "compare",
        "latency_target_ms": 4000,
    },
    {
        "query": "What is the meaning of life?",
        "expected_intent": None,  # No-context expected
        "latency_target_ms": 2000,
    },
    {
        "query": "What is ADANIPORTS revenue?",  # Repeat of #1 (cache test)
        "expected_intent": "lookup",
        "latency_target_ms": 200,  # Should be cached
        "cache_test": True,
    },
    {
        "query": "ADANIPORTS capex spending",
        "expected_intent": "lookup",
        "latency_target_ms": 3000,
    },
]

BASE_URL = "http://127.0.0.1:8000"


# =============================================================================
# RUNNER
# =============================================================================

def run_benchmark():
    """Execute all benchmark queries and print results."""
    print("\n" + "=" * 70)
    print("  FinSight AI — Benchmark Runner (Phase 7)")
    print("=" * 70)

    # Check server is running
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            print(f"❌ Server returned {r.status_code}. Start backend first.")
            return
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}. Start backend first.")
        return

    print(f"\n📊 Running {len(BENCHMARK_QUERIES)} queries...\n")

    results = []
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        result = _run_single(i, q)
        results.append(result)

    # Print summary
    _print_summary(results)

    # Print diagnostics
    _print_diagnostics()


def _run_single(index: int, query_spec: dict) -> dict:
    """Run a single benchmark query."""
    query = query_spec["query"]
    is_cache_test = query_spec.get("cache_test", False)

    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": query},
            timeout=30,
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            actual_latency = data.get("latency_ms", elapsed_ms)
            confidence = data.get("confidence", 0)
            confidence_label = data.get("confidence_label", "")
            citations = data.get("citations", [])
            answer_preview = data.get("answer", "")[:80]

            passed = actual_latency <= query_spec["latency_target_ms"]

            tag = "⚡CACHE" if (is_cache_test and actual_latency < 200) else ""
            status = "✅" if passed else "❌"

            print(
                f"  {status} #{index:2d} | {actual_latency:7.0f}ms"
                f" | conf={confidence:.2f}"
                f" | cites={len(citations)}"
                f" | {query[:45]:<45s}"
                f" {tag}"
            )

            return {
                "index": index,
                "query": query,
                "latency_ms": actual_latency,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "citations": len(citations),
                "passed": passed,
                "target_ms": query_spec["latency_target_ms"],
                "cache_test": is_cache_test,
            }
        else:
            elapsed_ms = (time.time() - start) * 1000
            print(f"  ❌ #{index:2d} | HTTP {response.status_code} | {query[:50]}")
            return {
                "index": index, "query": query, "latency_ms": elapsed_ms,
                "passed": False, "error": f"HTTP {response.status_code}",
            }

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        print(f"  ❌ #{index:2d} | ERROR: {e} | {query[:50]}")
        return {
            "index": index, "query": query, "latency_ms": elapsed_ms,
            "passed": False, "error": str(e),
        }


def _print_summary(results: List[dict]):
    """Print benchmark summary."""
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    latencies = [r["latency_ms"] for r in results if "error" not in r]
    confidences = [r.get("confidence", 0) for r in results if r.get("confidence")]

    print("\n" + "-" * 70)
    print(f"  Results: {passed}/{total} passed")

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        print(f"  Latency: avg={avg_lat:.0f}ms | min={min_lat:.0f}ms | max={max_lat:.0f}ms")

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"  Confidence: avg={avg_conf:.2f}")

    cache_tests = [r for r in results if r.get("cache_test")]
    if cache_tests:
        cache_lat = cache_tests[0].get("latency_ms", 0)
        print(f"  Cache response: {cache_lat:.0f}ms")

    print("-" * 70)


def _print_diagnostics():
    """Fetch and print /diagnostics endpoint."""
    try:
        r = requests.get(f"{BASE_URL}/diagnostics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print("\n📈 Diagnostics:")
            print(f"  Cache: {json.dumps(data.get('cache', {}), indent=2)}")
            print(f"  Queries: {json.dumps(data.get('queries', {}), indent=2)}")
    except Exception:
        pass


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_benchmark()
