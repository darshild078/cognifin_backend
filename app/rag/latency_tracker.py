"""
FinSight AI — Latency Tracker (Phase 6: Performance)
======================================================
Per-request latency breakdown tracking using context managers.

Usage:
    tracker = LatencyTracker()
    with tracker.track("llm_parse"):
        result = llm_parse_query(...)
    with tracker.track("retrieval"):
        results = search(...)
    breakdown = tracker.get_breakdown()
    # {"llm_parse": 195.2, "retrieval": 312.5, "total": 507.7}

Thread-safe via per-instance storage (no globals).

Author: FinSight AI Team
Phase: 6 (Performance & Production)
"""

import time
from contextlib import contextmanager
from typing import Dict
from collections import defaultdict


class LatencyTracker:
    """
    Tracks per-stage latency for a single request.

    Create one instance per request. Use track() context manager
    to time each stage. Call get_breakdown() at the end.
    """

    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
        self._request_start = time.time()

    @contextmanager
    def track(self, stage: str):
        """
        Context manager that times a named stage.

        Args:
            stage: Name of the pipeline stage (e.g., "llm_parse", "retrieval")

        Example:
            with tracker.track("reranking"):
                results = rerank(query, candidates)
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self._timings[stage] = round(elapsed_ms, 1)

    def get_breakdown(self) -> Dict[str, float]:
        """
        Return timing breakdown for all tracked stages.

        Returns:
            Dict mapping stage name → milliseconds.
            Includes "total" key for end-to-end time.
        """
        result = dict(self._timings)
        result["total"] = round((time.time() - self._request_start) * 1000, 1)
        return result

    def get_total_ms(self) -> float:
        """Return total elapsed time in milliseconds."""
        return round((time.time() - self._request_start) * 1000, 1)


# =============================================================================
# GLOBAL STATS ACCUMULATOR
# =============================================================================

class LatencyStats:
    """
    Accumulates latency stats across all requests for /diagnostics.

    Thread-safe via simple append (no locks needed for list.append in CPython).
    """

    def __init__(self, max_history: int = 200):
        self._history = []
        self._max = max_history

    def record(self, breakdown: Dict[str, float]):
        """Record one request's latency breakdown."""
        self._history.append(breakdown)
        if len(self._history) > self._max:
            self._history = self._history[-self._max:]

    def get_averages(self) -> Dict[str, float]:
        """Return average latency per stage across recent requests."""
        if not self._history:
            return {}

        stages = set()
        for b in self._history:
            stages.update(b.keys())

        avgs = {}
        for stage in stages:
            values = [b.get(stage, 0) for b in self._history if stage in b]
            if values:
                avgs[f"avg_{stage}_ms"] = round(sum(values) / len(values), 1)

        avgs["request_count"] = len(self._history)
        return avgs


# Global instance
latency_stats = LatencyStats()
