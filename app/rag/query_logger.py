"""
FinSight AI — Query Logger (Phase 7: Evaluation & Monitoring)
===============================================================
Persistent JSONL query log for evaluation and debugging.

Each query writes one JSON line to `logs/query_log.jsonl`.
Captures: query, intent, scores, confidence, latency, errors.

Format: JSON Lines (one JSON object per line, appendable).
Location: backend/logs/query_log.jsonl

Author: FinSight AI Team
Phase: 7 (Evaluation & Monitoring)
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Log file location
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "query_log.jsonl"

# Global counters
_query_counts = {
    "total": 0,
    "no_context": 0,
    "low_confidence": 0,
    "fallback_triggers": 0,
    "cached": 0,
    "errors": 0,
}


def _ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_query(
    query: str,
    intent: str = "unknown",
    parse_method: str = "unknown",
    num_chunks: int = 0,
    top_score: float = 0.0,
    confidence: float = 0.0,
    confidence_label: str = "",
    latency_ms: float = 0.0,
    latency_breakdown: Optional[Dict] = None,
    cached: bool = False,
    error: Optional[str] = None,
) -> None:
    """
    Log a query event to the JSONL file.

    Args:
        query:             User's original question
        intent:            Detected intent (lookup, compare, etc.)
        parse_method:      "llm" or "fallback"
        num_chunks:        Number of chunks in final context
        top_score:         Highest retrieval score
        confidence:        Confidence score (0-1)
        confidence_label:  "high_confidence", etc.
        latency_ms:        Total request time in milliseconds
        latency_breakdown: Per-stage timing dict
        cached:            Whether response was from cache
        error:             Error message if request failed
    """
    if not os.getenv("QUERY_LOG_ENABLED", "true").lower() == "true":
        return

    # Update counters
    _query_counts["total"] += 1
    if cached:
        _query_counts["cached"] += 1
    if num_chunks == 0 and not cached:
        _query_counts["no_context"] += 1
    if confidence_label == "low_confidence":
        _query_counts["low_confidence"] += 1
    if parse_method == "fallback":
        _query_counts["fallback_triggers"] += 1
    if error:
        _query_counts["errors"] += 1

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:200],
        "intent": intent,
        "parse_method": parse_method,
        "num_chunks": num_chunks,
        "top_score": round(top_score, 4),
        "confidence": round(confidence, 3),
        "confidence_label": confidence_label,
        "latency_ms": round(latency_ms, 1),
        "latency_breakdown": latency_breakdown or {},
        "cached": cached,
        "error": error,
    }

    try:
        _ensure_log_dir()
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write query log: %s", e)


def get_query_stats() -> Dict:
    """Return query statistics for /diagnostics."""
    return dict(_query_counts)


def get_recent_logs(n: int = 20) -> List[Dict]:
    """Return the most recent N query log entries."""
    if not _LOG_FILE.exists():
        return []

    try:
        lines = _LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent if line.strip()]
    except Exception as e:
        logger.error("Failed to read query log: %s", e)
        return []
