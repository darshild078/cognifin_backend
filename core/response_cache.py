"""
FinSight AI — Response Cache (Phase 6: Performance)
=====================================================
In-memory LRU response cache with TTL expiration.

Identical queries return cached responses in <5ms, bypassing
the entire retrieval + LLM pipeline.

Key: SHA-256 hash of (query + session_id)
Value: Full response dict (answer, citations, evidence, confidence)

Design:
- OrderedDict for LRU eviction (most recently used at end)
- TTL-based expiration (default: 1 hour)
- Thread-safe (threading.Lock)
- Max entries configurable (default: 500)
- Zero external dependencies

Author: FinSight AI Team
Phase: 6 (Performance & Production)
"""

import os
import time
import hashlib
import threading
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Thread-safe LRU response cache with TTL.

    Usage:
        cache = ResponseCache()
        hit = cache.get("What is RELIANCE revenue?")
        if hit:
            return hit  # <5ms
        # ... run full pipeline ...
        cache.set("What is RELIANCE revenue?", response_dict)
    """

    def __init__(self):
        self._enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self._max_size = int(os.getenv("CACHE_MAX_SIZE", "500"))
        self._ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        if self._enabled:
            logger.info(
                "Response cache initialized (max=%d, ttl=%ds)",
                self._max_size, self._ttl,
            )

    @staticmethod
    def _make_key(query: str, session_id: str = None) -> str:
        """Generate cache key from query + optional session_id."""
        raw = f"{query.strip().lower()}|{session_id or 'global'}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, session_id: str = None) -> Optional[Dict]:
        """
        Look up a cached response.

        Returns:
            Cached response dict if found and not expired, None otherwise.
        """
        if not self._enabled:
            return None

        key = self._make_key(query, session_id)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry["_cached_at"] > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            logger.debug("Cache HIT for query: '%s...'", query[:40])
            return entry["response"]

    def set(self, query: str, response: Dict, session_id: str = None) -> None:
        """
        Store a response in the cache.

        Evicts the least recently used entry if cache is full.
        """
        if not self._enabled:
            return

        key = self._make_key(query, session_id)

        with self._lock:
            # Remove old entry if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = {
                "response": response,
                "_cached_at": time.time(),
            }

    def invalidate_all(self) -> int:
        """Clear the entire cache. Returns number of evicted entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics for /diagnostics."""
        total = self._hits + self._misses
        return {
            "enabled": self._enabled,
            "entries": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
        }


# Global singleton
response_cache = ResponseCache()
