"""
test_corpus_router.py
---------------------
Unit tests for CorpusRouter (Stage 8B).

Self-contained: CorpusManager.search() is mocked.
No FAISS, no disk I/O, no embedding model required.

Run with:
    python -m pytest test_corpus_router.py -v
"""

import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, replace as dc_replace
from unittest.mock import MagicMock
from typing import List

from core.corpus_manager import CorpusManager
from core.corpus_router import CorpusRouter, _merge_by_score, _apply_merge_strategy
from core.metadata_schema import RetrievalResult
from core.lookup_index import RetrievalScope
from query.search_plan import SearchPlan, SubQuery, MergeStrategy


# =============================================================================
# HELPERS
# =============================================================================

def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, snippet=f"text_{chunk_id}")


def make_results(tag: str, scores: List[float]) -> List[RetrievalResult]:
    return [make_result(f"{tag}_{i}", s) for i, s in enumerate(scores)]


def make_cm() -> CorpusManager:
    """Minimal CorpusManager bypassing __init__; search() must be mocked."""
    cm = CorpusManager.__new__(CorpusManager)
    cm.retriever = MagicMock()
    cm.retriever.is_indexed = True
    cm.documents = {}
    cm.chunk_metadata = {}
    cm._total_vectors = 0
    cm.lookup_index = MagicMock()
    cm.search = MagicMock()
    cm.execute_plan = MagicMock()
    cm.list_available_entities = MagicMock(return_value={"companies": [], "document_types": [], "years": [], "documents": {}, "total_chunks": 0})
    return cm


def make_scope(label: str, top_k: int = 5) -> RetrievalScope:
    return RetrievalScope(label=label, top_k=top_k)


def make_sub(label: str, query: str = "q", top_k: int = 5) -> SubQuery:
    return SubQuery(label=label, rewritten_query=query, scope=make_scope(label, top_k))


def fixed_embed(_: str):
    import numpy as np
    return np.zeros(384)


# =============================================================================
# MERGE BY SCORE
# =============================================================================

class TestMergeByScore(unittest.TestCase):

    def test_interleave_by_score(self):
        a = make_results("a", [0.9, 0.7, 0.5])
        b = make_results("b", [0.8, 0.6, 0.4])
        merged = _merge_by_score(a, b)
        scores = [r.score for r in merged]
        self.assertEqual(scores, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    def test_empty_a(self):
        b = make_results("b", [0.8, 0.6])
        merged = _merge_by_score([], b)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].chunk_id, "b_0")

    def test_empty_b(self):
        a = make_results("a", [0.9, 0.7])
        merged = _merge_by_score(a, [])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].chunk_id, "a_0")

    def test_both_empty(self):
        merged = _merge_by_score([], [])
        self.assertEqual(merged, [])


# =============================================================================
# GLOBAL-ONLY ROUTING
# =============================================================================

class TestGlobalOnly(unittest.TestCase):

    def test_delegates_to_global(self):
        """session_id=None should delegate entirely to global_corpus.execute_plan()"""
        global_cm = make_cm()
        expected = make_results("g", [0.9, 0.8])
        global_cm.execute_plan.return_value = expected

        router = CorpusRouter(global_cm)
        plan = SearchPlan(
            sub_queries=[make_sub("test")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        results = router.execute_plan(plan, fixed_embed, session_id=None)
        self.assertEqual(results, expected)
        global_cm.execute_plan.assert_called_once()


# =============================================================================
# SESSION LIFECYCLE
# =============================================================================

class TestSessionLifecycle(unittest.TestCase):

    def test_register_and_has(self):
        router = CorpusRouter(make_cm())
        session_cm = make_cm()
        router.register_session("s1", session_cm)
        self.assertTrue(router.has_session("s1"))
        self.assertFalse(router.has_session("s2"))

    def test_remove(self):
        router = CorpusRouter(make_cm())
        router.register_session("s1", make_cm())
        router.remove_session("s1")
        self.assertFalse(router.has_session("s1"))

    def test_remove_nonexistent_is_noop(self):
        router = CorpusRouter(make_cm())
        router.remove_session("nonexistent")  # should not raise

    def test_unknown_session_raises_keyerror(self):
        router = CorpusRouter(make_cm())
        plan = SearchPlan(
            sub_queries=[make_sub("test")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        with self.assertRaises(KeyError):
            router.execute_plan(plan, fixed_embed, session_id="nonexistent")


# =============================================================================
# DUAL-CORPUS ROUTING
# =============================================================================

class TestDualCorpus(unittest.TestCase):

    def _setup_router(self, global_results, session_results):
        """Helper: create router with mocked global + session corpora."""
        global_cm = make_cm()
        session_cm = make_cm()

        global_cm.search.return_value = global_results
        session_cm.search.return_value = session_results

        router = CorpusRouter(global_cm)
        router.register_session("s1", session_cm)
        return router

    def test_merges_by_score(self):
        """Dual-corpus results should be merged by score."""
        g = make_results("g", [0.9, 0.5])
        s = make_results("s", [0.8, 0.4])
        router = self._setup_router(g, s)

        plan = SearchPlan(
            sub_queries=[make_sub("test")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        results = router.execute_plan(plan, fixed_embed, session_id="s1")
        scores = [r.score for r in results]
        self.assertEqual(scores, [0.9, 0.8, 0.5, 0.4])

    def test_trims_to_scope_top_k(self):
        """Per-SubQuery results should be trimmed to scope.top_k."""
        g = make_results("g", [0.9, 0.7, 0.5])
        s = make_results("s", [0.8, 0.6, 0.4])
        router = self._setup_router(g, s)

        plan = SearchPlan(
            sub_queries=[make_sub("test", top_k=3)],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=3,
        )
        results = router.execute_plan(plan, fixed_embed, session_id="s1")
        self.assertEqual(len(results), 3)

    def test_interleaved_strategy(self):
        """INTERLEAVED should round-robin across SubQueries."""
        g1 = make_results("g1", [0.9])
        s1 = make_results("s1", [0.85])
        g2 = make_results("g2", [0.8])
        s2 = make_results("s2", [0.75])

        global_cm = make_cm()
        session_cm = make_cm()
        global_cm.search.side_effect = [g1, g2]
        session_cm.search.side_effect = [s1, s2]

        router = CorpusRouter(global_cm)
        router.register_session("s1", session_cm)

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.INTERLEAVED,
            final_top_k=4,
        )
        results = router.execute_plan(plan, fixed_embed, session_id="s1")
        # Round-robin: A[0], B[0], A[1], B[1] — 4 total (capped by final_top_k)
        self.assertEqual(len(results), 4)
        # First from each SubQuery's merged list (global wins by score)
        self.assertEqual(results[0].chunk_id, "g1_0")  # A's top
        self.assertEqual(results[1].chunk_id, "g2_0")  # B's top

    def test_sectioned_strategy(self):
        """SECTIONED should label-prefix chunk_ids."""
        g = make_results("g", [0.9])
        s = make_results("s", [0.8])

        global_cm = make_cm()
        session_cm = make_cm()
        global_cm.search.return_value = g
        session_cm.search.return_value = s

        router = CorpusRouter(global_cm)
        router.register_session("s1", session_cm)

        plan = SearchPlan(
            sub_queries=[make_sub("2023")],
            merge_strategy=MergeStrategy.SECTIONED,
            final_top_k=5,
        )
        results = router.execute_plan(plan, fixed_embed, session_id="s1")
        # SECTIONED prefixes chunk_ids with [label]:
        self.assertTrue(results[0].chunk_id.startswith("[2023]:"))

    def test_session_corpus_contributes_results(self):
        """When global returns empty, session results should still appear."""
        g = []
        s = make_results("s", [0.8, 0.6])
        router = self._setup_router(g, s)

        plan = SearchPlan(
            sub_queries=[make_sub("test")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        results = router.execute_plan(plan, fixed_embed, session_id="s1")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "s_0")


if __name__ == "__main__":
    unittest.main()
