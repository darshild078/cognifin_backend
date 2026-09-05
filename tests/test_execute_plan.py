"""
test_execute_plan.py
--------------------
Unit tests for CorpusManager.execute_plan().

Fully self-contained: CorpusManager.search() is mocked so no
FAISS index, RetrieverPipeline, or disk I/O is required.

Run with:
    python -m pytest test_execute_plan.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.metadata_schema import RetrievalResult
from core.lookup_index import RetrievalScope
from query.search_plan import SearchPlan, SubQuery, MergeStrategy
from core.corpus_manager import CorpusManager


# =============================================================================
# HELPERS
# =============================================================================

def make_results(tag: str, n: int):
    """Return n RetrievalResults with tag-namespaced snippet/chunk_id."""
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            score=round(1.0 - i * 0.1, 1),
            snippet=f"{tag}_snippet_{i}",
        )
        for i in range(n)
    ]


def make_cm() -> CorpusManager:
    """Minimal CorpusManager bypassing __init__; search() must be mocked."""
    cm = CorpusManager.__new__(CorpusManager)
    cm.retriever = MagicMock()
    cm.documents = {}
    cm.chunk_metadata = {}
    cm._total_vectors = 0
    cm.lookup_index = MagicMock()
    return cm


def make_scope(label: str, top_k: int = 5) -> RetrievalScope:
    return RetrievalScope(label=label, top_k=top_k)


def make_sub(label: str, query: str = "q", top_k: int = 5) -> SubQuery:
    return SubQuery(
        label=label,
        rewritten_query=query,
        scope=make_scope(label, top_k),
    )


def fixed_embed(_: str) -> np.ndarray:
    """Deterministic embed_query stub — returns a unit vector."""
    return np.ones((1, 384), dtype=np.float32)


# =============================================================================
# SINGLE
# =============================================================================

class TestSingle:

    def test_returns_first_subquery_results(self):
        cm = make_cm()
        expected = make_results("A", 5)
        cm.search = MagicMock(return_value=expected)

        plan = SearchPlan(
            sub_queries=[make_sub("A")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        out = cm.execute_plan(plan, fixed_embed)
        assert out == expected

    def test_trims_to_final_top_k(self):
        cm = make_cm()
        cm.search = MagicMock(return_value=make_results("A", 10))

        plan = SearchPlan(
            sub_queries=[make_sub("A", top_k=10)],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=3,
        )
        out = cm.execute_plan(plan, fixed_embed)
        assert len(out) == 3

    def test_empty_subquery_results(self):
        cm = make_cm()
        cm.search = MagicMock(return_value=[])

        plan = SearchPlan(
            sub_queries=[make_sub("A")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        out = cm.execute_plan(plan, fixed_embed)
        assert out == []


# =============================================================================
# INTERLEAVED
# =============================================================================

class TestInterleaved:

    def test_round_robin_order(self):
        cm = make_cm()
        cm.search = MagicMock(side_effect=[
            make_results("A", 3),
            make_results("B", 3),
        ])

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.INTERLEAVED,
            final_top_k=6,
        )
        out = cm.execute_plan(plan, fixed_embed)

        # Strict round-robin: A0 B0 A1 B1 A2 B2
        assert [r.snippet for r in out] == [
            "A_snippet_0", "B_snippet_0",
            "A_snippet_1", "B_snippet_1",
            "A_snippet_2", "B_snippet_2",
        ]

    def test_caps_at_final_top_k(self):
        cm = make_cm()
        cm.search = MagicMock(side_effect=[
            make_results("A", 5),
            make_results("B", 5),
        ])

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.INTERLEAVED,
            final_top_k=4,
        )
        out = cm.execute_plan(plan, fixed_embed)
        assert len(out) == 4

    def test_uneven_lists_shorter_exhausts(self):
        """A has 1 result, B has 2. After A drains, B continues."""
        cm = make_cm()
        cm.search = MagicMock(side_effect=[
            make_results("A", 1),
            make_results("B", 2),
        ])

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.INTERLEAVED,
            final_top_k=10,
        )
        out = cm.execute_plan(plan, fixed_embed)

        # Round 0: A0, B0 — Round 1: (A empty), B1 — Round 2: both empty → stop
        assert len(out) == 3
        assert out[0].snippet == "A_snippet_0"
        assert out[1].snippet == "B_snippet_0"
        assert out[2].snippet == "B_snippet_1"

    def test_both_empty(self):
        cm = make_cm()
        cm.search = MagicMock(side_effect=[[], []])

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.INTERLEAVED,
            final_top_k=5,
        )
        out = cm.execute_plan(plan, fixed_embed)
        assert out == []


# =============================================================================
# SECTIONED
# =============================================================================

class TestSectioned:

    def test_label_prefix_on_chunk_id(self):
        cm = make_cm()
        cm.search = MagicMock(side_effect=[
            make_results("tcs", 2),
            make_results("infy", 2),
        ])

        plan = SearchPlan(
            sub_queries=[make_sub("tcs"), make_sub("infy")],
            merge_strategy=MergeStrategy.SECTIONED,
            final_top_k=4,
        )
        out = cm.execute_plan(plan, fixed_embed)

        assert out[0].chunk_id == "[tcs]:chunk_0"
        assert out[1].chunk_id == "[tcs]:chunk_1"
        assert out[2].chunk_id == "[infy]:chunk_0"
        assert out[3].chunk_id == "[infy]:chunk_1"

    def test_sequential_fill_then_cap(self):
        cm = make_cm()
        cm.search = MagicMock(side_effect=[
            make_results("A", 5),
            make_results("B", 5),
        ])

        plan = SearchPlan(
            sub_queries=[make_sub("A"), make_sub("B")],
            merge_strategy=MergeStrategy.SECTIONED,
            final_top_k=3,
        )
        out = cm.execute_plan(plan, fixed_embed)

        # All 3 come from A (first section, not yet exhausted)
        assert len(out) == 3
        assert all("[A]:" in r.chunk_id for r in out)

    def test_dc_replace_does_not_mutate_original(self):
        """dc_replace must return a new object; original must be unchanged."""
        cm = make_cm()
        source = make_results("A", 2)
        original_ids = [r.chunk_id for r in source]
        cm.search = MagicMock(return_value=source)

        plan = SearchPlan(
            sub_queries=[make_sub("A")],
            merge_strategy=MergeStrategy.SECTIONED,
            final_top_k=2,
        )
        cm.execute_plan(plan, fixed_embed)

        # Original list must be untouched
        assert [r.chunk_id for r in source] == original_ids


# =============================================================================
# EMBED_QUERY CONTRACT
# =============================================================================

class TestEmbedQueryContract:

    def test_called_once_per_subquery(self):
        cm = make_cm()
        cm.search = MagicMock(return_value=[])
        spy = MagicMock(return_value=np.ones((1, 384), dtype=np.float32))

        plan = SearchPlan(
            sub_queries=[
                make_sub("A", query="q1"),
                make_sub("B", query="q2"),
                make_sub("C", query="q3"),
            ],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        cm.execute_plan(plan, spy)
        assert spy.call_count == 3

    def test_called_with_rewritten_query_string(self):
        cm = make_cm()
        cm.search = MagicMock(return_value=[])
        spy = MagicMock(return_value=np.ones((1, 384), dtype=np.float32))

        plan = SearchPlan(
            sub_queries=[make_sub("A", query="revenue growth 2023")],
            merge_strategy=MergeStrategy.SINGLE,
            final_top_k=5,
        )
        cm.execute_plan(plan, spy)
        spy.assert_called_once_with("revenue growth 2023")
