"""
test_query_understanding.py
----------------------------
Offline test suite for Stage 4 — query understanding.

No server, no FAISS, no embeddings, no external dependencies.
Tests are deterministic and can run in any environment.

Usage:
    python -m pytest test_query_understanding.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from query.query_understanding import (
    ParsedQuery,
    detect_companies,
    extract_years,
    detect_document_types,
    classify_intent,
    clean_query,
    parse_query,
)


# =============================================================================
# FIXTURES
# =============================================================================

KNOWN_COMPANIES = [
    "TCS",
    "Infosys",
    "Reliance Industries",
    "HDFC Bank",
    "ITC",
    "Wipro",
    "State Bank of India",
]


# =============================================================================
# COMPANY DETECTOR TESTS
# =============================================================================

class TestCompanyDetector:

    def test_single_company(self):
        companies, spans = detect_companies("Tell me about TCS revenue", KNOWN_COMPANIES)
        assert companies == ["TCS"]
        assert len(spans) == 1

    def test_multiple_companies(self):
        companies, _ = detect_companies(
            "Compare TCS and Infosys profits", KNOWN_COMPANIES,
        )
        assert "TCS" in companies
        assert "Infosys" in companies
        assert len(companies) == 2

    def test_multi_word_company(self):
        companies, _ = detect_companies(
            "Reliance Industries annual report", KNOWN_COMPANIES,
        )
        assert companies == ["Reliance Industries"]

    def test_case_insensitive(self):
        companies, _ = detect_companies("tcs revenue growth", KNOWN_COMPANIES)
        assert companies == ["TCS"]

    def test_abbreviation_match(self):
        """Multi-word abbreviation: 'RI' for 'Reliance Industries'."""
        companies, _ = detect_companies("RI revenue in 2023", KNOWN_COMPANIES)
        assert companies == ["Reliance Industries"]

    def test_abbreviation_collision_guard(self):
        """'practice' must NOT trigger a match for any company."""
        companies, _ = detect_companies(
            "Best practice in accounting", KNOWN_COMPANIES,
        )
        assert companies == []

    def test_no_company_found(self):
        companies, _ = detect_companies(
            "What are the risk factors?", KNOWN_COMPANIES,
        )
        assert companies == []

    def test_abbreviation_word_boundary(self):
        """'ITC' should match only as an isolated token."""
        companies, _ = detect_companies("ITC profits this year", KNOWN_COMPANIES)
        assert "ITC" in companies

    def test_no_false_positive_in_longer_word(self):
        """Abbreviation must not match inside a longer word."""
        companies, _ = detect_companies(
            "The switch from HITCS to modern", KNOWN_COMPANIES,
        )
        # 'TCS' should not match inside 'HITCS'
        assert "TCS" not in companies


# =============================================================================
# YEAR EXTRACTOR TESTS
# =============================================================================

class TestYearExtractor:

    def test_single_year(self):
        years, _ = extract_years("Revenue in 2023")
        assert years == ["2023"]

    def test_multiple_years(self):
        years, _ = extract_years("Compare 2022 and 2023 performance")
        assert years == ["2022", "2023"]

    def test_year_range_dash(self):
        years, _ = extract_years("Growth from 2021-2024")
        assert years == ["2021", "2022", "2023", "2024"]

    def test_year_range_to(self):
        years, _ = extract_years("Revenue 2021 to 2023")
        assert years == ["2021", "2022", "2023"]

    def test_reversed_range(self):
        """Fix 3: reversed ranges must be sorted before expansion."""
        years, _ = extract_years("Growth from 2024 to 2021")
        assert years == ["2021", "2022", "2023", "2024"]

    def test_fy_single(self):
        years, _ = extract_years("FY2023 annual report")
        assert years == ["2023"]

    def test_fy_with_space(self):
        years, _ = extract_years("FY 2023 revenue")
        assert years == ["2023"]

    def test_fy_short_range(self):
        """'FY 2022-23' → ending year 2023."""
        years, _ = extract_years("FY 2022-23 annual report")
        assert years == ["2023"]

    def test_no_year(self):
        years, _ = extract_years("What are the risk factors?")
        assert years == []

    def test_out_of_range_year(self):
        """Years outside 2000-2035 should not match."""
        years, _ = extract_years("Revenue in 1999 and 2036")
        assert years == []


# =============================================================================
# DOCUMENT TYPE DETECTOR TESTS
# =============================================================================

class TestDocumentTypeDetector:

    def test_annual_report(self):
        types, _ = detect_document_types("Show me the annual report")
        assert types == ["Annual_Report"]

    def test_balance_sheet(self):
        types, _ = detect_document_types("Balance sheet of the company")
        assert types == ["Balance_Sheet"]

    def test_profit_loss(self):
        types, _ = detect_document_types("profit and loss statement")
        assert types == ["Profit_Loss"]

    def test_pl_abbreviation(self):
        types, _ = detect_document_types("Show me the P&L")
        assert types == ["Profit_Loss"]

    def test_cash_flow(self):
        types, _ = detect_document_types("cash flow statement analysis")
        assert types == ["Cash_Flow"]

    def test_drhp(self):
        types, _ = detect_document_types("DRHP highlights")
        assert types == ["DRHP"]

    def test_no_doc_type(self):
        types, _ = detect_document_types("What are the risk factors?")
        assert types == []

    def test_multiple_doc_types(self):
        types, _ = detect_document_types(
            "Compare balance sheet and cash flow"
        )
        assert "Balance_Sheet" in types
        assert "Cash_Flow" in types


# =============================================================================
# INTENT CLASSIFIER TESTS
# =============================================================================

class TestIntentClassifier:

    def test_comparison_multiple_companies(self):
        intent = classify_intent(
            companies=["TCS", "Infosys"],
            years=["2023"],
            query="Compare TCS and Infosys",
        )
        assert intent == "comparison"

    def test_comparison_keyword_single_company(self):
        """Fix 4: comparison keyword overrides even with 1 company."""
        intent = classify_intent(
            companies=["TCS"],
            years=["2022", "2023"],
            query="difference between 2022 and 2023 revenue of TCS",
        )
        assert intent == "comparison"

    def test_comparison_keyword_no_company(self):
        """Comparison keyword alone triggers comparison intent."""
        intent = classify_intent(
            companies=[],
            years=["2022", "2023"],
            query="Compare 2022 and 2023 revenue",
        )
        assert intent == "comparison"

    def test_temporal(self):
        intent = classify_intent(
            companies=["TCS"],
            years=["2021", "2022", "2023"],
            query="TCS revenue from 2021 to 2023",
        )
        # No comparison keyword, 1 company, multi-year → temporal
        assert intent == "temporal"

    def test_single_entity(self):
        intent = classify_intent(
            companies=["TCS"],
            years=["2023"],
            query="TCS revenue in 2023",
        )
        assert intent == "single_entity"

    def test_generic(self):
        intent = classify_intent(
            companies=[],
            years=[],
            query="What are the risk factors?",
        )
        assert intent == "generic"

    def test_vs_keyword(self):
        intent = classify_intent(
            companies=["TCS"],
            years=[],
            query="TCS vs market average",
        )
        assert intent == "comparison"


# =============================================================================
# QUERY CLEANER TESTS
# =============================================================================

class TestQueryCleaner:

    def test_basic_cleaning(self):
        result = clean_query(
            "What is the revenue of TCS in 2023",
            [(23, 26), (30, 34)],  # "TCS", "2023"
        )
        assert "TCS" not in result
        assert "2023" not in result
        assert "revenue" in result

    def test_no_broken_tokens(self):
        """Fix 2: span removal inserts space, never concatenates."""
        result = clean_query("TCS growth", [(0, 3)])
        # "TCS" removed → " growth" → "growth" after strip
        assert result == "growth"

    def test_collapse_whitespace(self):
        result = clean_query(
            "What is the revenue of TCS in 2023",
            [(23, 26), (30, 34)],
        )
        assert "  " not in result

    def test_empty_fallback(self):
        """Fix 5: if cleaning empties the query, fall back to original."""
        raw = "TCS 2023"
        result = clean_query(raw, [(0, 3), (4, 8)])
        assert result == raw.strip()

    def test_no_spans(self):
        result = clean_query("What are the risk factors?", [])
        assert result == "What are the risk factors?"


# =============================================================================
# FULL PIPELINE (parse_query) TESTS
# =============================================================================

class TestParseQuery:

    def test_single_entity_query(self):
        result = parse_query("TCS revenue in 2023", KNOWN_COMPANIES)
        assert result.companies == ["TCS"]
        assert result.years == ["2023"]
        assert result.intent == "single_entity"
        assert "TCS" not in result.cleaned_query
        assert "2023" not in result.cleaned_query

    def test_comparison_query(self):
        result = parse_query(
            "Compare TCS and Infosys profits in 2023", KNOWN_COMPANIES,
        )
        assert "TCS" in result.companies
        assert "Infosys" in result.companies
        assert result.years == ["2023"]
        assert result.intent == "comparison"

    def test_temporal_query(self):
        result = parse_query(
            "TCS revenue from 2021 to 2023", KNOWN_COMPANIES,
        )
        assert result.companies == ["TCS"]
        assert result.years == ["2021", "2022", "2023"]
        assert result.intent == "temporal"

    def test_generic_query(self):
        result = parse_query(
            "What are the risk factors?", KNOWN_COMPANIES,
        )
        assert result.companies == []
        assert result.years == []
        assert result.document_types == []
        assert result.intent == "generic"
        assert result.cleaned_query == "What are the risk factors?"

    def test_doc_type_with_company(self):
        result = parse_query(
            "HDFC Bank annual report for 2023", KNOWN_COMPANIES,
        )
        assert result.companies == ["HDFC Bank"]
        assert result.document_types == ["Annual_Report"]
        assert result.years == ["2023"]
        assert result.intent == "single_entity"

    def test_fy_pattern_integration(self):
        result = parse_query(
            "FY 2022-23 annual report of TCS", KNOWN_COMPANIES,
        )
        assert result.years == ["2023"]
        assert result.document_types == ["Annual_Report"]
        assert result.companies == ["TCS"]

    def test_cleaned_query_is_not_empty(self):
        """Full scope-only query falls back to raw."""
        result = parse_query("TCS 2023", KNOWN_COMPANIES)
        assert result.cleaned_query.strip() != ""

    def test_immutability(self):
        """ParsedQuery is frozen — any mutation attempt raises."""
        result = parse_query("TCS revenue in 2023", KNOWN_COMPANIES)
        with pytest.raises(AttributeError):
            result.intent = "generic"  # type: ignore

    def test_keyword_comparison_single_company(self):
        """Fix 4: comparison keyword overrides single-entity."""
        result = parse_query(
            "difference between 2022 and 2023 revenue of Infosys",
            KNOWN_COMPANIES,
        )
        assert result.intent == "comparison"
        assert result.companies == ["Infosys"]

    def test_reversed_year_range_integration(self):
        """Fix 3: reversed range normalized in full pipeline."""
        result = parse_query(
            "Wipro growth from 2024 to 2021", KNOWN_COMPANIES,
        )
        assert result.years == ["2021", "2022", "2023", "2024"]
