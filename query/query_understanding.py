"""
query_understanding.py
----------------------
Deterministic, rule-based query parser for FinSight AI.

Layer  : Above SearchPlanBuilder (future), below API
Stage  : 4

Owns:
    ParsedQuery       — immutable structured representation of user intent
    CompanyDetector   — fuzzy match company names against corpus registry
    YearExtractor     — regex extraction of years and ranges
    DocumentTypeDetector — keyword phrase → canonical corpus doc_type
    IntentClassifier  — rule-based intent from entity counts + keywords
    QueryCleaner      — token-aware span removal

Does NOT own:
    FAISS / vectors       (RetrieverPipeline)
    SearchPlan            (search_plan.py)
    LookupIndex           (lookup_index.py)
    CorpusManager         (corpus_manager.py)
    Embedding logic       (RetrieverPipeline)

Design:
    ParsedQuery is a pure linguistic value object.
    It interprets user language — it does NOT plan retrieval.
    All parsing is deterministic and offline-testable.
    No LLM calls. No embedding imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Tuple


# =============================================================================
# PARSED QUERY DATACLASS
# =============================================================================

@dataclass(frozen=True)
class ParsedQuery:
    """
    Immutable structured representation of a user query.

    Fields:
        cleaned_query   — query with detected scope tokens removed.
                          Falls back to raw_query if removal yields empty string.
        companies       — canonical registry keys, exactly as stored in
                          CorpusManager's company_to_docs.
        years           — chronologically sorted fiscal years (["2021", "2022"]).
        document_types  — canonical corpus doc_type keys (e.g. "Annual_Report").
        intent          — one of: single_entity, comparison, temporal, generic.
    """
    cleaned_query:  str
    companies:      List[str]
    years:          List[str]
    document_types: List[str]
    intent:         Literal["single_entity", "comparison", "temporal", "generic"]


# =============================================================================
# DOCUMENT TYPE KEYWORDS
# =============================================================================

# Keys are canonical corpus doc_type values (must match metadata vocabulary).
# Values are lowercase NLP synonyms that users might type.

DOC_TYPE_KEYWORDS = {
    "DRHP":             ["drhp", "draft red herring", "draft prospectus"],
    "Annual_Report":    ["annual report", "annual filing", "yearly report"],
    "Quarterly_Report": ["quarterly report", "quarterly filing",
                         "q1 report", "q2 report", "q3 report", "q4 report",
                         "quarter"],
    "Balance_Sheet":    ["balance sheet"],
    "Profit_Loss":      ["profit and loss", "profit & loss", "p&l",
                         "income statement"],
    "Cash_Flow":        ["cash flow", "cashflow", "cash flow statement"],
}


# =============================================================================
# COMPARISON KEYWORDS
# =============================================================================

_COMPARISON_KEYWORDS = [
    "compare", "comparison", "versus", " vs ", " vs.",
    "difference between", "differences between",
    "how does", "how do",
]


# =============================================================================
# YEAR RANGE BOUNDS
# =============================================================================

_YEAR_MIN = 2000
_YEAR_MAX = 2035


# =============================================================================
# COMPANY DETECTOR
# =============================================================================

def detect_companies(
    query: str,
    known_companies: List[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Detect company names in a query via case-insensitive matching.

    Strategy:
        1. For each known company, attempt case-insensitive word-boundary
           match in the query.  Multi-word names are matched as phrases.
        2. Auto-generate abbreviations for multi-word company names
           (first letter of each word).  These are matched ONLY at word
           boundaries to prevent "practice" matching "ITC".
        3. Returns canonical registry keys (not the matched surface text).

    Args:
        query:            User's raw query string.
        known_companies:  Registry keys from list_available_entities()["companies"].

    Returns:
        (matched_companies, match_spans)
        matched_companies : canonical registry keys, deduplicated.
        match_spans       : list of (start, end) character positions in query.
    """
    matched: List[str] = []
    spans: List[Tuple[int, int]] = []
    query_lower = query.lower()

    # Sort by name length descending so "Reliance Industries" matches
    # before "Reliance" and doesn't leave a partial span.
    sorted_companies = sorted(known_companies, key=len, reverse=True)

    for company in sorted_companies:
        if company in matched:
            continue

        company_lower = company.lower()

        # --- Full name match (word-boundary) ---
        pattern = re.compile(
            r'\b' + re.escape(company_lower) + r'\b',
            re.IGNORECASE,
        )
        for m in pattern.finditer(query):
            if not _overlaps_existing(m.start(), m.end(), spans):
                matched.append(company)
                spans.append((m.start(), m.end()))

        # --- Abbreviation match (multi-word names only) ---
        words = company.split()
        if len(words) >= 2:
            abbrev = "".join(w[0] for w in words).upper()
            if len(abbrev) >= 2:
                abbrev_pattern = re.compile(
                    r'\b' + re.escape(abbrev) + r'\b',
                )
                for m in abbrev_pattern.finditer(query):
                    if not _overlaps_existing(m.start(), m.end(), spans):
                        if company not in matched:
                            matched.append(company)
                        spans.append((m.start(), m.end()))

    return matched, spans


def _overlaps_existing(
    start: int,
    end: int,
    existing: List[Tuple[int, int]],
) -> bool:
    """Check whether a new span overlaps any existing span."""
    for s, e in existing:
        if start < e and end > s:
            return True
    return False


# =============================================================================
# YEAR EXTRACTOR
# =============================================================================

def extract_years(query: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Extract fiscal years from a query string.

    Patterns handled (case-insensitive):
        - Standalone 4-digit years: "2023"
        - Ranges with dash:         "2021-2024"
        - Ranges with "to":         "2021 to 2024"
        - FY prefix:                "FY2023", "FY 2023"
        - FY short range:           "FY 2022-23", "FY2022-23"

    Range normalization:
        Reversed ranges ("2024 to 2021") are sorted before expansion.

    Returns:
        (years, match_spans)
        years       : chronologically sorted, deduplicated list of year strings.
        match_spans : character positions of matched tokens.
    """
    years: List[str] = []
    spans: List[Tuple[int, int]] = []

    # --- FY short range: "FY 2022-23" or "FY2022-23" ---
    fy_short = re.compile(
        r'\bFY\s?(\d{4})-(\d{2})\b', re.IGNORECASE,
    )
    for m in fy_short.finditer(query):
        # "FY 2022-23" → ending year = 2023
        base_century = m.group(1)[:2]
        end_year = int(base_century + m.group(2))
        if _YEAR_MIN <= end_year <= _YEAR_MAX:
            years.append(str(end_year))
            spans.append((m.start(), m.end()))

    # --- FY standalone: "FY2023" or "FY 2023" ---
    fy_single = re.compile(
        r'\bFY\s?(\d{4})\b', re.IGNORECASE,
    )
    for m in fy_single.finditer(query):
        if not _overlaps_existing(m.start(), m.end(), spans):
            year = int(m.group(1))
            if _YEAR_MIN <= year <= _YEAR_MAX:
                years.append(str(year))
                spans.append((m.start(), m.end()))

    # --- Full range: "2021-2024" or "2021 to 2024" ---
    range_pattern = re.compile(
        r'\b(\d{4})\s*(?:-|to)\s*(\d{4})\b', re.IGNORECASE,
    )
    for m in range_pattern.finditer(query):
        if not _overlaps_existing(m.start(), m.end(), spans):
            y1, y2 = int(m.group(1)), int(m.group(2))
            lo, hi = sorted([y1, y2])  # Fix 3: normalize reversed ranges
            if _YEAR_MIN <= lo and hi <= _YEAR_MAX:
                for y in range(lo, hi + 1):
                    years.append(str(y))
                spans.append((m.start(), m.end()))

    # --- Standalone year: "2023" ---
    single_year = re.compile(r'\b(\d{4})\b')
    for m in single_year.finditer(query):
        if not _overlaps_existing(m.start(), m.end(), spans):
            year = int(m.group(1))
            if _YEAR_MIN <= year <= _YEAR_MAX:
                years.append(str(year))
                spans.append((m.start(), m.end()))

    # Deduplicate and sort chronologically
    seen = set()
    unique: List[str] = []
    for y in years:
        if y not in seen:
            seen.add(y)
            unique.append(y)
    unique.sort()

    return unique, spans


# =============================================================================
# DOCUMENT TYPE DETECTOR
# =============================================================================

def detect_document_types(
    query: str,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Detect document types from keyword phrases.

    Maps NLP synonyms → canonical corpus doc_type keys.
    Multi-word phrases are matched first (longest match wins).

    Returns:
        (doc_types, match_spans)
        doc_types   : canonical doc_type keys matching corpus vocabulary.
        match_spans : character positions of matched tokens.
    """
    doc_types: List[str] = []
    spans: List[Tuple[int, int]] = []
    query_lower = query.lower()

    # Build (synonym, canonical_key) pairs sorted by synonym length desc
    # so "cash flow statement" matches before "cash flow".
    synonym_pairs: List[Tuple[str, str]] = []
    for canonical_key, synonyms in DOC_TYPE_KEYWORDS.items():
        for syn in synonyms:
            synonym_pairs.append((syn, canonical_key))
    synonym_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    for synonym, canonical_key in synonym_pairs:
        # Case-insensitive search
        pattern = re.compile(
            r'\b' + re.escape(synonym) + r'\b',
            re.IGNORECASE,
        )
        for m in pattern.finditer(query_lower):
            if not _overlaps_existing(m.start(), m.end(), spans):
                if canonical_key not in doc_types:
                    doc_types.append(canonical_key)
                spans.append((m.start(), m.end()))

    return doc_types, spans


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

def classify_intent(
    companies: List[str],
    years: List[str],
    query: str,
) -> Literal["single_entity", "comparison", "temporal", "generic"]:
    """
    Rule-based intent classification.

    Priority order (Fix 4 — highest wins):
        1. Explicit comparison keywords → "comparison"
        2. Multiple companies (≥2)      → "comparison"
        3. Multiple years (≥2), ≤1 co.  → "temporal"
        4. Exactly 1 company            → "single_entity"
        5. Fallback                     → "generic"
    """
    query_lower = query.lower()

    # Priority 1: explicit comparison keywords
    has_comparison_keyword = any(
        kw in query_lower for kw in _COMPARISON_KEYWORDS
    )
    if has_comparison_keyword:
        return "comparison"

    # Priority 2: multiple companies
    if len(companies) >= 2:
        return "comparison"

    # Priority 3: multiple years with at most 1 company
    if len(years) >= 2 and len(companies) <= 1:
        return "temporal"

    # Priority 4: single company
    if len(companies) == 1:
        return "single_entity"

    # Priority 5: fallback
    return "generic"


# =============================================================================
# QUERY CLEANER
# =============================================================================

def clean_query(
    query: str,
    match_spans: List[Tuple[int, int]],
) -> str:
    """
    Remove detected scope tokens from query (token-aware).

    Fix 2: Replace each matched span with a single space, then collapse
    multi-spaces and strip.  Never concatenates surrounding characters
    across a removed span.

    Fix 5: If the result is empty, fall back to the original query.

    Args:
        query:        Original query string.
        match_spans:  Character-position spans to remove.

    Returns:
        Cleaned query string (never empty).
    """
    if not match_spans:
        return query.strip()

    # Sort spans by start position
    sorted_spans = sorted(match_spans, key=lambda s: s[0])

    # Merge overlapping/adjacent spans
    merged: List[Tuple[int, int]] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Build result by replacing spans with single space
    parts: List[str] = []
    prev_end = 0
    for start, end in merged:
        parts.append(query[prev_end:start])
        parts.append(" ")  # token-aware: insert space instead of nothing
        prev_end = end
    parts.append(query[prev_end:])

    result = "".join(parts)

    # Collapse multi-spaces and strip
    result = re.sub(r'\s+', ' ', result).strip()

    # Fix 5: never return empty
    if not result:
        return query.strip()

    return result


# =============================================================================
# PUBLIC API — PARSE QUERY
# =============================================================================

def parse_query(
    raw_query: str,
    known_companies: List[str],
) -> ParsedQuery:
    """
    Primary entry point for Stage 4 — query understanding.

    Converts a raw user query into a structured ParsedQuery.
    Deterministic, offline-testable, zero retrieval coupling.

    Args:
        raw_query:        User's natural language question.
        known_companies:  Canonical registry keys from
                          CorpusManager.list_available_entities()["companies"].

    Returns:
        ParsedQuery — immutable structured representation.
    """
    all_spans: List[Tuple[int, int]] = []

    # 1. Company detection (first — multi-word names most ambiguous)
    companies, company_spans = detect_companies(raw_query, known_companies)
    all_spans.extend(company_spans)

    # 2. Year extraction (pure regex, unambiguous)
    years, year_spans = extract_years(raw_query)
    all_spans.extend(year_spans)

    # 3. Document type detection (keyword phrases)
    document_types, doctype_spans = detect_document_types(raw_query)
    all_spans.extend(doctype_spans)

    # 4. Intent classification (depends on counts from 1-2 + keywords)
    intent = classify_intent(companies, years, raw_query)

    # 5. Query cleaning (single-pass span removal, always last)
    cleaned = clean_query(raw_query, all_spans)

    return ParsedQuery(
        cleaned_query=cleaned,
        companies=companies,
        years=years,
        document_types=document_types,
        intent=intent,
    )
