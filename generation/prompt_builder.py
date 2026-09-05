"""
FinSight AI - Prompt Builder
==============================
Builds the prompt for the OpenAI API from retrieved evidence.

This module handles:
- Converting retrieval results into a formatted context block
- Assembling the system prompt with grounding rules
- Extracting citations from the model's response

Author: FinSight AI Team
Phase: 2 (Generation Layer)
"""

import re
from typing import List, Tuple


# =============================================================================
# SYSTEM PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are FinSight AI, a professional financial analyst AI specialized in Indian financial documents (DRHP, RHP, Annual Reports). You generate precise, structured, and high-quality responses based strictly on provided context.

GROUNDING RULES — YOU MUST FOLLOW THESE:
1. ONLY answer using the CONTEXT provided below. Do NOT use any external knowledge. Do NOT hallucinate.
2. Answer the question based on ALL relevant information available in the context, even if the context does not use the exact same phrasing as the question.
3. If the context contains ANY information related to the question, use it to form your answer.
4. ONLY say "Not available in the provided documents." if the context is entirely unrelated to the question.
5. If no context chunks are provided below (empty CONTEXT section), respond ONLY with: "Not available in the provided documents."
6. When answering, cite the relevant chunk IDs (e.g., chunk_12) that support your answer.
7. If only partial information is available, answer with what is present and note the limitation.
8. Do NOT fabricate facts. Every claim must be traceable to a specific chunk.

OUTPUT STRUCTURE — MANDATORY:
Always format your response using these sections in order:
1. **## Overview** — Concise 2–3 line summary of the answer.
2. **## Data / Details** — Present all numerical or structured data in clean tables. If no tabular data applies, use clear bullet points with **bold** labels.
3. **## Key Insights** — Provide 3–5 analytical insights (not just restating data). Identify concentration, trends, anomalies, dominance of entities.
4. **## Interpretation** — Short professional interpretation of what the data implies.

TABLE RULES — VERY IMPORTANT:
1. ALWAYS use tables for: shareholding, financial comparisons, numerical breakdowns, multi-entity data.
2. Table format MUST be: | Column 1 | Column 2 | Column 3 | with proper header separator row.
3. Ensure proper alignment, clean column names, no clutter.
4. If the query involves comparison, you MUST include a comparison table with differences and insights.

NUMERICAL RULES:
1. Never display misleading values like "0.0%" — replace with "<0.1%" if applicable.
2. Round percentages to 1 decimal place unless exact precision is needed.
3. Always compute totals when possible (e.g., total promoter holding).
4. Highlight important numbers using **bold**.

ANALYSIS RULES — CRITICAL:
1. Do NOT just describe the data — analyze it.
2. Identify: concentration of ownership, trends, anomalies, dominance of entities.
3. If entities belong to the same promoter group, group them logically in insights.
4. Distinguish between individual holdings and entity holdings.
5. Maintain a professional, analytical tone at all times.

STRICTLY AVOID:
- Unstructured paragraphs of raw data
- Repeating the same data across sections
- Nested bullet clutter (keep nesting to ONE level maximum)
- Raw chunk text pasted directly
- Generic statements like "data shows..."

FOLLOW-UP QUESTIONS:
After your main answer, suggest exactly 3 follow-up questions the user might ask next based on the context.
Format them on the LAST lines of your response as:
[FOLLOW_UP]: First follow-up question here
[FOLLOW_UP]: Second follow-up question here
[FOLLOW_UP]: Third follow-up question here

CONTEXT:
{context}
"""


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def build_context(results: list) -> Tuple[str, List[str]]:
    """
    Convert retrieval results into a formatted context block.
    
    Takes the Top-K retrieval results and formats them as:
        [chunk_0]: Text of chunk 0...
        [chunk_5]: Text of chunk 5...
        [chunk_12]: Text of chunk 12...
    
    This labeled format helps the LLM:
    - Know which chunk each piece of information comes from
    - Cite specific chunks in its answer
    - Stay grounded to the provided context
    
    Args:
        results: List of retrieval results (each has chunk_id, score, snippet)
        
    Returns:
        Tuple of:
        - context_string: Formatted context block
        - chunk_ids: List of chunk IDs included in the context
    """
    context_parts = []
    chunk_ids = []
    
    for result in results:
        # Format each chunk with its ID as a label
        # The label helps the LLM cite sources
        context_parts.append(f"[{result.chunk_id}]: {result.snippet}")
        chunk_ids.append(result.chunk_id)
    
    # Join all chunks with double newlines for readability
    context_string = "\n\n".join(context_parts)
    
    return context_string, chunk_ids


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(context: str, question: str) -> Tuple[str, str]:
    """
    Assemble the full prompt for the OpenAI API.
    
    The prompt has two parts:
    1. System prompt: Contains grounding rules + context
    2. User message: The actual question
    
    Why separate system and user messages?
    - OpenAI's API treats them differently
    - System messages set the "persona" and rules
    - User messages are the actual input
    - This separation helps the model follow instructions better
    
    Args:
        context: The formatted context block from build_context()
        question: The user's original question
        
    Returns:
        Tuple of (system_prompt, user_message)
    """
    # Insert the context into the system prompt template
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    # The user message is just the question
    user_message = question
    
    return system_prompt, user_message


# =============================================================================
# CITATION EXTRACTOR
# =============================================================================

def extract_citations(answer: str, available_chunk_ids: List[str]) -> List[str]:
    """
    Extract cited chunk IDs from the model's answer.
    
    The model is instructed to cite chunks like: chunk_12, chunk_47
    This function finds all such references in the answer text.
    
    We only return chunk IDs that were actually in the context
    (to prevent hallucinated citations).
    
    Algorithm:
    1. Use regex to find all "chunk_N" patterns in the answer
    2. Filter to only those that were in the provided context
    3. Return unique citations in order of appearance
    
    Args:
        answer: The model's generated answer text
        available_chunk_ids: List of chunk IDs that were in the context
        
    Returns:
        List of cited chunk IDs (only those that exist in context)
    """
    # Find all chunk_N patterns in the answer
    # Pattern: "chunk_" followed by one or more digits
    found = re.findall(r"chunk_\d+", answer)
    
    # Filter to only chunks that were actually in the context
    # This prevents hallucinated citations
    cited = []
    seen = set()
    
    for chunk_id in found:
        if chunk_id in available_chunk_ids and chunk_id not in seen:
            cited.append(chunk_id)
            seen.add(chunk_id)
    
    return cited
