from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to ask grounded in financial filings")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of evidence chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Optional uploaded session ID")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID to append to")


class EvidenceItem(BaseModel):
    chunk_id: str = Field(..., description="Chunk identifier")
    snippet: str = Field(..., description="Text content")
    page_number: int = Field(default=0, description="1-based page number")
    document_label: str = Field(default="", description="Source document label")
    pdf_url: str = Field(default="", description="Basename of source PDF")


class ChatResponseData(BaseModel):
    answer: str = Field(..., description="Grounded AI generated answer")
    citations: List[str] = Field(default_factory=list, description="Chunk IDs cited in the answer")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Evidence passages used")
    conversation_id: Optional[str] = Field(default=None, description="Persistent conversation ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata (confidence, latency, intent)")
    follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
