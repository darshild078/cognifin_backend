from typing import List, Optional
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The query to search in the document corpus")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results to return (default: 5)")
    session_id: Optional[str] = Field(default=None, description="Session ID from uploaded PDF")


class RetrieveResultItem(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Relevance / similarity score")
    snippet: str = Field(..., description="Text content snippet")


class RetrieveResponseData(BaseModel):
    query: str = Field(..., description="Original search query")
    top_k: int = Field(..., description="Number of retrieved items")
    results: List[RetrieveResultItem] = Field(default_factory=list, description="Retrieved passages")
    filtered_count: int = Field(default=0, description="Number of items filtered below threshold")
