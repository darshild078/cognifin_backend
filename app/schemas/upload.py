from pydantic import BaseModel, Field


class UploadResponseData(BaseModel):
    session_id: str = Field(..., description="Session identifier for uploaded document")
    chunks: int = Field(..., description="Number of indexed chunks")
    company: str = Field(..., description="Company name")
    year: str = Field(..., description="Filing year")
    document_type: str = Field(..., description="Document type")
