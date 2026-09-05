from pydantic import BaseModel, Field


class HealthResponseData(BaseModel):
    status: str = Field(default="ok", description="Service health status")
    indexed: bool = Field(..., description="Whether global corpus is indexed")
    num_chunks: int = Field(..., description="Total chunks in global index")
    generation_ready: bool = Field(..., description="Whether LLM client is ready")
