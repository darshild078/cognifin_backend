from fastapi import APIRouter

from app.schemas.common import ApiResponse, success_response
from app.schemas.retrieval import RetrieveRequest, RetrieveResponseData
from app.services.rag_service import rag_service

router = APIRouter(tags=["Retrieval"])


@router.post("/retrieve", response_model=ApiResponse[RetrieveResponseData])
def retrieve(request: RetrieveRequest):
    result = rag_service.retrieve(request=request)
    return success_response(message="Passages retrieved successfully.", data=result)
