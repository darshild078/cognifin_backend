from typing import Dict, Any
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.schemas.common import ApiResponse, success_response
from app.schemas.chat import ChatRequest, ChatResponseData
from app.services.rag_service import rag_service

router = APIRouter(tags=["Generation"])


@router.post("/chat", response_model=ApiResponse[ChatResponseData])
def chat(request: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    result = rag_service.chat(request=request, user_id=current_user["user_id"])
    return success_response(message="Answer generated successfully.", data=result)
