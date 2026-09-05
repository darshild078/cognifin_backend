from typing import Dict, Any
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.schemas.common import ApiResponse, success_response
from app.schemas.conversation import (
    ConversationListResponseData,
    ConversationDetail,
    RenameRequest,
)
from app.services.conversation_service import (
    list_user_conversations,
    get_user_conversation,
    rename_user_conversation,
    delete_user_conversation,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=ApiResponse[ConversationListResponseData])
def list_conversations(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = list_user_conversations(user_id=current_user["user_id"], page=page, limit=limit)
    return success_response(message="Conversations fetched successfully.", data=result)


@router.get("/{conversation_id}", response_model=ApiResponse[ConversationDetail])
def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = get_user_conversation(conversation_id=conversation_id, user_id=current_user["user_id"])
    return success_response(message="Conversation fetched successfully.", data=result)


@router.patch("/{conversation_id}", response_model=ApiResponse[Dict[str, Any]])
def rename_conversation(
    conversation_id: str,
    payload: RenameRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = rename_user_conversation(
        conversation_id=conversation_id,
        user_id=current_user["user_id"],
        new_title=payload.title,
    )
    return success_response(message="Conversation renamed successfully.", data=result)


@router.delete("/{conversation_id}", response_model=ApiResponse[Dict[str, Any]])
def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = delete_user_conversation(conversation_id=conversation_id, user_id=current_user["user_id"])
    return success_response(message="Conversation deleted successfully.", data=result)
