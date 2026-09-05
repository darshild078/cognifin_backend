from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Thread title")
    updatedAt: str = Field(..., description="Last updated ISO timestamp")
    createdAt: str = Field(..., description="Created ISO timestamp")
    messageCount: int = Field(default=0, description="Total message count")


class MessageItem(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Citations and evidence metadata")
    timestamp: str = Field(default="", description="ISO timestamp")


class ConversationDetail(BaseModel):
    id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    messages: List[MessageItem] = Field(default_factory=list, description="Chronological message list")
    created_at: str = Field(..., description="Created ISO timestamp")
    updated_at: str = Field(..., description="Updated ISO timestamp")


class ConversationListResponseData(BaseModel):
    conversations: List[ConversationSummary] = Field(default_factory=list, description="Conversation summaries")
    total: int = Field(..., description="Total matching conversations")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Page limit")


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="New conversation title")
