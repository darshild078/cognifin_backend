"""
FinSight AI - Conversations Router
====================================
User-scoped conversation CRUD endpoints.

Every conversation is tied to a user_id (from JWT).
Users can only access their own conversations.

Endpoints:
  GET    /conversations          — List conversations for current user
  GET    /conversations/{id}     — Get single conversation with messages
  DELETE /conversations/{id}     — Delete a conversation
"""

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth import get_current_user
from db import conversations_collection


router = APIRouter(tags=["Conversations"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ConversationSummary(BaseModel):
    """Lightweight conversation item for sidebar listing."""
    id: str
    title: str
    updatedAt: str
    createdAt: str
    messageCount: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int
    page: int
    limit: int


class MessageItem(BaseModel):
    role: str
    content: str
    metadata: dict = {}
    timestamp: str = ""


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[MessageItem]
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oid(id_str: str) -> ObjectId:
    """Convert string to ObjectId, raise 400 on invalid format."""
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    """
    List conversations for the authenticated user.

    Returns lightweight summaries (no full message bodies) sorted by
    most-recently-updated first.
    """
    user_id = current_user["user_id"]
    skip = (page - 1) * limit

    # Count total for pagination
    total = conversations_collection.count_documents({"user_id": user_id})

    # Fetch summaries (exclude full message content for performance)
    cursor = (
        conversations_collection
        .find({"user_id": user_id})
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
    )

    items = []
    for doc in cursor:
        items.append(ConversationSummary(
            id=str(doc["_id"]),
            title=doc.get("title", "New Chat"),
            updatedAt=doc.get("updated_at", doc.get("created_at", "")).isoformat()
                if isinstance(doc.get("updated_at"), datetime)
                else str(doc.get("updated_at", "")),
            createdAt=doc.get("created_at", "").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else str(doc.get("created_at", "")),
            messageCount=len(doc.get("messages", [])),
        ))

    return ConversationListResponse(
        conversations=items,
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get a single conversation with full message history.

    Only the owner (matched by user_id) can access it.
    """
    user_id = current_user["user_id"]
    oid = _oid(conversation_id)

    doc = conversations_collection.find_one({"_id": oid, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    for m in doc.get("messages", []):
        messages.append(MessageItem(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            metadata=m.get("metadata", {}),
            timestamp=m.get("timestamp", "").isoformat()
                if isinstance(m.get("timestamp"), datetime)
                else str(m.get("timestamp", "")),
        ))

    return ConversationDetail(
        id=str(doc["_id"]),
        title=doc.get("title", "New Chat"),
        messages=messages,
        created_at=doc.get("created_at", "").isoformat()
            if isinstance(doc.get("created_at"), datetime)
            else str(doc.get("created_at", "")),
        updated_at=doc.get("updated_at", "").isoformat()
            if isinstance(doc.get("updated_at"), datetime)
            else str(doc.get("updated_at", "")),
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a conversation. Only the owner can delete it.
    """
    user_id = current_user["user_id"]
    oid = _oid(conversation_id)

    result = conversations_collection.delete_one({"_id": oid, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "deleted", "id": conversation_id}


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    body: RenameRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Rename a conversation. Only the owner can rename it.
    """
    user_id = current_user["user_id"]
    oid = _oid(conversation_id)

    result = conversations_collection.update_one(
        {"_id": oid, "user_id": user_id},
        {"$set": {"title": body.title, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "renamed", "id": conversation_id, "title": body.title}


# ---------------------------------------------------------------------------
# Persistence helpers (used by /chat in main.py)
# ---------------------------------------------------------------------------

def create_conversation(user_id: str, title: str, user_msg: dict, assistant_msg: dict) -> str:
    """
    Create a new conversation document in MongoDB.

    Returns the new conversation's string ID.
    """
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "title": title,
        "messages": [user_msg, assistant_msg],
        "created_at": now,
        "updated_at": now,
    }
    result = conversations_collection.insert_one(doc)
    return str(result.inserted_id)


def append_to_conversation(conversation_id: str, user_id: str, user_msg: dict, assistant_msg: dict):
    """
    Append user + assistant messages to an existing conversation.

    Validates ownership via user_id filter.
    """
    now = datetime.now(timezone.utc)
    oid = _oid(conversation_id)
    conversations_collection.update_one(
        {"_id": oid, "user_id": user_id},
        {
            "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
            "$set": {"updated_at": now},
        },
    )
