import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from bson import ObjectId

from app.core.database import conversations_collection
from app.core.exceptions import NotFoundException, BadRequestException
from app.schemas.conversation import (
    ConversationSummary,
    ConversationDetail,
    MessageItem,
    ConversationListResponseData,
)

logger = logging.getLogger("cognifin.service.conversations")


def _to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise BadRequestException(message="Invalid conversation identifier format.")


def list_user_conversations(user_id: str, page: int = 1, limit: int = 50) -> ConversationListResponseData:
    skip = (page - 1) * limit
    total = conversations_collection.count_documents({"user_id": user_id})

    cursor = (
        conversations_collection.find({"user_id": user_id})
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
    )

    items: List[ConversationSummary] = []
    for doc in cursor:
        items.append(
            ConversationSummary(
                id=str(doc["_id"]),
                title=doc.get("title", "New Chat"),
                updatedAt=doc.get("updated_at", doc.get("created_at", "")).isoformat()
                if isinstance(doc.get("updated_at"), datetime)
                else str(doc.get("updated_at", "")),
                createdAt=doc.get("created_at", "").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else str(doc.get("created_at", "")),
                messageCount=len(doc.get("messages", [])),
            )
        )

    return ConversationListResponseData(conversations=items, total=total, page=page, limit=limit)


def get_user_conversation(conversation_id: str, user_id: str) -> ConversationDetail:
    oid = _to_object_id(conversation_id)
    doc = conversations_collection.find_one({"_id": oid, "user_id": user_id})
    if not doc:
        raise NotFoundException(message="The requested conversation was not found.")

    messages: List[MessageItem] = []
    for m in doc.get("messages", []):
        messages.append(
            MessageItem(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                metadata=m.get("metadata", {}),
                timestamp=m.get("timestamp", "").isoformat()
                if isinstance(m.get("timestamp"), datetime)
                else str(m.get("timestamp", "")),
            )
        )

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


def rename_user_conversation(conversation_id: str, user_id: str, new_title: str) -> Dict[str, Any]:
    oid = _to_object_id(conversation_id)
    result = conversations_collection.update_one(
        {"_id": oid, "user_id": user_id},
        {"$set": {"title": new_title.strip(), "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise NotFoundException(message="The requested conversation was not found.")

    logger.info(f"action=rename_conversation id={conversation_id} user_id={user_id} title='{new_title}'")
    return {"id": conversation_id, "title": new_title.strip()}


def delete_user_conversation(conversation_id: str, user_id: str) -> Dict[str, Any]:
    oid = _to_object_id(conversation_id)
    result = conversations_collection.delete_one({"_id": oid, "user_id": user_id})
    if result.deleted_count == 0:
        raise NotFoundException(message="The requested conversation was not found.")

    logger.info(f"action=delete_conversation id={conversation_id} user_id={user_id} status=deleted")
    return {"id": conversation_id}


def create_conversation(user_id: str, title: str, user_msg: dict, assistant_msg: dict) -> str:
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
    now = datetime.now(timezone.utc)
    oid = _to_object_id(conversation_id)
    conversations_collection.update_one(
        {"_id": oid, "user_id": user_id},
        {
            "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
            "$set": {"updated_at": now},
        },
    )
