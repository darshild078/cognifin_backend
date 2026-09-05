import logging
from typing import Dict, Any, List, Optional
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from app.core.config import settings

logger = logging.getLogger("cognifin.db")

_mongo_available = False

# Initialize MongoDB client
try:
    client = MongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=1000,
        connectTimeoutMS=1000,
    )
    db = client[settings.MONGO_DB_NAME]
    _raw_users = db["users"]
    _raw_conversations = db["conversations"]
except Exception:
    client = None
    db = None
    _raw_users = None
    _raw_conversations = None


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Fallback Storage (Seamless Operation when MongoDB is offline)
# ─────────────────────────────────────────────────────────────────────────────

class InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class UpdateResult:
    def __init__(self, matched_count: int, modified_count: int):
        self.matched_count = matched_count
        self.modified_count = modified_count


class DeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


class InMemoryCursor:
    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items
        self._sort_key = None
        self._sort_dir = 1
        self._skip = 0
        self._limit = None

    def sort(self, key, direction=1):
        self._sort_key = key
        self._sort_dir = direction
        return self

    def skip(self, count: int):
        self._skip = count
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def __iter__(self):
        items = list(self._items)
        if self._sort_key:
            items.sort(
                key=lambda x: (x.get(self._sort_key) is not None, x.get(self._sort_key)),
                reverse=(self._sort_dir == -1),
            )
        if self._skip:
            items = items[self._skip:]
        if self._limit is not None:
            items = items[:self._limit]
        return iter(items)


class InMemoryCollection:
    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, Dict[str, Any]] = {}

    def _matches(self, doc: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        for k, v in filter_dict.items():
            doc_val = doc.get(k)
            if isinstance(v, ObjectId) or isinstance(doc_val, ObjectId):
                if str(doc_val) != str(v):
                    return False
            elif doc_val != v:
                return False
        return True

    def find_one(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for doc in self._data.values():
            if self._matches(doc, filter_dict):
                return dict(doc)
        return None

    def insert_one(self, doc: Dict[str, Any]) -> InsertResult:
        doc_copy = dict(doc)
        if "_id" not in doc_copy:
            doc_copy["_id"] = ObjectId()
        key = str(doc_copy["_id"])
        self._data[key] = doc_copy
        return InsertResult(doc_copy["_id"])

    def update_one(self, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> UpdateResult:
        for key, doc in self._data.items():
            if self._matches(doc, filter_dict):
                if "$set" in update_dict:
                    for sk, sv in update_dict["$set"].items():
                        doc[sk] = sv
                if "$push" in update_dict:
                    for pk, pv in update_dict["$push"].items():
                        if pk not in doc or not isinstance(doc[pk], list):
                            doc[pk] = []
                        if isinstance(pv, dict) and "$each" in pv:
                            doc[pk].extend(pv["$each"])
                        else:
                            doc[pk].append(pv)
                return UpdateResult(1, 1)
        return UpdateResult(0, 0)

    def delete_one(self, filter_dict: Dict[str, Any]) -> DeleteResult:
        for key, doc in list(self._data.items()):
            if self._matches(doc, filter_dict):
                del self._data[key]
                return DeleteResult(1)
        return DeleteResult(0)

    def count_documents(self, filter_dict: Dict[str, Any]) -> int:
        return sum(1 for doc in self._data.values() if self._matches(doc, filter_dict))

    def find(self, filter_dict: Dict[str, Any]) -> InMemoryCursor:
        matched = [dict(doc) for doc in self._data.values() if self._matches(doc, filter_dict)]
        return InMemoryCursor(matched)

    def create_index(self, *args, **kwargs):
        pass


class SafeCollection:
    """
    Transparent Proxy Collection:
    Uses real MongoDB when available; seamlessly falls back to InMemoryCollection when MongoDB is offline.
    """

    def __init__(self, raw_col, fallback_col: InMemoryCollection):
        self._raw = raw_col
        self._fallback = fallback_col

    def _execute(self, method_name: str, *args, **kwargs):
        global _mongo_available
        if _mongo_available and self._raw is not None:
            try:
                method = getattr(self._raw, method_name)
                return method(*args, **kwargs)
            except (ServerSelectionTimeoutError, PyMongoError) as e:
                logger.warning(f"MongoDB query failed ({e}) — switching to in-memory store.")
                _mongo_available = False

        method = getattr(self._fallback, method_name)
        return method(*args, **kwargs)

    def find_one(self, *args, **kwargs):
        return self._execute("find_one", *args, **kwargs)

    def insert_one(self, *args, **kwargs):
        return self._execute("insert_one", *args, **kwargs)

    def update_one(self, *args, **kwargs):
        return self._execute("update_one", *args, **kwargs)

    def delete_one(self, *args, **kwargs):
        return self._execute("delete_one", *args, **kwargs)

    def count_documents(self, *args, **kwargs):
        return self._execute("count_documents", *args, **kwargs)

    def find(self, *args, **kwargs):
        return self._execute("find", *args, **kwargs)

    def create_index(self, *args, **kwargs):
        return self._execute("create_index", *args, **kwargs)


# Exported Collections (Safe for both online MongoDB and offline in-memory operation)
users_collection = SafeCollection(_raw_users, InMemoryCollection("users"))
conversations_collection = SafeCollection(_raw_conversations, InMemoryCollection("conversations"))


def ensure_indexes():
    global _mongo_available
    try:
        if client:
            client.admin.command("ping")
            _mongo_available = True
            _raw_users.create_index("email", unique=True)
            _raw_conversations.create_index("user_id")
            _raw_conversations.create_index(
                [("user_id", ASCENDING), ("updated_at", DESCENDING)]
            )
            logger.info("MongoDB connected and indexes verified.")
    except Exception:
        _mongo_available = False
        logger.info("MongoDB not running locally — proceeding in memory/stateless mode.")
