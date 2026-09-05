"""
FinSight AI - MongoDB Connection
=================================
Provides database connection and collection handles.

Collections:
  - users:         User accounts (name, email, password_hash)
  - conversations: Per-user chat history with messages

Author: FinSight AI Team
"""

import os
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "finsightai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

users_collection = db["users"]
conversations_collection = db["conversations"]

# ---------------------------------------------------------------------------
# Indexes (idempotent — safe to call on every startup)
# ---------------------------------------------------------------------------

def ensure_indexes():
    """Create required indexes. Call once during app startup."""
    # Unique email for users
    users_collection.create_index("email", unique=True)

    # Fast user-scoped conversation lookups
    conversations_collection.create_index("user_id")
    conversations_collection.create_index(
        [("user_id", ASCENDING), ("updated_at", DESCENDING)]
    )

    print("📦 MongoDB indexes ensured")


# ---------------------------------------------------------------------------
# Google OAuth User Upsert
# ---------------------------------------------------------------------------

def upsert_google_user(email: str, name: str, picture: str) -> dict:
    """
    Find or create a user document for Google OAuth login.

    - If the user does NOT exist: creates a new document with auth_provider='google'.
    - If the user already exists: updates last_login, name, and profile_picture.
      Does NOT overwrite password_hash or auth_provider (preserves local accounts).

    Returns a dict with: _id (str), name, email, profile_picture.
    """
    now = datetime.now(timezone.utc)
    existing = users_collection.find_one({"email": email})

    if existing is None:
        result = users_collection.insert_one({
            "name": name,
            "email": email,
            "password_hash": None,
            "profile_picture": picture,
            "auth_provider": "google",
            "created_at": now,
            "last_login": now,
        })
        return {
            "_id": str(result.inserted_id),
            "name": name,
            "email": email,
            "profile_picture": picture,
        }
    else:
        update_fields = {
            "last_login": now,
            "profile_picture": picture,
        }
        # Keep name fresh only if user didn't set one manually
        if not existing.get("name"):
            update_fields["name"] = name

        users_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": update_fields},
        )
        return {
            "_id": str(existing["_id"]),
            "name": existing.get("name") or name,
            "email": existing["email"],
            "profile_picture": picture,
        }

