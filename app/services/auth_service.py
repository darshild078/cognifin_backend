import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from app.core.database import users_collection
from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import ConflictException, UnauthorizedException, BadRequestException
from app.core.constants import ErrorCode
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponseData, UserInfo

logger = logging.getLogger("cognifin.service.auth")


def register_user(payload: RegisterRequest) -> AuthResponseData:
    existing = users_collection.find_one({"email": payload.email.lower()})
    if existing:
        raise ConflictException(
            message="An account with this email address already exists.",
            error_code=ErrorCode.CONFLICT,
        )

    now = datetime.now(timezone.utc)
    hashed_pw = hash_password(payload.password)
    user_doc = {
        "name": payload.name.strip(),
        "email": payload.email.lower().strip(),
        "password_hash": hashed_pw,
        "auth_provider": "local",
        "created_at": now,
        "last_login": now,
    }
    result = users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)

    token = create_access_token(user_id=user_id, email=user_doc["email"], name=user_doc["name"])
    logger.info(f"action=register_user user_id={user_id} email={user_doc['email']} status=success")

    return AuthResponseData(
        token=token,
        user=UserInfo(id=user_id, name=user_doc["name"], email=user_doc["email"], profile_picture=""),
    )


def login_user(payload: LoginRequest) -> AuthResponseData:
    user = users_collection.find_one({"email": payload.email.lower().strip()})
    if not user or not user.get("password_hash"):
        raise UnauthorizedException(
            message="Invalid email or password.",
            error_code=ErrorCode.AUTH_FAILED,
        )

    if not verify_password(payload.password, user["password_hash"]):
        raise UnauthorizedException(
            message="Invalid email or password.",
            error_code=ErrorCode.AUTH_FAILED,
        )

    user_id = str(user["_id"])
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )

    token = create_access_token(user_id=user_id, email=user["email"], name=user.get("name", ""))
    logger.info(f"action=login_user user_id={user_id} email={user['email']} status=success")

    return AuthResponseData(
        token=token,
        user=UserInfo(
            id=user_id,
            name=user.get("name", ""),
            email=user["email"],
            profile_picture=user.get("profile_picture", ""),
        ),
    )


def upsert_google_user(email: str, name: str, picture: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    existing = users_collection.find_one({"email": email.lower()})

    if existing is None:
        result = users_collection.insert_one({
            "name": name,
            "email": email.lower(),
            "password_hash": None,
            "profile_picture": picture,
            "auth_provider": "google",
            "created_at": now,
            "last_login": now,
        })
        user_id = str(result.inserted_id)
        logger.info(f"action=upsert_google_user user_id={user_id} email={email} status=created")
        return {"_id": user_id, "name": name, "email": email, "profile_picture": picture}
    else:
        update_fields: Dict[str, Any] = {"last_login": now, "profile_picture": picture}
        if not existing.get("name"):
            update_fields["name"] = name

        users_collection.update_one({"_id": existing["_id"]}, {"$set": update_fields})
        user_id = str(existing["_id"])
        logger.info(f"action=upsert_google_user user_id={user_id} email={email} status=updated")
        return {
            "_id": user_id,
            "name": existing.get("name") or name,
            "email": existing["email"],
            "profile_picture": picture,
        }
