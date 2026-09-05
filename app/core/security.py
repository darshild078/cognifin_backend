from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import jwt, ExpiredSignatureError, JWTError

from app.core.config import settings
from app.core.exceptions import UnauthorizedException


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(user_id: str, email: str, name: str = "") -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise UnauthorizedException("Invalid token: missing subject.")
        return payload
    except ExpiredSignatureError:
        raise UnauthorizedException("Token has expired. Please log in again.")
    except JWTError:
        raise UnauthorizedException("Invalid authentication token.")
