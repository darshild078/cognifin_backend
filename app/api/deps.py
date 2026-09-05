import os
from typing import Dict, Any
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from authlib.integrations.starlette_client import OAuth

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.exceptions import UnauthorizedException

bearer_scheme = HTTPBearer(auto_error=True)

# Google OAuth instance
oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Dict[str, Any]:
    payload = decode_access_token(credentials.credentials)
    return {
        "user_id": payload["sub"],
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
    }
