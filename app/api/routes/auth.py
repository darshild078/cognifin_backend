import json
import base64
import logging
from fastapi import APIRouter, Request, status
from starlette.responses import RedirectResponse

from app.core.config import settings
from app.core.exceptions import BadRequestException, DependencyUnavailableException
from app.core.security import create_access_token
from app.api.deps import oauth
from app.schemas.common import ApiResponse, success_response
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponseData
from app.services.auth_service import register_user, login_user, upsert_google_user

logger = logging.getLogger("cognifin.api.auth")
router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=ApiResponse[AuthResponseData], status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    result = register_user(payload)
    return success_response(message="Account registered successfully.", data=result)


@router.post("/login", response_model=ApiResponse[AuthResponseData], status_code=status.HTTP_200_OK)
def login(payload: LoginRequest):
    result = login_user(payload)
    return success_response(message="Logged in successfully.", data=result)


@router.get("/auth/login", summary="Initiate Google OAuth")
async def google_login(request: Request):
    if not settings.GOOGLE_CLIENT_ID or settings.GOOGLE_CLIENT_ID.startswith("your_"):
        raise DependencyUnavailableException(message="Google OAuth credentials are not configured.")
    return await oauth.google.authorize_redirect(request, settings.GOOGLE_REDIRECT_URI)


@router.get("/auth/callback", summary="Google OAuth Callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error(f"Google OAuth token exchange failed: {e}")
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error=google_auth_failed", status_code=302)

    user_info = token.get("userinfo")
    if not user_info or not user_info.get("email"):
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error=no_email", status_code=302)

    user = upsert_google_user(
        email=user_info["email"],
        name=user_info.get("name", ""),
        picture=user_info.get("picture", ""),
    )

    jwt_token = create_access_token(
        user_id=user["_id"],
        email=user["email"],
        name=user["name"],
    )

    user_payload = {
        "id": user["_id"],
        "name": user["name"],
        "email": user["email"],
        "profile_picture": user.get("profile_picture", ""),
    }
    user_b64 = base64.urlsafe_b64encode(json.dumps(user_payload).encode()).decode()
    redirect_url = f"{settings.FRONTEND_URL}/auth/callback?token={jwt_token}&user={user_b64}"
    return RedirectResponse(url=redirect_url, status_code=302)
