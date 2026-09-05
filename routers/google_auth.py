"""
FinSight AI - Google OAuth Router
===================================
Handles Google OAuth 2.0 login flow with two endpoints:

  GET /auth/login    → Redirects to Google consent screen
  GET /auth/callback → Handles Google redirect, upserts user, issues JWT

Author: FinSight AI Team
"""

import os
import json
import base64
import logging

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request
from starlette.responses import RedirectResponse

from auth import oauth, create_access_token
from db import upsert_google_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback"
)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/login", summary="Initiate Google OAuth Login")
async def google_login(request: Request):
    """
    Redirects the user to Google's OAuth consent screen.

    The frontend should navigate to this URL (full page redirect, not fetch).
    After the user authorizes, Google redirects back to /auth/callback.
    """
    # Verify Google OAuth is configured
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if not client_id or client_id.startswith("your_"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env",
        )

    return await oauth.google.authorize_redirect(request, GOOGLE_REDIRECT_URI)


@router.get("/callback", summary="Google OAuth Callback")
async def google_callback(request: Request):
    """
    Handles the redirect from Google after user authorization.

    Flow:
      1. Exchange authorization code for access token + id_token
      2. Extract user info (email, name, picture)
      3. Upsert user in MongoDB
      4. Generate JWT
      5. Redirect to frontend with token + user info
    """
    try:
        # Step 1: Exchange code for tokens
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error("Google OAuth token exchange failed: %s", e)
        # Redirect to frontend login with error
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_auth_failed",
            status_code=302,
        )

    # Step 2: Extract user info from the id_token
    user_info = token.get("userinfo")

    if user_info is None:
        logger.error("Google OAuth: No userinfo in token response")
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=no_user_info",
            status_code=302,
        )

    email = user_info.get("email")
    name = user_info.get("name", "")
    picture = user_info.get("picture", "")

    if not email:
        logger.error("Google OAuth: No email in userinfo")
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=no_email",
            status_code=302,
        )

    # Step 3: Upsert user in MongoDB
    try:
        user = upsert_google_user(email=email, name=name, picture=picture)
    except Exception as e:
        logger.error("MongoDB upsert failed during Google OAuth: %s", e)
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=db_error",
            status_code=302,
        )

    # Step 4: Generate JWT (reuses existing JWT infrastructure)
    jwt_token = create_access_token(
        user_id=user["_id"],
        email=user["email"],
        name=user["name"],
    )

    # Step 5: Redirect to frontend callback page with token + user info
    # Encode user data as base64 JSON for safe URL transport
    user_payload = {
        "id": user["_id"],
        "name": user["name"],
        "email": user["email"],
        "profile_picture": user.get("profile_picture", ""),
    }
    user_b64 = base64.urlsafe_b64encode(
        json.dumps(user_payload).encode()
    ).decode()

    redirect_url = (
        f"{FRONTEND_URL}/auth/callback"
        f"?token={jwt_token}"
        f"&user={user_b64}"
    )

    return RedirectResponse(url=redirect_url, status_code=302)
