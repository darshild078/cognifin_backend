from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Display name")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 characters)")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="Password")


class UserInfo(BaseModel):
    id: str = Field(..., description="User identifier")
    name: str = Field(..., description="User display name")
    email: str = Field(..., description="User email address")
    profile_picture: Optional[str] = Field(default="", description="Profile picture URL if available")


class AuthResponseData(BaseModel):
    token: str = Field(..., description="JWT access token")
    user: UserInfo = Field(..., description="User profile data")
