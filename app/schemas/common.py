from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True, description="Indicates if the operation was successful")
    message: str = Field(default="Operation completed successfully.", description="User-facing safe message")
    data: Optional[T] = Field(default=None, description="Response payload")


class ApiErrorResponse(BaseModel):
    success: bool = Field(default=False, description="Always false for error responses")
    message: str = Field(..., description="User-facing safe error message")
    data: Optional[Any] = Field(default=None, description="Always null on error")
    error_code: str = Field(..., description="Stable, machine-readable error code")


def success_response(message: str = "Operation completed successfully.", data: Any = None) -> ApiResponse[Any]:
    return ApiResponse(success=True, message=message, data=data)
