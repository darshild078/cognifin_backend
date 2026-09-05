from typing import Any, Dict, Optional
from app.core.constants import ErrorCode


class AppException(Exception):
    """Base exception for all application-controlled errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: ErrorCode = ErrorCode.BAD_REQUEST,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.context = context or {}


class BadRequestException(AppException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.BAD_REQUEST, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=400, error_code=error_code, context=context)


class UnauthorizedException(AppException):
    def __init__(self, message: str = "Authentication required.", error_code: ErrorCode = ErrorCode.UNAUTHORIZED, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=401, error_code=error_code, context=context)


class ForbiddenException(AppException):
    def __init__(self, message: str = "Access forbidden.", error_code: ErrorCode = ErrorCode.FORBIDDEN, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=403, error_code=error_code, context=context)


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found.", error_code: ErrorCode = ErrorCode.NOT_FOUND, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=404, error_code=error_code, context=context)


class ConflictException(AppException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.CONFLICT, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=409, error_code=error_code, context=context)


class ValidationException(AppException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.VALIDATION_ERROR, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=422, error_code=error_code, context=context)


class DependencyUnavailableException(AppException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=503, error_code=error_code, context=context)
