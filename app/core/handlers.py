import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.exceptions import AppException
from app.core.constants import ErrorCode
from app.core.logging import request_id_ctx_var

logger = logging.getLogger("cognifin.handlers")


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    req_id = request_id_ctx_var.get() or "-"
    logger.warning(
        f"action=handled_app_exception status={exc.status_code} error_code={exc.error_code} "
        f"message='{exc.message}' path={request.url.path}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "data": None,
            "error_code": exc.error_code.value if hasattr(exc.error_code, "value") else str(exc.error_code),
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    field = ".".join(str(loc) for loc in first_error.get("loc", []))
    msg = first_error.get("msg", "Invalid input")
    safe_message = f"Validation failed for {field}: {msg}" if field else f"Validation failed: {msg}"

    logger.warning(f"action=validation_failed path={request.url.path} message='{safe_message}'")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": safe_message,
            "data": None,
            "error_code": ErrorCode.VALIDATION_ERROR.value,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = request_id_ctx_var.get() or "-"
    logger.error(
        f"action=unhandled_exception path={request.url.path} exception='{type(exc).__name__}' "
        f"error='{str(exc)}'",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected internal error occurred. Please try again later.",
            "data": None,
            "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
        },
    )
