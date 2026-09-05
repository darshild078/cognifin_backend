import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import request_id_ctx_var

logger = logging.getLogger("cognifin.access")


class RequestIdAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        token = request_id_ctx_var.set(request_id)

        start_time = time.time()
        try:
            response = await call_next(request)
            process_time_ms = (time.time() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id

            logger.info(
                f"method={request.method} path={request.url.path} status={response.status_code} "
                f"duration={process_time_ms:.1f}ms client={request.client.host if request.client else '-'}"
            )
            return response
        except Exception as exc:
            process_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"method={request.method} path={request.url.path} status=500 duration={process_time_ms:.1f}ms error={exc}"
            )
            raise exc
        finally:
            request_id_ctx_var.reset(token)
