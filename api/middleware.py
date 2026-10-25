"""Request tracing, correlation ID, and structured latency logging middleware.
Compliance pattern: Audit trail for regulated enterprise banking environments (JPMC / Goldman / MUFG).
"""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("finsight.audit")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Assigns a UUID4 request_id to each incoming request, records execution latency,
    and returns X-Request-ID in response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Honor client-provided request ID if present, otherwise generate new UUID4
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Inject tracing header into response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        # Structured audit log
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"request_id={request_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={latency_ms} client_ip={client_host}"
        )

        return response
