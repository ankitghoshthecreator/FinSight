"""Authentication and rate-limiting middleware for FinSight RAG API.

Two independent layers:

1. **API Key auth** (``X-API-Key`` header)
   - Enabled only when ``settings.api_key_secret`` is non-empty.
   - Public endpoints (``/``, ``/health``, ``/docs``, ``/openapi.json``) are
     always exempt so health-check infrastructure and OpenAPI UI keep working.

2. **IP-level rate limiting** (``slowapi``)
   - Default: 60 requests/minute per remote IP (configurable via
     ``settings.rate_limit_per_minute``).
   - A limit of 0 disables rate limiting entirely.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)

# Endpoints that bypass API key authentication
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates the ``X-API-Key`` header when ``settings.api_key_secret`` is set."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # If no secret is configured, auth is disabled — pass all requests through
        if not settings.api_key_secret:
            return await call_next(request)

        # Public endpoints are always exempt
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_key_secret:
            logger.warning(
                "Rejected request to %s — invalid or missing API key",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "UNAUTHORIZED",
                    "detail": "Missing or invalid X-API-Key header.",
                },
            )

        return await call_next(request)
