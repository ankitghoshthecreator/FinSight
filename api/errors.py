"""Structured error handling adhering to RFC 7807 Problem Details standards.
Compliance pattern: Audit-grade error response formatting for financial microservices.
"""

import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("finsight.errors")


class FinSightBaseException(Exception):
    """Base exception for all FinSight domain errors."""
    def __init__(self, message: str, error_code: str = "FINSIGHT_ERROR", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class DocumentProcessingError(FinSightBaseException):
    """Raised when document extraction or chunking encounters an unrecoverable error."""
    def __init__(self, message: str, doc_name: Optional[str] = None):
        super().__init__(
            message=f"Processing failed for document '{doc_name}': {message}" if doc_name else message,
            error_code="DOCUMENT_PROCESSING_FAILED",
            status_code=422,
        )


class StorageError(FinSightBaseException):
    """Raised when vector store persistence or serialization fails."""
    def __init__(self, message: str):
        super().__init__(message=message, error_code="STORAGE_OPERATION_FAILED", status_code=500)


def format_error_response(
    error_code: str,
    detail: str,
    status_code: int,
    request: Optional[Request] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    request_id = getattr(getattr(request, "state", None), "request_id", "unknown") if request else "unknown"
    payload = {
        "error": error_code,
        "detail": detail,
        "status_code": status_code,
        "request_id": request_id,
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def register_error_handlers(app: FastAPI):
    """Binds global exception handlers to FastAPI application."""

    @app.exception_handler(FinSightBaseException)
    async def handle_finsight_domain_exception(request: Request, exc: FinSightBaseException):
        logger.warning(f"Domain exception {exc.error_code}: {exc.message}")
        return format_error_response(
            error_code=exc.error_code,
            detail=exc.message,
            status_code=exc.status_code,
            request=request,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        logger.info(f"Request validation error on {request.url.path}: {exc.errors()}")
        return format_error_response(
            error_code="VALIDATION_ERROR",
            detail="Incoming request payload failed schema validation",
            status_code=422,
            request=request,
            extra={"validation_errors": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        return format_error_response(
            error_code="HTTP_ERROR",
            detail=str(exc.detail),
            status_code=exc.status_code,
            request=request,
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
        return format_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred while processing the financial workflow",
            status_code=500,
            request=request,
        )
