"""Global error handling middleware for FastAPI."""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error class."""

    def __init__(
        self,
        error: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ):
        self.error = error
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None):
        super().__init__(
            error="not_found",
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class UnauthorizedError(APIError):
    """Authentication required error."""

    def __init__(
        self, message: str = "Authentication required", details: dict[str, Any] | None = None
    ):
        super().__init__(
            error="unauthorized",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class ForbiddenError(APIError):
    """Access forbidden error."""

    def __init__(self, message: str = "Access forbidden", details: dict[str, Any] | None = None):
        super().__init__(
            error="forbidden",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ConflictError(APIError):
    """Resource conflict error."""

    def __init__(self, message: str = "Resource conflict", details: dict[str, Any] | None = None):
        super().__init__(
            error="conflict",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""

    def __init__(
        self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None
    ):
        super().__init__(
            error="rate_limit_exceeded",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


def setup_error_handlers(app: FastAPI) -> None:
    """Register error handlers with the FastAPI app."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handle custom API errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        logger.exception("Unexpected error occurred: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
            },
        )
