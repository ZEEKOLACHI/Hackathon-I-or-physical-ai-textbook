"""Rate limiting middleware for API endpoints."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Max requests in a short burst


@dataclass
class ClientRateInfo:
    """Track rate limiting info for a client."""

    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)


class RateLimiter:
    """In-memory rate limiter."""

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self.clients: dict[str, ClientRateInfo] = defaultdict(ClientRateInfo)

    def _clean_old_requests(self, requests: list, window_seconds: int) -> list:
        """Remove requests outside the time window."""
        cutoff = time.time() - window_seconds
        return [t for t in requests if t > cutoff]

    def is_allowed(self, client_id: str) -> tuple[bool, str | None]:
        """
        Check if a request is allowed for the given client.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        info = self.clients[client_id]

        # Clean old requests
        info.minute_requests = self._clean_old_requests(info.minute_requests, 60)
        info.hour_requests = self._clean_old_requests(info.hour_requests, 3600)

        # Check minute limit
        if len(info.minute_requests) >= self.config.requests_per_minute:
            return False, "Rate limit exceeded. Please wait a minute before trying again."

        # Check hour limit
        if len(info.hour_requests) >= self.config.requests_per_hour:
            return False, "Hourly rate limit exceeded. Please try again later."

        # Check burst limit (requests in last 5 seconds)
        recent = [t for t in info.minute_requests if t > now - 5]
        if len(recent) >= self.config.burst_limit:
            return False, "Too many requests. Please slow down."

        # Record this request
        info.minute_requests.append(now)
        info.hour_requests.append(now)

        return True, None

    def get_remaining(self, client_id: str) -> dict:
        """Get remaining request counts for a client."""
        info = self.clients[client_id]

        # Clean old requests
        info.minute_requests = self._clean_old_requests(info.minute_requests, 60)
        info.hour_requests = self._clean_old_requests(info.hour_requests, 3600)

        return {
            "minute_remaining": max(0, self.config.requests_per_minute - len(info.minute_requests)),
            "hour_remaining": max(0, self.config.requests_per_hour - len(info.hour_requests)),
        }


# Global rate limiter instances for different endpoint types
default_limiter = RateLimiter(RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_limit=10,
))

# Stricter limits for AI-powered endpoints (LLM calls are expensive)
ai_limiter = RateLimiter(RateLimitConfig(
    requests_per_minute=10,
    requests_per_hour=100,
    burst_limit=3,
))

# Auth endpoints (prevent brute force)
auth_limiter = RateLimiter(RateLimitConfig(
    requests_per_minute=10,
    requests_per_hour=50,
    burst_limit=5,
))


def get_client_id(request: Request) -> str:
    """Extract client identifier from request."""
    # Try to get real IP from proxy headers
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to apply rate limiting to all requests."""

    def __init__(self, app, limiter: RateLimiter | None = None):
        super().__init__(app)
        self.limiter = limiter or default_limiter

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        client_id = get_client_id(request)

        # Determine which limiter to use based on path
        path = request.url.path
        if "/chat/" in path or "/content/personalize" in path or "/content/translate" in path:
            limiter = ai_limiter
        elif "/auth/" in path:
            limiter = auth_limiter
        else:
            limiter = self.limiter

        is_allowed, error_message = limiter.is_allowed(client_id)

        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limited",
                    "message": error_message,
                },
                headers={
                    "Retry-After": "60",
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        remaining = limiter.get_remaining(client_id)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute_remaining"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour_remaining"])

        return response


def rate_limit(limiter: RateLimiter = default_limiter):
    """
    Dependency for rate limiting specific endpoints.

    Usage:
        @router.post("/endpoint")
        async def endpoint(request: Request, _: None = Depends(rate_limit(ai_limiter))):
            ...
    """
    async def dependency(request: Request):
        client_id = get_client_id(request)
        is_allowed, error_message = limiter.is_allowed(client_id)

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limited",
                    "message": error_message,
                },
            )

    return dependency
