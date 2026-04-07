"""Rate limiting middleware (sliding window via Redis)."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from src.infra.logger import setup_logger
from src.infra.redis_client import check_rate_limit

logger = setup_logger(__name__)


async def require_rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 if the user has exceeded their rate limit."""
    user_id: str = getattr(request.state, "user_id", request.client.host if request.client else "unknown")
    allowed, remaining = await check_rate_limit(user_id)
    if not allowed:
        logger.warning("Rate limit exceeded for %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )
    # Attach remaining to request state for response headers
    request.state.rate_limit_remaining = remaining
