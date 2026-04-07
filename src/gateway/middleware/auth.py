"""JWT authentication middleware.

Validates Supabase-issued JWTs on every protected request.
Extracts `sub` (user_id) and injects it into request.state.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

from src.infra.config import get_settings
from src.infra.logger import setup_logger

logger = setup_logger(__name__)


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def decode_jwt(token: str) -> dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},  # Supabase doesn't always set aud
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_auth(request: Request) -> str:
    """FastAPI dependency — returns user_id or raises 401."""
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_jwt(token)
    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim.",
        )
    request.state.user_id = user_id
    return user_id
