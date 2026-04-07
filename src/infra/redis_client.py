"""Redis async client — singleton for rate limiting, idempotency keys, and TTL cache."""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis

from src.infra.config import get_settings
from src.infra.logger import setup_logger

logger = setup_logger(__name__)

_pool: aioredis.ConnectionPool | None = None


def _get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            get_settings().redis_url,
            max_connections=20,
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis:
    """Return a Redis client backed by the shared connection pool."""
    return aioredis.Redis(connection_pool=_get_pool())


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


# ── Rate limiting (sliding window counter) ────────────────────────────────────

async def check_rate_limit(user_id: str) -> tuple[bool, int]:
    """
    Sliding-window rate limit check.

    Returns (allowed: bool, remaining: int).
    Uses a per-user key with Redis INCR + EXPIRE.
    """
    settings = get_settings()
    redis = get_redis()
    key = f"rl:{user_id}"
    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = await pipe.execute()
        if ttl == -1:
            await redis.expire(key, settings.rate_limit_window_seconds)
        if count > settings.rate_limit_requests:
            return False, 0
        remaining = settings.rate_limit_requests - count
        return True, remaining
    except Exception:
        logger.exception("Redis rate limit check failed — allowing request")
        return True, settings.rate_limit_requests


# ── Idempotency keys ──────────────────────────────────────────────────────────

async def set_idempotency_key(key: str, value: str, ttl_seconds: int = 86_400) -> bool:
    """Set an idempotency key. Returns True if newly set, False if already existed."""
    redis = get_redis()
    try:
        result = await redis.set(f"idem:{key}", value, ex=ttl_seconds, nx=True)
        return result is True
    except Exception:
        logger.exception("Failed to set idempotency key %s", key)
        return True  # fail open


async def get_idempotency_key(key: str) -> str | None:
    redis = get_redis()
    try:
        return await redis.get(f"idem:{key}")
    except Exception:
        logger.exception("Failed to get idempotency key %s", key)
        return None


# ── Profile TTL cache ─────────────────────────────────────────────────────────

async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Cache a JSON-serialisable value."""
    import json
    redis = get_redis()
    try:
        await redis.set(f"cache:{key}", json.dumps(value), ex=ttl_seconds)
    except Exception:
        logger.exception("Cache set failed for key %s", key)


async def cache_get(key: str) -> Any | None:
    """Retrieve a cached value or None if missing / expired."""
    import json
    redis = get_redis()
    try:
        raw = await redis.get(f"cache:{key}")
        return json.loads(raw) if raw else None
    except Exception:
        logger.exception("Cache get failed for key %s", key)
        return None


async def cache_delete(key: str) -> None:
    redis = get_redis()
    try:
        await redis.delete(f"cache:{key}")
    except Exception:
        logger.exception("Cache delete failed for key %s", key)
