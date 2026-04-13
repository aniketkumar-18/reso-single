"""Chat route — single-agent wellness assistant endpoint.

POST /api/v1/chat
  • Runs the unified single wellness agent (context_hydration + wellness_agent ReAct loop).
  • Returns a streaming SSE response (text/event-stream) or JSON.

SSE event schema:
  data: {"type": "token",    "content": "..."}
  data: {"type": "done",     "response": "...", "refresh_entities": [...], "trace_id": "..."}
  data: {"type": "error",    "message": "..."}
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from jose import jwt as jose_jwt

from src.agent.graph import invoke_graph as invoke_single, stream_graph_events as stream_single
from src.gateway.middleware.auth import require_auth
from src.gateway.middleware.rate_limit import require_rate_limit
from src.gateway.middleware.tracing import get_current_trace_id
from src.gateway.schemas import ChatRequest, ChatResponse
from src.infra import db
from src.infra.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

# Final-output node name for single-agent workflow
_FINAL_NODE = {"single": "wellness_agent"}


# ── SSE generator ──────────────────────────────────────────────────────────────

async def _stream_graph(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
    image_url: str,
    compiled_graph,
    workflow_mode: str,
) -> AsyncGenerator[dict, None]:
    """Run the single-agent graph with real token streaming via astream_events v2."""
    stream_fn = stream_single
    final_node = _FINAL_NODE[workflow_mode]

    try:
        yield {"data": json.dumps({"type": "thinking"})}

        streamed_tokens = False
        aggregated = ""
        refresh_entities: list[str] = []

        async for kind, event in stream_fn(
            compiled_graph,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            user_message=user_message,
            image_url=image_url,
        ):
            # Real token streaming: only LLMs tagged "final_response"
            if kind == "on_chat_model_stream" and "final_response" in event.get("tags", []):
                chunk = event["data"]["chunk"]
                content = chunk.content if hasattr(chunk, "content") else ""
                tool_call_chunks = getattr(chunk, "tool_call_chunks", [])
                if content and not tool_call_chunks:
                    streamed_tokens = True
                    yield {"data": json.dumps({"type": "token", "content": content})}

            # Capture final response from the terminal node of the active workflow
            elif kind == "on_chain_end" and event.get("name") == final_node:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    aggregated = output.get("aggregated_response", aggregated)
                    refresh_entities = output.get("refresh_entities", refresh_entities)

        # Fallback: if no tokens were streamed (e.g. single-agent with tool calls only),
        # chunk the final aggregated text so the UI doesn't wait in silence.
        if not streamed_tokens and aggregated:
            chunk_size = 30
            for i in range(0, len(aggregated), chunk_size):
                yield {"data": json.dumps({"type": "token", "content": aggregated[i : i + chunk_size]})}

        trace_id = get_current_trace_id()
        yield {
            "data": json.dumps(
                {
                    "type": "done",
                    "response": aggregated,
                    "conversation_id": conversation_id,
                    "refresh_entities": refresh_entities,
                    "trace_id": trace_id,
                    "workflow_mode": workflow_mode,
                }
            )
        }

        await db.persist_message(conversation_id, "user", user_message)
        await db.persist_message(
            conversation_id,
            "assistant",
            aggregated,
            metadata={"trace_id": trace_id, "refresh_entities": refresh_entities, "workflow_mode": workflow_mode},
        )

    except Exception as exc:
        logger.exception("Graph streaming failed for user=%s mode=%s", user_id, workflow_mode)
        yield {"data": json.dumps({"type": "error", "message": str(exc)})}


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=None)
async def chat(
    request: Request,
    body: ChatRequest,
    user_id: str = Depends(require_auth),
    _rate: None = Depends(require_rate_limit),
) -> EventSourceResponse | ChatResponse:
    """
    Single-agent wellness assistant: all tools in one ReAct loop.
    """
    mode = body.workflow_mode

    compiled_graph = getattr(request.app.state, "single_graph", None)
    invoke_fn = invoke_single

    if compiled_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Single-agent graph is not initialised.",
        )

    conversation_id = body.conversation_id or str(uuid.uuid4())
    session_id = body.session_id or conversation_id

    await db.get_or_create_conversation(user_id, conversation_id)

    image_url = body.image_url or ""

    if body.stream:
        return EventSourceResponse(
            _stream_graph(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                user_message=body.message,
                image_url=image_url,
                compiled_graph=compiled_graph,
                workflow_mode=mode,
            ),
            media_type="text/event-stream",
            headers={
                "X-RateLimit-Remaining": str(
                    getattr(request.state, "rate_limit_remaining", 0)
                ),
                "Cache-Control": "no-cache",
            },
        )

    # Non-streaming fallback
    final_state = await invoke_fn(
        compiled_graph,
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_message=body.message,
        image_url=image_url,
    )
    await db.persist_message(conversation_id, "user", body.message)
    await db.persist_message(conversation_id, "assistant", final_state.get("aggregated_response", ""))

    return ChatResponse(
        response=final_state.get("aggregated_response", ""),
        conversation_id=conversation_id,
        session_id=session_id,
        refresh_entities=final_state.get("refresh_entities", []),
        trace_id=get_current_trace_id(),
        workflow_mode=mode,
    )


@router.get("/conversations")
async def list_conversations(
    request: Request,
    limit: int = 30,
    user_id: str = Depends(require_auth),
) -> list[dict]:
    """Return recent conversations for the authenticated user, newest first."""
    return await db.get_conversations_for_user(user_id, limit=limit)


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    limit: int = 100,
    user_id: str = Depends(require_auth),
) -> list[dict]:
    """Return messages for a conversation (oldest first)."""
    return await db.get_conversation_history(conversation_id, limit=limit)


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness + readiness probe."""
    checks: dict[str, str] = {}

    try:
        from src.infra.redis_client import _get_pool
        import redis.asyncio as aioredis
        r = aioredis.Redis(connection_pool=_get_pool())
        await r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    try:
        from src.infra.db import get_client
        client = await get_client()
        checks["supabase"] = "not configured" if client is None else "ok"
    except Exception as exc:
        checks["supabase"] = f"error: {exc}"

    checks["single_graph"] = "ok" if getattr(request.app.state, "single_graph", None) is not None else "not ready"

    # Feature 9 — Mem0 OSS component status + safe config summary
    mem0_config: dict = {}
    try:
        from src.infra.mem0_client import get_mem0_config_summary
        mem0_config = get_mem0_config_summary()
        checks["mem0_configured"] = "yes" if mem0_config["mem0_configured"] else "no"
        checks["mem0_proxy"] = "yes" if mem0_config["proxy_configured"] else "no"
        checks["mem0_collection"] = mem0_config["vector_store"]["collection"]
        checks["mem0_reranker_top_k"] = str(mem0_config["reranker"]["top_k"])
    except Exception as exc:
        checks["mem0_configured"] = f"error: {exc}"

    overall = "ok" if all(
        not v.startswith("error") and v != "not ready"
        for v in checks.values()
    ) else "degraded"
    return {"status": overall, "checks": checks, "mem0_config": mem0_config}


@router.get("/dev-token", include_in_schema=False)
async def dev_token(user_id: str | None = None) -> dict:
    """Development only — returns a signed JWT for any user_id. Blocked in production.

    Also ensures the user exists in auth.users so FK constraints on all domain
    tables are satisfied. Safe to call multiple times — create_user is idempotent.
    """
    from src.infra.config import get_settings
    from src.infra.db import get_client
    settings = get_settings()
    if settings.is_production:
        raise HTTPException(status_code=404)
    resolved_user_id = user_id or "990e931b-5246-417a-8f1a-5b92a2efb01f"

    # Register the user in auth.users so all FK constraints pass.
    # Uses admin API (service role key) — bypasses email confirmation.
    client = await get_client()
    if client is not None:
        try:
            await client.auth.admin.create_user({
                "user_id": resolved_user_id,
                "email": f"dev_{resolved_user_id}@test.local",
                "email_confirm": True,
            })
        except Exception:
            pass  # User already exists — that's fine

    payload = {
        "sub": resolved_user_id,
        "role": "authenticated",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=1),
    }
    token = jose_jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return {"token": token, "user_id": resolved_user_id}
