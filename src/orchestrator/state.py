"""LangGraph state definitions for the Zone 1 orchestration graph.

Key design decisions:
- `agent_outputs` uses `operator.add` reducer so parallel agents can write
  concurrently without overwriting each other.
- `messages` is intentionally NOT in GraphState — each domain agent manages
  its own internal message list; only the final text surfaces via agent_outputs.
- All fields have sensible defaults so nodes only need to return their delta.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


DomainName = Literal["nutrition", "fitness", "medical", "general"]


# ── Router output ──────────────────────────────────────────────────────────────

class AgentPlan(BaseModel):
    """Structured output from the router node."""

    selected_agents: list[DomainName] = Field(
        ...,
        description="Which domain agents to invoke (1–3). Always at least one.",
    )
    reasoning: str = Field(
        ...,
        description="One-sentence explanation of why these agents were selected.",
    )


# ── Individual agent result ────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Result produced by a single domain agent."""

    domain: DomainName
    content: str
    refresh_entities: list[str] = Field(
        default_factory=list,
        description="Client-side entity keys that need refreshing (e.g. 'meal-items').",
    )
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="Names of tools that were called during the ReAct loop.",
    )


# ── Custom reducer ─────────────────────────────────────────────────────────────

def _add_or_reset_outputs(existing: list, update: list) -> list:
    """Append agent outputs within a turn; an empty-list update resets the slate.

    _join_node sends [] at the start of every turn to clear stale outputs that
    the Redis checkpointer would otherwise carry over from the previous turn.
    """
    if not update:
        return []
    return existing + update


# ── Main graph state ───────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    # ── Identity ─────────────────────────────────────────────────────────────
    user_id: str
    session_id: str
    conversation_id: str
    user_message: str

    # ── Context (populated by context_hydration_node) ─────────────────────
    user_profile: dict[str, Any]
    conversation_history: list[dict[str, Any]]   # [{role, content, created_at}]
    memory_context: list[str]                     # extracted fact strings
    constraint_rules: list[str]                   # auto-derived safety rules

    # ── Routing (populated by classify_route_node) ───────────────────────
    agent_plan: AgentPlan | None

    # ── Clarification (populated by classify_route_node) ─────────────────
    needs_clarification: bool          # True → ask user before running agents
    clarification_question: str        # The single question to ask

    # ── Agent results — reducer accumulates across parallel writes ─────────
    agent_outputs: Annotated[list[AgentOutput], _add_or_reset_outputs]

    # ── Final response (populated by aggregator_node) ─────────────────────
    aggregated_response: str
    refresh_entities: list[str]
