"""LangGraph state for the single-agent graph.

Simplified from the multi-agent version:
- No AgentPlan / routing — single agent handles all domains.
- No agent_outputs reducer — single agent writes directly to aggregated_response.
- Keeps the same field names (aggregated_response, refresh_entities) so the
  gateway layer requires no changes.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


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
    graph_relations: list[dict[str, Any]]         # entity relationships from Neo4j graph memory
    constraint_rules: list[str]                   # auto-derived safety rules

    # ── Meal context (populated by context_hydration_node) ───────────────
    meal_items_context: list[dict[str, Any]]        # today's logged meals with IDs

    # ── Multimodal (optional — set when user attaches an image) ───────────
    image_url: str                                  # base64 data URI or https:// URL

    # ── Final response (populated by wellness_agent_node) ─────────────────
    aggregated_response: str
    refresh_entities: list[str]
