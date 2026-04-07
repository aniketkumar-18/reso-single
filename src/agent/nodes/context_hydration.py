"""Context Hydration Node — fetches profile, history, memories in parallel.

Identical logic to the multi-agent version; only the state import differs.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.infra import db
from src.infra.logger import setup_logger
from src.infra.redis_client import cache_get, cache_set
from src.agent.state import GraphState

logger = setup_logger(__name__)


# ── Constraint rule derivation ─────────────────────────────────────────────────

_CONDITION_RULES: dict[str, list[str]] = {
    "type 2 diabetes": [
        "No recommendations involving refined sugar or high-GI foods.",
        "Keep carbohydrate suggestions moderate and specify slow-digesting options.",
    ],
    "diabetes": [
        "No recommendations involving refined sugar or high-GI foods.",
    ],
    "hypertension": [
        "Avoid high-sodium food suggestions.",
        "Recommend low-sodium alternatives where relevant.",
    ],
    "celiac": [
        "All food suggestions must be strictly gluten-free.",
    ],
    "lactose intolerance": [
        "Avoid dairy products unless lactose-free alternatives are specified.",
    ],
}

_MEDICATION_RULES: dict[str, list[str]] = {
    "metformin": [
        "Do not recommend alcohol — it can cause lactic acidosis with metformin.",
    ],
    "warfarin": [
        "Avoid recommending high-Vitamin K foods (spinach, kale, broccoli) without caveats.",
    ],
    "ssri": [
        "Avoid recommending St. John's Wort supplements.",
    ],
}

_ALLERGY_RULE = "ALLERGY — never suggest {allergen} or foods containing it."


def derive_constraint_rules(profile: dict[str, Any]) -> list[str]:
    """Convert profile facts into a deterministic set of safety constraint strings."""
    rules: list[str] = []

    conditions: list[str] = profile.get("conditions", []) or []
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        lc = condition.lower()
        for key, condition_rules in _CONDITION_RULES.items():
            if key in lc:
                rules.extend(condition_rules)
                break

    medications: list[str] = profile.get("medications", []) or []
    if isinstance(medications, str):
        medications = [medications]
    for med in medications:
        lc = med.lower()
        for key, med_rules in _MEDICATION_RULES.items():
            if re.search(rf"\b{re.escape(key)}\b", lc):
                rules.extend(med_rules)
                break

    allergies: list[str] = profile.get("allergies", []) or []
    if isinstance(allergies, str):
        allergies = [s.strip() for s in allergies.split(",") if s.strip()]
    for allergen in allergies:
        rules.append(_ALLERGY_RULE.format(allergen=allergen))

    return list(dict.fromkeys(rules))  # deduplicate, preserve order


# ── Node ───────────────────────────────────────────────────────────────────────

async def context_hydration_node(state: GraphState) -> dict:
    """Populate user_profile, conversation_history, memory_context, constraint_rules."""
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "") or None
    conversation_id = state.get("conversation_id", "")

    profile: dict[str, Any] = {}
    if user_id:
        cached = await cache_get(f"profile:{user_id}")
        if cached:
            profile = cached
            logger.debug("Profile cache hit for %s", user_id)

    async def _fetch_profile() -> dict[str, Any]:
        p = await db.get_user_profile(user_id)
        if p:
            await cache_set(f"profile:{user_id}", p, ttl_seconds=300)
        return p

    tasks: list = []
    profile_future = None

    if not profile:
        profile_future = asyncio.create_task(_fetch_profile())
        tasks.append(profile_future)

    from datetime import date as date_cls
    today = date_cls.today().isoformat()

    user_message = state.get("user_message", "")

    from src.infra.mem0_client import search_memories_with_graph, infer_domains
    history_future = asyncio.create_task(db.get_conversation_history(conversation_id))

    # Infer relevant domains from user message for metadata-filtered retrieval (Feature 2).
    # Doing inference here (not inside mem0_client) so we can log it at the node level.
    memory_query = user_message or "recent context"
    active_domains = infer_domains(memory_query)
    logger.debug("Memory domains inferred — user=%s domains=%s", user_id, active_domains)

    memory_future = asyncio.create_task(
        search_memories_with_graph(
            memory_query, user_id=user_id, domains=active_domains, run_id=session_id
        )
    )
    meal_context_future = asyncio.create_task(db.get_meal_items(user_id, date=today, limit=30))
    tasks.extend([history_future, memory_future, meal_context_future])

    await asyncio.gather(*tasks, return_exceptions=True)

    if profile_future is not None:
        exc = profile_future.exception()
        if exc:
            logger.warning("Profile fetch failed: %s", exc)
        else:
            profile = profile_future.result()

    history_exc = history_future.exception()
    history: list[dict] = [] if history_exc else history_future.result()

    memory_exc = memory_future.exception()
    memory_result: dict = {} if memory_exc else memory_future.result()
    memories: list[str] = memory_result.get("facts", []) if memory_result else []
    graph_relations: list[dict] = memory_result.get("relations", []) if memory_result else []

    meal_ctx_exc = meal_context_future.exception()
    meal_items_context: list[dict] = [] if meal_ctx_exc else meal_context_future.result()

    constraint_rules = derive_constraint_rules(profile)

    logger.info(
        "Context hydrated — user=%s profile_keys=%d history=%d memories=%d relations=%d constraints=%d meals=%d",
        user_id,
        len(profile),
        len(history),
        len(memories),
        len(graph_relations),
        len(constraint_rules),
        len(meal_items_context),
    )

    return {
        "user_profile": profile,
        "conversation_history": history,
        "memory_context": memories,
        "graph_relations": graph_relations,
        "constraint_rules": constraint_rules,
        "meal_items_context": meal_items_context,
    }
