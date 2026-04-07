"""Context Hydration Node — Zone 1, Step 1.

Fetches in parallel:
  - User profile (Supabase user_profiles, Redis-cached for 5 min)
  - Conversation history (last N messages)
  - Recent memory facts (last 7 days)

Then derives constraint rules from the profile.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.infra import db
from src.infra.logger import setup_logger
from src.infra.redis_client import cache_get, cache_set
from src.orchestrator.state import GraphState

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

    # Conditions
    conditions: list[str] = profile.get("conditions", []) or []
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        lc = condition.lower()
        for key, condition_rules in _CONDITION_RULES.items():
            if key in lc:
                rules.extend(condition_rules)
                break

    # Medications
    medications: list[str] = profile.get("medications", []) or []
    if isinstance(medications, str):
        medications = [medications]
    for med in medications:
        lc = med.lower()
        for key, med_rules in _MEDICATION_RULES.items():
            if re.search(rf"\b{re.escape(key)}\b", lc):
                rules.extend(med_rules)
                break

    # Allergies
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
    conversation_id = state.get("conversation_id", "")

    # Try Redis cache for profile first (5-min TTL, invalidated on write)
    profile: dict[str, Any] = {}
    if user_id:
        cached = await cache_get(f"profile:{user_id}")
        if cached:
            profile = cached
            logger.debug("Profile cache hit for %s", user_id)

    # Parallel fetch: profile (if not cached) + history + memories
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

    user_message = state.get("user_message", "")

    from src.infra.mem0_client import search_memories
    history_future = asyncio.create_task(db.get_conversation_history(conversation_id))
    memory_future = asyncio.create_task(
        search_memories(user_message, user_id=user_id)
        if user_message else db.get_recent_memories(user_id)
    )
    tasks.extend([history_future, memory_future])

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
    memories: list[str] = [] if memory_exc else memory_future.result()

    constraint_rules = derive_constraint_rules(profile)

    logger.info(
        "Context hydrated — user=%s profile_keys=%d history=%d memories=%d constraints=%d",
        user_id,
        len(profile),
        len(history),
        len(memories),
        len(constraint_rules),
    )

    return {
        "user_profile": profile,
        "conversation_history": history,
        "memory_context": memories,
        "constraint_rules": constraint_rules,
    }
