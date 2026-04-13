"""Tool registry — single source of truth for all agent tools.

Import ``ALL_TOOLS`` to wire up ``create_react_agent()``.
Individual tools are re-exported here so callers never need to reach into
domain sub-modules directly.

Adding a new tool:
  1. Implement it in the appropriate domain module (profile / nutrition /
     fitness / medical / multimodal).
  2. Import it here and append it to ``ALL_TOOLS``.
  That's all — wellness_agent.py picks it up automatically.
"""

from __future__ import annotations

from src.tools.fitness import (
    calculate_macro_targets,
    log_body_metrics,
    log_workout,
    update_fitness_goals,
)
from src.tools.medical import (
    log_medical_condition,
    log_medication,
    update_medical_condition,
    update_medication,
)
from src.tools.multimodal import log_from_image
from src.tools.nutrition import (
    delete_meal_item,
    edit_meal_item,
    get_meal_items,
    log_meal,
)
from src.tools.profile import save_memory_fact, update_user_profile

# Ordered by domain for readability; order does not affect agent behaviour.
ALL_TOOLS = [
    # ── Profile & memory ──────────────────────────────────────────────────────
    update_user_profile,
    save_memory_fact,
    # ── Nutrition ─────────────────────────────────────────────────────────────
    get_meal_items,
    log_meal,
    edit_meal_item,
    delete_meal_item,
    # ── Fitness ───────────────────────────────────────────────────────────────
    log_workout,
    log_body_metrics,
    update_fitness_goals,
    calculate_macro_targets,
    # ── Medical ───────────────────────────────────────────────────────────────
    log_medical_condition,
    update_medical_condition,
    log_medication,
    update_medication,
    # ── Multimodal ────────────────────────────────────────────────────────────
    log_from_image,
]

__all__ = [
    "ALL_TOOLS",
    # profile
    "update_user_profile",
    "save_memory_fact",
    # nutrition
    "get_meal_items",
    "log_meal",
    "edit_meal_item",
    "delete_meal_item",
    # fitness
    "log_workout",
    "log_body_metrics",
    "update_fitness_goals",
    "calculate_macro_targets",
    # medical
    "log_medical_condition",
    "update_medical_condition",
    "log_medication",
    "update_medication",
    # multimodal
    "log_from_image",
]
