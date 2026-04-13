"""Multimodal tools — extract and log health data from images.

Enables the wellness agent to process:
  - Food photos        → log meal items with estimated macros
  - Nutrition labels   → exact macros saved as memory + meal log
  - Workout screenshots→ log workout with extracted stats
  - Medical documents  → save condition/medication facts

Mem0 Feature 5: image content blocks sent to AsyncMemory.add() so
extracted facts become searchable memories alongside text memories.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.infra.logger import setup_logger

logger = setup_logger(__name__)


def _ctx(config: RunnableConfig) -> tuple[str, str, str]:
    c = config.get("configurable", {})
    return c.get("user_id", ""), c.get("conversation_id", ""), c.get("session_id", "")


class LogFromImageInput(BaseModel):
    image_url: str = Field(
        ...,
        description=(
            "A publicly accessible https:// URL pointing to the image, "
            "or a data:image/<type>;base64,<data> string for local files. "
            "Supported formats: JPEG, PNG, WebP, GIF. Maximum size: 15 MB."
        ),
    )
    context: str = Field(
        "",
        description=(
            "User's description of what the image shows. "
            "Examples: 'This is my lunch', 'Nutrition label for oats', "
            "'My workout summary from the app', 'My prescription label'."
        ),
    )
    domain: str = Field(
        "nutrition",
        description=(
            "Health domain of the image content: "
            "'nutrition' (food photo, nutrition label, meal), "
            "'fitness' (workout screenshot, fitness tracker), "
            "'medical' (prescription, medical document, test result), "
            "'general' (any other health-relevant image)."
        ),
    )


@tool("log_from_image", args_schema=LogFromImageInput)
async def log_from_image(
    image_url: str,
    context: str,
    domain: str,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Extract health data from an image and save it as a memory.

    Processes food photos, nutrition labels, workout screenshots, or medical
    documents. Extracted facts are stored as searchable memories so the agent
    can recall them in future conversations.
    """
    user_id, _, session_id = _ctx(config)

    from src.infra.mem0_client import add_memory_from_image
    result = await add_memory_from_image(
        image_url=image_url,
        context_text=context,
        user_id=user_id,
        domain=domain,
        run_id=session_id or None,
    )

    if result.get("status") == "error":
        return result

    # Return a refresh hint so the UI knows relevant data changed
    refresh_map = {
        "nutrition": "meal-items",
        "fitness": "workouts",
        "medical": "user-profile",
        "general": "user-profile",
    }
    return {
        **result,
        "refresh": refresh_map.get(domain, "user-profile"),
        "message": (
            f"Image processed and saved to {domain} memory. "
            + (
                f"Extracted: {result['extracted_fact']}"
                if "extracted_fact" in result
                else "Facts stored for future recall."
            )
        ),
    }


