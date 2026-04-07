"""Mode Classifier Node — Zone 1, Step 2.

Uses a fast LLM call (gpt-4o-mini) with structured output to split traffic:
  - free_flow  → LLM advice/question answering, no tools
  - action_flow → router → domain agents with tools
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.infra.config import get_settings
from src.orchestrator.prompts import MODE_CLASSIFIER_SYSTEM
from src.orchestrator.state import FlowMode, GraphState

logger = logging.getLogger(__name__)


class _ModeClassification(BaseModel):
    mode: FlowMode = Field(..., description="free_flow or action_flow")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="One sentence explaining the classification.")


async def mode_classifier_node(state: GraphState) -> dict:
    """Classify the user's message as free_flow or action_flow."""
    user_message = state.get("user_message", "")

    llm = ChatOpenAI(
        model=get_settings().llm_model_pipeline,
        temperature=0,
        api_key=get_settings().openai_api_key,
    )
    structured_llm = llm.with_structured_output(_ModeClassification)

    try:
        result: _ModeClassification = await structured_llm.ainvoke(
            [
                SystemMessage(content=MODE_CLASSIFIER_SYSTEM),
                HumanMessage(content=user_message),
            ]
        )
        logger.info(
            "Mode classified: %s (confidence=%.2f) — %s",
            result.mode,
            result.confidence,
            result.reasoning,
        )
        return {"flow_mode": result.mode}
    except Exception:
        # Fail open: default to free_flow (cheaper, never destructive)
        logger.exception("Mode classification failed — defaulting to free_flow")
        return {"flow_mode": "free_flow"}
