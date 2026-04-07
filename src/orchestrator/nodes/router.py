"""Router Node — Zone 1, action_flow branch, Step 1.

Produces a typed AgentPlan: which domain agents to invoke and why.
Uses a fast LLM call (gpt-4o-mini) with structured output.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.infra.config import get_settings
from src.orchestrator.prompts import ROUTER_SYSTEM
from src.orchestrator.state import AgentPlan, GraphState

logger = logging.getLogger(__name__)


async def router_node(state: GraphState) -> dict:
    """Decide which domain agents should handle the user's action request."""
    user_message = state.get("user_message", "")

    llm = ChatOpenAI(
        model=get_settings().llm_model_pipeline,
        temperature=0,
        api_key=get_settings().openai_api_key,
    )
    structured_llm = llm.with_structured_output(AgentPlan)

    try:
        plan: AgentPlan = await structured_llm.ainvoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=user_message),
            ]
        )

        # Deduplicate and validate
        seen: set[str] = set()
        unique_agents = [a for a in plan.selected_agents if a not in seen and not seen.add(a)]  # type: ignore[func-returns-value]
        plan = AgentPlan(selected_agents=unique_agents or ["nutrition"], reasoning=plan.reasoning)

        logger.info(
            "Router selected agents=%s — %s",
            plan.selected_agents,
            plan.reasoning,
        )
        return {"agent_plan": plan}

    except Exception:
        # Fail safe: route to a single best-guess agent
        logger.exception("Router LLM call failed — defaulting to nutrition agent")
        return {
            "agent_plan": AgentPlan(
                selected_agents=["nutrition"],
                reasoning="Router failed — defaulting to nutrition agent.",
            )
        }
