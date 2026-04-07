"""Classify + Route Node — Zone 1.

Single LLM call that:
  1. Selects which domain agents to invoke.
  2. Detects whether a critical required field is missing and, if so, produces
     a single clarification question — routing to aggregator instead of agents.

No extra LLM calls are added; clarification detection is baked into the same
structured output as agent selection.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.orchestrator.prompts import ROUTER_SYSTEM
from src.orchestrator.state import AgentPlan, DomainName, GraphState

logger = setup_logger(__name__)


class _ClassifyRouteOutput(BaseModel):
    """Structured output: agent selection + optional clarification signal."""

    selected_agents: list[DomainName] = Field(
        ..., description="Which domain agents to invoke (1–3)."
    )
    reasoning: str = Field(
        ..., description="One-sentence explanation of agent selection."
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when a critical required field is missing from the user's message.",
    )
    clarification_question: str = Field(
        default="",
        description="The single question to ask the user when needs_clarification is True.",
    )


async def classify_route_node(state: GraphState) -> dict:
    """Select agents and check whether clarification is needed before acting."""
    user_message = state.get("user_message", "")

    llm = ChatOpenAI(
        model=get_settings().llm_model_pipeline,
        temperature=0,
        api_key=get_settings().openai_api_key,
    )
    structured_llm = llm.with_structured_output(_ClassifyRouteOutput)

    try:
        result: _ClassifyRouteOutput = await structured_llm.ainvoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=user_message),
            ]
        )

        # Deduplicate agents while preserving order
        seen: set[str] = set()
        unique_agents: list[DomainName] = [
            a for a in result.selected_agents if a not in seen and not seen.add(a)  # type: ignore[func-returns-value]
        ]
        agents: list[DomainName] = unique_agents or ["nutrition"]  # type: ignore[list-item]
        agent_plan = AgentPlan(selected_agents=agents, reasoning=result.reasoning)

        if result.needs_clarification:
            logger.info(
                "Classify+route: clarification needed — agents=%s question=%r",
                agents,
                result.clarification_question,
            )
        else:
            logger.info("Classify+route: agents=%s — %s", agents, result.reasoning)

        return {
            "agent_plan": agent_plan,
            "needs_clarification": result.needs_clarification,
            "clarification_question": result.clarification_question,
        }

    except Exception:
        logger.exception("Classify+route failed — defaulting to nutrition agent")
        return {
            "agent_plan": AgentPlan(selected_agents=["nutrition"], reasoning="fallback"),
            "needs_clarification": False,
            "clarification_question": "",
        }
