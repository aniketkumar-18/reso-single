"""Medical domain agent node."""

from __future__ import annotations

from src.agents.base import run_domain_agent
from src.orchestrator.prompts import (
    MEDICAL_AGENT_SYSTEM,
    build_constraints_section,
    build_profile_section,
    build_response_format,
)
from src.orchestrator.state import GraphState
from src.tools.medical import MEDICAL_TOOLS


async def medical_agent_node(state: GraphState) -> dict:
    """ReAct agent specialised in medical conditions and medication logging."""
    system_prompt = MEDICAL_AGENT_SYSTEM.format(
        profile_section=build_profile_section(state.get("user_profile", {})),
        constraints_section=build_constraints_section(state.get("constraint_rules", [])),
        response_format=build_response_format(),
    )
    return await run_domain_agent(
        domain="medical",
        tools=MEDICAL_TOOLS,
        system_prompt=system_prompt,
        state=state,
    )
