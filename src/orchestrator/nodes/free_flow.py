"""Free-Flow Node — Zone 1 (left branch).

A single LLM call with full context injected into the system prompt.
No tools are called. Output goes directly to the aggregator.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.orchestrator.prompts import (
    FREE_FLOW_SYSTEM,
    build_constraints_section,
    build_memory_section,
    build_profile_section,
    build_response_format,
)
from src.orchestrator.state import AgentOutput, GraphState

logger = setup_logger(__name__)


async def free_flow_node(state: GraphState) -> dict:
    """Answer the user's question using context from hydration, without tools."""
    profile = state.get("user_profile", {})
    memories = state.get("memory_context", [])
    constraints = state.get("constraint_rules", [])
    history = state.get("conversation_history", [])
    user_message = state.get("user_message", "")

    system_prompt = FREE_FLOW_SYSTEM.format(
        profile_section=build_profile_section(profile),
        memory_section=build_memory_section(memories),
        constraints_section=build_constraints_section(constraints),
        response_format=build_response_format(),
    )

    # Build message list: system + history + current turn
    messages: list = [SystemMessage(content=system_prompt)]
    for msg in history[-10:]:  # last 10 turns for context window efficiency
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    llm = ChatOpenAI(
        model=get_settings().llm_model_agent,
        temperature=0.4,
        api_key=get_settings().openai_api_key,
    ).with_config({"tags": ["final_response"]})

    try:
        response = await llm.ainvoke(messages)
        content = str(response.content)
    except Exception:
        logger.exception("Free-flow LLM call failed")
        content = "I'm sorry, I encountered an error processing your request. Please try again."

    logger.info("Free-flow response generated (%d chars)", len(content))

    output = AgentOutput(
        domain="general",
        content=content,
        refresh_entities=[],
        tool_calls_made=[],
    )
    return {"agent_outputs": [output]}
