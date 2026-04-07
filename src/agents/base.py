"""Base agent factory."""

from __future__ import annotations

import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.orchestrator.state import AgentOutput, DomainName, GraphState

logger = setup_logger(__name__)


async def run_domain_agent(
    *,
    domain: DomainName,
    tools: list[BaseTool],
    system_prompt: str,
    state: GraphState,
) -> dict:
    """Run a ReAct agent loop for a single domain and return the state delta.

    user_id and conversation_id are forwarded via RunnableConfig.configurable
    so every tool can read them without being in the agent's message state.
    """
    user_id = state.get("user_id", "")
    conversation_id = state.get("conversation_id", "")
    user_message = state.get("user_message", "")

    llm = ChatOpenAI(
        model=get_settings().llm_model_agent,
        temperature=0,
        api_key=get_settings().openai_api_key,
    )

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    config = {
        "configurable": {
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
    }

    # Reconstruct conversation history so the agent can look up previously embedded
    # meal-ids and other turn-level context (e.g. for edit/delete flows).
    messages: list = []
    for msg in state.get("conversation_history", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    try:
        result = await agent.ainvoke(
            {"messages": messages},
            config=config,
        )

        messages = result.get("messages", [])
        final_content = ""
        tool_calls_made: list[str] = []
        refresh_entities: list[str] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)
            # Final AI message (no tool calls)
            if hasattr(msg, "content") and not getattr(msg, "tool_calls", None):
                final_content = str(msg.content)

        # Collect refresh keys from ToolMessages
        for msg in messages:
            raw = getattr(msg, "content", "")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and "refresh" in payload:
                        entity = payload["refresh"]
                        if entity not in refresh_entities:
                            refresh_entities.append(entity)
                except (json.JSONDecodeError, TypeError):
                    pass

        output = AgentOutput(
            domain=domain,
            content=final_content,
            refresh_entities=refresh_entities,
            tool_calls_made=tool_calls_made,
        )
        logger.info("%s agent done — tools=%s refresh=%s", domain, tool_calls_made, refresh_entities)
        return {"agent_outputs": [output]}

    except Exception:
        logger.exception("%s agent failed", domain)
        return {"agent_outputs": [AgentOutput(
            domain=domain,
            content=f"The {domain} specialist encountered an error. Please try again.",
        )]}
