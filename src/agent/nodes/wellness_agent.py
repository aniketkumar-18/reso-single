"""Wellness Agent Node — single ReAct agent handling all three domains.

This replaces the classify_route + [nutrition_agent, fitness_agent, medical_agent] + aggregator
pipeline from the multi-agent version. One agent, all tools, unified context.

Design decisions:
- ALL 13 tools are available (profile × 2 + nutrition × 3 + fitness × 4 + medical × 4).
  Tool deduplication: NUTRITION/FITNESS/MEDICAL_TOOLS each include PROFILE_TOOLS, so we
  build ALL_TOOLS explicitly to avoid registering the same tool twice.
- Conversation history is injected as Human/AI message pairs so the agent has proper
  multi-turn context without needing a separate classify step.
- The LLM is tagged "final_response" so the gateway's SSE stream can surface real tokens.
  Tool-calling intermediate steps are naturally filtered out by the streaming code
  (they carry tool_call_chunks, not plain content).
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import (
    build_constraints_section,
    build_graph_relations_section,
    build_meal_context_section,
    build_memory_section,
    build_profile_section,
    build_response_format,
    WELLNESS_AGENT_SYSTEM,
)
from src.agent.state import GraphState
from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.tools.profile import update_user_profile, save_memory_fact
from src.tools.nutrition import get_meal_items, log_meal, edit_meal_item, delete_meal_item
from src.tools.fitness import log_workout, log_body_metrics, update_fitness_goals, calculate_macro_targets
from src.tools.medical import log_medical_condition, update_medical_condition, log_medication, update_medication
from src.tools.multimodal import log_from_image
from src.infra.mem0_client import auto_save_conversation_memory

logger = setup_logger(__name__)

# All tools — explicit list avoids duplicating PROFILE_TOOLS that are bundled
# inside each domain's tool list.
ALL_TOOLS = [
    # Profile & memory (shared)
    update_user_profile,
    save_memory_fact,
    # Nutrition
    get_meal_items,
    log_meal,
    edit_meal_item,
    delete_meal_item,
    # Fitness
    log_workout,
    log_body_metrics,
    update_fitness_goals,
    calculate_macro_targets,
    # Medical
    log_medical_condition,
    update_medical_condition,
    log_medication,
    update_medication,
    # Multimodal
    log_from_image,
]


async def wellness_agent_node(state: GraphState, config: RunnableConfig) -> dict:  # noqa: ARG001
    """Single ReAct agent that handles nutrition, fitness, and medical domains."""
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    conversation_id = state.get("conversation_id", "")
    user_message = state.get("user_message", "")

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model_agent,
        temperature=0,
        api_key=settings.openai_api_key,
        streaming=True,
    ).with_config({"tags": ["final_response"]})

    system_prompt = WELLNESS_AGENT_SYSTEM.format(
        profile_section=build_profile_section(state.get("user_profile", {})),
        memory_section=build_memory_section(state.get("memory_context", [])),
        graph_relations_section=build_graph_relations_section(state.get("graph_relations", [])),
        constraints_section=build_constraints_section(state.get("constraint_rules", [])),
        meal_context_section=build_meal_context_section(state.get("meal_items_context", [])),
        response_format=build_response_format(),
    )

    agent = create_react_agent(model=llm, tools=ALL_TOOLS, prompt=system_prompt)

    # Build message list: conversation history provides multi-turn context;
    # the current message is appended last.
    messages: list = []
    for msg in state.get("conversation_history", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Build the current user message — multimodal when image is attached
    image_url = state.get("image_url", "")
    if image_url:
        messages.append(HumanMessage(content=[
            {"type": "text", "text": user_message or "Please analyse this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]))
    else:
        messages.append(HumanMessage(content=user_message))

    agent_config: dict = {
        "configurable": {
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
        }
    }

    try:
        result = await agent.ainvoke({"messages": messages}, config=agent_config)

        result_messages = result.get("messages", [])
        final_content = ""
        refresh_entities: list[str] = []
        tool_calls_made: list[str] = []

        # Collect tool names from AI messages that had tool calls
        for msg in result_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)

        # Final response = last AI message that has no pending tool calls
        for msg in result_messages:
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                final_content = str(msg.content)

        # Collect refresh entity keys from ToolMessage JSON payloads
        for msg in result_messages:
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

        logger.info(
            "Wellness agent done — tools=%s refresh=%s",
            tool_calls_made,
            refresh_entities,
        )

        # Feature 8 — OpenAI Compatibility: auto-save any wellness facts the agent
        # may not have explicitly captured via save_memory_fact (e.g. incidental
        # mentions). Non-critical: timeout after 15 s, never delays the response.
        if final_content and user_id:
            try:
                import asyncio as _asyncio
                await _asyncio.wait_for(
                    auto_save_conversation_memory(
                        user_message=user_message,
                        assistant_response=final_content,
                        user_id=user_id,
                        run_id=session_id or None,
                    ),
                    timeout=15.0,
                )
            except Exception:
                logger.debug("auto_save_conversation_memory skipped (non-critical)")

        return {"aggregated_response": final_content, "refresh_entities": refresh_entities}

    except Exception:
        logger.exception("Wellness agent failed")
        return {
            "aggregated_response": "I encountered an error processing your request. Please try again.",
            "refresh_entities": [],
        }
