"""Aggregator Node — Zone 1, final step.

Merges outputs from one or more domain agents into a single coherent response.
For single-agent outputs, returns content directly (no LLM overhead).
For multi-agent outputs, uses a synthesis LLM call.
Output feeds into Zone 2 (post-execution pipeline).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.orchestrator.prompts import AGGREGATOR_SYSTEM, build_response_format
from src.orchestrator.state import AgentOutput, GraphState

logger = setup_logger(__name__)


def _collect_refresh_entities(outputs: list[AgentOutput]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for out in outputs:
        for entity in out.refresh_entities:
            if entity not in seen:
                seen.add(entity)
                result.append(entity)
    return result


async def aggregator_node(state: GraphState) -> dict:
    """Merge agent outputs into a single response string."""

    # Clarification short-circuit — no agents ran, just return the question
    if state.get("needs_clarification"):
        question = state.get("clarification_question") or "Could you provide a bit more detail?"
        logger.info("Aggregator: returning clarification question")
        return {"aggregated_response": question, "refresh_entities": []}

    outputs: list[AgentOutput] = state.get("agent_outputs", [])

    if not outputs:
        logger.warning("Aggregator received no agent outputs")
        return {
            "aggregated_response": "I was unable to generate a response. Please try again.",
            "refresh_entities": [],
        }

    refresh_entities = _collect_refresh_entities(outputs)

    # Single agent — return directly, no synthesis overhead
    if len(outputs) == 1:
        logger.info("Aggregator: single agent output, passing through")
        return {
            "aggregated_response": outputs[0].content,
            "refresh_entities": refresh_entities,
        }

    # Multi-agent — synthesise into one coherent response
    logger.info("Aggregator: synthesising %d agent outputs", len(outputs))

    domain_sections = "\n\n".join(
        f"=== {out.domain.upper()} AGENT ===\n{out.content}"
        for out in outputs
    )
    synthesis_prompt = (
        f"User question: {state.get('user_message', '')}\n\n"
        f"Agent outputs:\n{domain_sections}"
    )

    llm = ChatOpenAI(
        model=get_settings().llm_model_pipeline,
        temperature=0.2,
        api_key=get_settings().openai_api_key,
    ).with_config({"tags": ["final_response"]})

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=AGGREGATOR_SYSTEM.format(response_format=build_response_format())),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        aggregated = str(response.content)
    except Exception:
        logger.exception("Aggregator synthesis failed — concatenating outputs")
        aggregated = "\n\n".join(
            f"**{out.domain.title()}:** {out.content}" for out in outputs
        )

    return {
        "aggregated_response": aggregated,
        "refresh_entities": refresh_entities,
    }
