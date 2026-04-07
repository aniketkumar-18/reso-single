"""Zone 1 LangGraph — Ingress & Orchestration.

Graph topology (parallel hydrate + route, with pre-agent clarification check):

    START
      ├─────────────────────────┐
      ▼                         ▼
  context_hydration         classify_route      ← run concurrently (~350 ms saved)
      │                         │               ← also detects missing required info
      └─────────┬───────────────┘
                ▼
              _join                             ← no-op fan-in sync node
                │
                ├── needs_clarification=True ──▶ aggregator ──▶ END  (returns question)
                │
                └── needs_clarification=False ─▶ [Send API dispatcher]
                                                        │
                                            ┌───────────┼───────────┐
                                            ▼           ▼           ▼
                                    nutrition_agent fitness_agent medical_agent
                                            │           │           │
                                            └───────────┴───────────┘
                                                        │
                                                   aggregator
                                                        │
                                                       END   ──▶ Zone 2

Key patterns:
- context_hydration and classify_route run in parallel from START (~350 ms saved).
- _join is a no-op synchronisation node; LangGraph waits for both branches.
- LangGraph Send API powers the parallel fan-out from _join → selected agents.
- Agents decide internally when to call tools vs respond conversationally.
- agent_outputs reducer (operator.add) lets concurrent agents write safely.
- Redis checkpointer enables pod-crash recovery (any pod resumes any session).
"""

from __future__ import annotations

from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.fitness import fitness_agent_node
from src.agents.medical import medical_agent_node
from src.agents.nutrition import nutrition_agent_node
from src.infra.logger import setup_logger
from src.orchestrator.nodes.aggregator import aggregator_node
from src.orchestrator.nodes.classify_route import classify_route_node
from src.orchestrator.nodes.context_hydration import context_hydration_node
from src.orchestrator.state import AgentPlan, GraphState

logger = setup_logger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _join_node(_state: GraphState) -> dict:
    """Fan-in sync node. Resets agent_outputs so stale checkpoint data never bleeds
    into a new turn (the custom reducer treats [] as a clear signal)."""
    return {"agent_outputs": []}


def _dispatch_agents(state: GraphState) -> list[Send]:
    """Fan-out: fire each selected agent concurrently via LangGraph Send API."""
    plan: AgentPlan | None = state.get("agent_plan")
    if not plan or not plan.selected_agents:
        logger.warning("Dispatcher: no agent plan — sending to nutrition as fallback")
        return [Send("nutrition_agent", state)]

    sends = [Send(f"{agent}_agent", state) for agent in plan.selected_agents]
    logger.info("Dispatcher: sending to agents=%s", [s.node for s in sends])
    return sends


def _route_after_join(state: GraphState) -> str | list[Send]:
    """Route after _join: short-circuit to aggregator for clarification, else fan-out."""
    if state.get("needs_clarification"):
        logger.info("Router: clarification needed — skipping agents")
        return "aggregator"
    return _dispatch_agents(state)


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("context_hydration", context_hydration_node)
    graph.add_node("classify_route", classify_route_node)
    graph.add_node("_join", _join_node)
    graph.add_node("nutrition_agent", nutrition_agent_node)
    graph.add_node("fitness_agent", fitness_agent_node)
    graph.add_node("medical_agent", medical_agent_node)
    graph.add_node("aggregator", aggregator_node)

    # ── Parallel start ─────────────────────────────────────────────────────────
    graph.add_edge(START, "context_hydration")
    graph.add_edge(START, "classify_route")

    # ── Fan-in at _join (waits for both branches) ──────────────────────────────
    graph.add_edge("context_hydration", "_join")
    graph.add_edge("classify_route", "_join")

    # ── Fan-out from _join → agents, or short-circuit → aggregator ────────────
    graph.add_conditional_edges(
        "_join",
        _route_after_join,
        ["nutrition_agent", "fitness_agent", "medical_agent", "aggregator"],
    )

    # ── Convergence → aggregator ───────────────────────────────────────────────
    graph.add_edge("nutrition_agent", "aggregator")
    graph.add_edge("fitness_agent", "aggregator")
    graph.add_edge("medical_agent", "aggregator")

    graph.add_edge("aggregator", END)

    return graph


# ── Compiled graph (module-level singleton) ────────────────────────────────────

def compile_graph(checkpointer):
    """Compile the graph with the given checkpointer. Call after asetup()."""
    graph = build_graph()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph Zone 1 compiled (parallel classify+hydrate)")
    return compiled


# ── Graph invocation helpers ───────────────────────────────────────────────────

def _build_initial_state(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
) -> GraphState:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "user_profile": {},
        "conversation_history": [],
        "memory_context": [],
        "constraint_rules": [],
        "agent_plan": None,
        "needs_clarification": False,
        "clarification_question": "",
        "agent_outputs": [],
        "aggregated_response": "",
        "refresh_entities": [],
    }


async def invoke_graph(
    compiled_graph,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
) -> GraphState:
    """Run the full Zone 1 graph and return the final state."""
    config = {"configurable": {"thread_id": session_id}}
    initial_state = _build_initial_state(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_message=user_message,
    )
    final_state: GraphState = await compiled_graph.ainvoke(initial_state, config=config)
    return final_state


async def stream_graph_events(
    compiled_graph,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Yield (event_kind, event) tuples from the graph via astream_events v2.

    Callers filter on:
      - kind == "on_chat_model_stream"  →  real token chunks
      - kind == "on_chain_end"          →  node completion + final state
    """
    config = {"configurable": {"thread_id": session_id}}
    initial_state = _build_initial_state(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_message=user_message,
    )
    async for event in compiled_graph.astream_events(
        initial_state, config=config, version="v2"
    ):
        yield event["event"], event
