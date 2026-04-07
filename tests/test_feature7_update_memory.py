"""Tests for Feature 7 — Custom Update Memory Prompt.

Verifies that WELLNESS_UPDATE_MEMORY_PROMPT correctly:
  - Defines all four action events (ADD, UPDATE, DELETE, NONE)
  - Covers wellness-specific lifecycles: body metrics, medications, conditions, preferences
  - Is injected into _build_mem0_config alongside Feature 6
  - resolve_memory_actions() returns correct actions for simulated scenarios (unit)
  - resolve_memory_actions() integrates with OpenAI correctly (live, skippable)

Run:
    python -m pytest tests/test_feature7_update_memory.py -v
    python -m pytest tests/test_feature7_update_memory.py -v -m live    # live OpenAI calls
"""

from __future__ import annotations

import asyncio
import os
import pytest


# ── Prompt structure tests (no network) ───────────────────────────────────────

class TestWellnessUpdatePromptStructure:
    """Validate the prompt constant meets Mem0 and wellness requirements."""

    def test_prompt_is_importable(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        assert isinstance(WELLNESS_UPDATE_MEMORY_PROMPT, str)
        assert len(WELLNESS_UPDATE_MEMORY_PROMPT) > 200

    def test_prompt_defines_all_four_events(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        for event in ("ADD", "UPDATE", "DELETE", "NONE"):
            assert event in WELLNESS_UPDATE_MEMORY_PROMPT, f"Missing event: {event}"

    def test_prompt_has_memory_output_key(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        assert '"memory"' in WELLNESS_UPDATE_MEMORY_PROMPT

    def test_prompt_requires_old_memory_for_update(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        assert "old_memory" in WELLNESS_UPDATE_MEMORY_PROMPT

    def test_prompt_covers_body_metrics(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        p = WELLNESS_UPDATE_MEMORY_PROMPT.lower()
        assert "weight" in p or "body metric" in p

    def test_prompt_covers_medication_lifecycle(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        p = WELLNESS_UPDATE_MEMORY_PROMPT.lower()
        assert "medication" in p or "metformin" in p

    def test_prompt_covers_condition_lifecycle(self):
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        p = WELLNESS_UPDATE_MEMORY_PROMPT.lower()
        assert "condition" in p or "resolved" in p

    def test_prompt_has_min_examples_per_event(self):
        """Prompt must include at least one JSON example for each event type."""
        from src.infra.mem0_client import WELLNESS_UPDATE_MEMORY_PROMPT
        for event in ('"ADD"', '"UPDATE"', '"DELETE"', '"NONE"'):
            count = WELLNESS_UPDATE_MEMORY_PROMPT.count(event)
            assert count >= 1, f"Prompt missing example for event {event}"


class TestMem0ConfigIncludesUpdatePrompt:
    """Verify _build_mem0_config injects both custom prompts correctly."""

    def _make_fake_settings(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
            qdrant_url="https://qdrant.test",
            qdrant_api_key="qdrant-key",
            neo4j_url="bolt://neo4j.test",
            neo4j_username="neo4j",
            neo4j_password="password",
            # Feature 9
            mem0_collection_name="reso_memories",
            mem0_reranker_top_k=10,
            # Feature 10
            mem0_llm_provider="openai",
            mem0_llm_model="",
            mem0_llm_temperature=0.0,
            mem0_llm_max_tokens=2000,
            mem0_llm_base_url="",
            anthropic_api_key="",
            groq_api_key="",
            azure_openai_api_key="",
            azure_openai_endpoint="",
            azure_deployment_name="",
            aws_region="us-east-1",
            aws_access_key_id="",
            aws_secret_access_key="",
            # Feature 11
            mem0_vector_store_provider="qdrant",
            supabase_connection_string="",
            chroma_path="./chroma_db",
            chroma_host="",
            chroma_port=8000,
            pinecone_api_key="",
            pinecone_index_name="reso-memories",
            # Feature 12
            mem0_embedder_provider="openai",
            mem0_embedder_model="",
            mem0_embedder_dims=1536,
            mem0_embedder_base_url="",
            # Feature 13
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_model="",
            mem0_reranker_use_wellness_prompt=True,
            cohere_api_key="",
            zero_entropy_api_key="",
            huggingface_api_key="",
        )

    def test_config_has_custom_update_memory_prompt(self):
        from src.infra.mem0_client import _build_mem0_config, WELLNESS_UPDATE_MEMORY_PROMPT
        cfg = _build_mem0_config(self._make_fake_settings())
        assert cfg.get("custom_update_memory_prompt") == WELLNESS_UPDATE_MEMORY_PROMPT

    def test_config_has_both_custom_prompts(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_fake_settings())
        assert "custom_fact_extraction_prompt" in cfg
        assert "custom_update_memory_prompt" in cfg

    def test_config_version_still_v1_1(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_fake_settings())
        assert cfg["version"] == "v1.1"

    def test_config_all_sections_present(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_fake_settings())
        for key in ("llm", "embedder", "vector_store", "graph_store", "reranker"):
            assert key in cfg


# ── resolve_memory_actions unit tests (mock LLM responses) ────────────────────

class TestResolveMemoryActionsUnit:
    """Unit tests using pre-built mock responses to validate the action parsing."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_empty_new_facts_returns_empty(self):
        from src.infra.mem0_client import resolve_memory_actions
        result = self._run(resolve_memory_actions([], []))
        assert result == []

    def test_no_existing_memories_all_add(self):
        """With no existing memories, every new fact must be ADD."""
        from src.infra.mem0_client import resolve_memory_actions
        result = self._run(resolve_memory_actions(
            ["User is vegetarian.", "User exercises 5 days a week."],
            [],
        ))
        assert len(result) == 2
        for action in result:
            assert action["event"] == "ADD"

    def test_fail_open_returns_add_actions(self):
        """When LLM is unavailable, resolve_memory_actions must return ADD for all facts."""
        import unittest.mock as mock
        from src.infra.mem0_client import resolve_memory_actions

        # Patch AsyncOpenAI at the module level so the local import inside the function
        # gets a mock that raises on create(), avoiding "coroutine never awaited" warnings.
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock.AsyncMock(side_effect=Exception("LLM down"))
        with mock.patch("openai.AsyncOpenAI", return_value=mock_client):
            result = self._run(resolve_memory_actions(
                ["User weighs 80 kg."],
                [{"id": "m1", "text": "User weighs 75 kg."}],
            ))
        # Fail-open: should still return ADD actions for all new facts
        assert len(result) >= 1
        for action in result:
            assert action["event"] == "ADD"


# ── Live integration tests (requires OPENAI_API_KEY) ──────────────────────────

@pytest.mark.live
class TestResolveMemoryActionsLive:
    """Live tests against OpenAI. Run with: pytest -m live"""

    @pytest.fixture(autouse=True)
    def require_openai_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_body_metric_returns_update(self):
        """New weight reading should UPDATE the existing weight memory."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User weighs 80 kg."],
            [{"id": "m1", "text": "User weighs 75 kg."}],
        ))
        events = [a["event"] for a in actions]
        assert "UPDATE" in events, f"Expected UPDATE for weight change, got: {actions}"

        update_action = next(a for a in actions if a["event"] == "UPDATE")
        assert update_action.get("id") == "m1", "UPDATE must preserve the original ID"
        assert "old_memory" in update_action, "UPDATE must include old_memory field"
        assert "75" in update_action["old_memory"]

    def test_medication_stopped_returns_delete(self):
        """Stopping a medication should DELETE that memory entry."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User stopped taking metformin."],
            [
                {"id": "m1", "text": "User takes metformin 500mg twice daily."},
                {"id": "m2", "text": "User has type 2 diabetes."},
            ],
        ))
        events = [a["event"] for a in actions]
        assert "DELETE" in events, f"Expected DELETE for stopped medication, got: {actions}"

        delete_action = next(a for a in actions if a["event"] == "DELETE")
        assert "metformin" in delete_action["text"].lower()

    def test_condition_resolved_returns_delete(self):
        """A resolved condition should DELETE that memory entry."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User's hypertension is resolved."],
            [{"id": "m1", "text": "User has hypertension."}],
        ))
        events = [a["event"] for a in actions]
        assert "DELETE" in events, f"Expected DELETE for resolved condition, got: {actions}"

    def test_workout_preference_flip_returns_update(self):
        """Changing workout time preference should UPDATE the existing preference."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User now prefers evening workouts."],
            [{"id": "m1", "text": "User prefers morning workouts."}],
        ))
        events = [a["event"] for a in actions]
        assert "UPDATE" in events, f"Expected UPDATE for preference flip, got: {actions}"

    def test_semantic_duplicate_returns_none(self):
        """Semantically identical fact should return NONE (no change)."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User follows a vegetarian diet."],
            [{"id": "m1", "text": "User is vegetarian."}],
        ))
        events = [a["event"] for a in actions]
        assert "NONE" in events, f"Expected NONE for semantic duplicate, got: {actions}"
        assert "ADD" not in events, "Should not ADD a duplicate"

    def test_new_allergy_returns_add(self):
        """A new allergy not in memory should be ADDed."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User is allergic to shellfish."],
            [{"id": "m1", "text": "User is vegetarian."}],
        ))
        add_actions = [a for a in actions if a["event"] == "ADD"]
        assert len(add_actions) >= 1, f"Expected at least one ADD, got: {actions}"
        assert any("shellfish" in a["text"].lower() for a in add_actions)

    def test_all_actions_have_required_fields(self):
        """Every returned action must have id, text, and event."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User weighs 82 kg.", "User started taking lisinopril 5mg once daily."],
            [{"id": "m1", "text": "User weighs 80 kg."}],
        ))
        for action in actions:
            assert "id" in action, f"Missing 'id' in action: {action}"
            assert "text" in action, f"Missing 'text' in action: {action}"
            assert "event" in action, f"Missing 'event' in action: {action}"
            assert action["event"] in {"ADD", "UPDATE", "DELETE", "NONE"}

    def test_update_always_has_old_memory(self):
        """Every UPDATE action must include old_memory for audit trail."""
        from src.infra.mem0_client import resolve_memory_actions
        actions = self._run(resolve_memory_actions(
            ["User weighs 83 kg."],
            [{"id": "m1", "text": "User weighs 80 kg."}],
        ))
        for action in actions:
            if action["event"] == "UPDATE":
                assert "old_memory" in action, f"UPDATE missing old_memory: {action}"
                assert action["old_memory"], "old_memory must not be empty on UPDATE"
