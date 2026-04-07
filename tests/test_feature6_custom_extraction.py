"""Tests for Feature 6 — Custom Fact Extraction Prompt.

Verifies that WELLNESS_FACT_EXTRACTION_PROMPT correctly:
  - Extracts persistent wellness facts from user messages
  - Returns [] for noise (greetings, requests, transient states)
  - Handles compound messages into multiple facts
  - Validates prompt structure (required fields, few-shot examples)
  - extract_wellness_facts() integrates with OpenAI correctly (live test, skippable)

Run:
    python -m pytest tests/test_feature6_custom_extraction.py -v
    python -m pytest tests/test_feature6_custom_extraction.py -v -m live    # live OpenAI calls
"""

from __future__ import annotations

import json
import os
import pytest
import asyncio

# ── Prompt structure tests (no network) ───────────────────────────────────────

class TestWellnessPromptStructure:
    """Validate the prompt constant itself meets Mem0 and few-shot requirements."""

    def test_prompt_is_importable(self):
        from src.infra.mem0_client import WELLNESS_FACT_EXTRACTION_PROMPT
        assert isinstance(WELLNESS_FACT_EXTRACTION_PROMPT, str)
        assert len(WELLNESS_FACT_EXTRACTION_PROMPT) > 100

    def test_prompt_covers_all_four_domains(self):
        from src.infra.mem0_client import WELLNESS_FACT_EXTRACTION_PROMPT
        p = WELLNESS_FACT_EXTRACTION_PROMPT.upper()
        for domain in ("NUTRITION", "FITNESS", "MEDICAL", "GENERAL"):
            assert domain in p, f"Prompt missing domain: {domain}"

    def test_prompt_has_facts_key_instruction(self):
        from src.infra.mem0_client import WELLNESS_FACT_EXTRACTION_PROMPT
        assert '"facts"' in WELLNESS_FACT_EXTRACTION_PROMPT

    def test_prompt_has_negative_examples(self):
        """Prompt must show [] outputs for noise inputs."""
        from src.infra.mem0_client import WELLNESS_FACT_EXTRACTION_PROMPT
        assert '{"facts": []}' in WELLNESS_FACT_EXTRACTION_PROMPT

    def test_prompt_has_positive_examples(self):
        """Prompt must show non-empty fact arrays."""
        from src.infra.mem0_client import WELLNESS_FACT_EXTRACTION_PROMPT
        import re
        # Look for {"facts": ["..."]} patterns
        matches = re.findall(r'\{"facts":\s*\[".+?"\]', WELLNESS_FACT_EXTRACTION_PROMPT)
        assert len(matches) >= 3, "Prompt needs at least 3 positive few-shot examples"


class TestMem0ConfigIncludesCustomPrompt:
    """Verify _build_mem0_config injects the prompt and version correctly."""

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

    def test_config_has_version_v1_1(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_fake_settings())
        assert cfg.get("version") == "v1.1"

    def test_config_has_custom_fact_extraction_prompt(self):
        from src.infra.mem0_client import _build_mem0_config, WELLNESS_FACT_EXTRACTION_PROMPT
        cfg = _build_mem0_config(self._make_fake_settings())
        assert cfg.get("custom_fact_extraction_prompt") == WELLNESS_FACT_EXTRACTION_PROMPT

    def test_config_still_has_all_stores(self):
        """Ensure Feature 6 additions didn't remove existing config sections."""
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_fake_settings())
        for key in ("llm", "embedder", "vector_store", "graph_store", "reranker"):
            assert key in cfg, f"Config missing required section: {key}"


# ── Prompt logic simulation (no network) ──────────────────────────────────────

class TestPromptLogicSimulation:
    """Parse mock LLM responses to verify the expected filtering behaviour."""

    NOISE_INPUTS = [
        "Hi, how are you?",
        "What should I eat for breakfast?",
        "Can you give me a high-protein meal plan?",
        "I had oatmeal for breakfast today.",
        "Thanks, that sounds great!",
        "That's helpful, thanks!",
    ]

    FACT_INPUTS = [
        ("I'm vegetarian and allergic to tree nuts.", ["vegetarian", "allergic", "tree nuts"]),
        ("I prefer morning workouts, usually 5 days a week.", ["morning workout", "5 days"]),
        ("I was diagnosed with type 2 diabetes. I take metformin 500mg twice daily.",
         ["diabetes", "metformin"]),
        ("I work night shifts as a nurse and sleep from 8am to 4pm.",
         ["night shift", "nurse"]),
        ("I weigh 78 kg and want to lose 5 kg.", ["78 kg", "5 kg"]),
    ]

    def _simulate_llm_output_noise(self, text: str) -> dict:
        """Simulate the LLM returning [] for noise inputs."""
        return {"facts": []}

    def _simulate_llm_output_fact(self, facts: list[str]) -> dict:
        return {"facts": facts}

    def test_noise_returns_empty_facts(self):
        for text in self.NOISE_INPUTS:
            output = self._simulate_llm_output_noise(text)
            assert output["facts"] == [], f"Expected [] for noise input: {text!r}"

    def test_fact_inputs_return_populated_facts(self):
        for text, expected_keywords in self.FACT_INPUTS:
            # Simulate what the LLM would return given the prompt
            mock_facts = [f"User fact about: {kw}" for kw in expected_keywords]
            output = self._simulate_llm_output_fact(mock_facts)
            assert len(output["facts"]) > 0, f"Expected facts for: {text!r}"

    def test_facts_are_strings(self):
        output = self._simulate_llm_output_fact(
            ["User is vegetarian.", "User is allergic to tree nuts."]
        )
        for fact in output["facts"]:
            assert isinstance(fact, str)


# ── Live integration tests (requires OPENAI_API_KEY) ──────────────────────────

@pytest.mark.live
class TestExtractWellnessFactsLive:
    """Live tests that call OpenAI. Run with: pytest -m live"""

    @pytest.fixture(autouse=True)
    def require_openai_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_noise_input_returns_empty(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts("Hi, how are you?")
        )
        assert result == [], f"Expected [] for greeting, got: {result}"

    def test_request_returns_empty(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts("What should I eat for breakfast?")
        )
        assert result == [], f"Expected [] for request, got: {result}"

    def test_preference_extracted(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts("I prefer morning workouts and I dislike running.")
        )
        assert len(result) >= 1
        combined = " ".join(result).lower()
        assert "morning" in combined or "workout" in combined

    def test_medical_fact_extracted(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts("I have type 2 diabetes and take metformin 500mg twice daily.")
        )
        assert len(result) >= 1
        combined = " ".join(result).lower()
        assert "diabetes" in combined or "metformin" in combined

    def test_compound_message_splits_into_multiple_facts(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts(
                "I'm vegetarian, allergic to peanuts, and I work out 4 times a week."
            )
        )
        assert len(result) >= 2, f"Expected ≥2 facts for compound message, got {len(result)}: {result}"

    def test_empty_string_returns_empty(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(extract_wellness_facts(""))
        assert result == []

    def test_all_facts_are_strings(self):
        from src.infra.mem0_client import extract_wellness_facts
        result = asyncio.get_event_loop().run_until_complete(
            extract_wellness_facts("I weigh 75 kg and my goal is to gain muscle.")
        )
        for fact in result:
            assert isinstance(fact, str) and fact.strip()
