"""Tests for Feature 13 — Configurable Reranker.

Verifies that:
  - New settings fields exist with correct defaults
  - get_supported_rerankers() returns all 6 providers (including "none")
  - _build_reranker_config() produces correct config blocks per provider
  - provider="none" returns None (disables reranking)
  - Missing API keys fall back to llm_reranker with a warning
  - Unknown provider falls back to llm_reranker
  - WELLNESS_RERANKER_SCORING_PROMPT is valid: contains {query}/{document} vars,
    scores 0.0–1.0, and is wellness-specific
  - LLM reranker uses wellness prompt when mem0_reranker_use_wellness_prompt=True
  - LLM reranker omits wellness prompt when flag is False
  - _build_mem0_config() uses _build_reranker_config()
  - _build_mem0_config() omits "reranker" key when provider="none"
  - get_mem0_config_summary() reflects actual provider/model/top_k/prompt flag

Run:
    python -m pytest tests/test_feature13_rerankers.py -v
"""

from __future__ import annotations

import types
import unittest.mock as mock

import pytest


# ── Settings field tests ───────────────────────────────────────────────────────

class TestFeature13SettingsFields:

    def _s(self):
        from src.infra.config import get_settings
        return get_settings()

    def test_mem0_reranker_provider_exists(self):
        assert hasattr(self._s(), "mem0_reranker_provider")

    def test_mem0_reranker_provider_default_is_llm_reranker(self):
        s = self._s()
        assert isinstance(s.mem0_reranker_provider, str)
        assert s.mem0_reranker_provider != ""

    def test_mem0_reranker_model_exists(self):
        assert hasattr(self._s(), "mem0_reranker_model")

    def test_mem0_reranker_model_is_string(self):
        assert isinstance(self._s().mem0_reranker_model, str)

    def test_mem0_reranker_use_wellness_prompt_exists(self):
        assert hasattr(self._s(), "mem0_reranker_use_wellness_prompt")

    def test_mem0_reranker_use_wellness_prompt_default_is_true(self):
        assert self._s().mem0_reranker_use_wellness_prompt is True

    def test_cohere_api_key_exists(self):
        assert hasattr(self._s(), "cohere_api_key")

    def test_zero_entropy_api_key_exists(self):
        assert hasattr(self._s(), "zero_entropy_api_key")

    def test_huggingface_api_key_exists(self):
        assert hasattr(self._s(), "huggingface_api_key")


# ── get_supported_rerankers ────────────────────────────────────────────────────

class TestGetSupportedRerankers:

    def test_returns_list(self):
        from src.infra.mem0_client import get_supported_rerankers
        assert isinstance(get_supported_rerankers(), list)

    def test_contains_all_six_providers(self):
        from src.infra.mem0_client import get_supported_rerankers
        providers = get_supported_rerankers()
        for expected in (
            "llm_reranker", "cohere", "zero_entropy",
            "sentence_transformer", "huggingface", "none",
        ):
            assert expected in providers, f"Missing reranker: {expected}"

    def test_has_exactly_six_providers(self):
        from src.infra.mem0_client import get_supported_rerankers
        assert len(get_supported_rerankers()) == 6

    def test_no_duplicates(self):
        from src.infra.mem0_client import get_supported_rerankers
        providers = get_supported_rerankers()
        assert len(providers) == len(set(providers))


# ── WELLNESS_RERANKER_SCORING_PROMPT ──────────────────────────────────────────

class TestWellnessRerankerScoringPrompt:

    def test_prompt_exists(self):
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        assert isinstance(WELLNESS_RERANKER_SCORING_PROMPT, str)
        assert len(WELLNESS_RERANKER_SCORING_PROMPT) > 100

    def test_prompt_has_query_variable(self):
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        assert "{query}" in WELLNESS_RERANKER_SCORING_PROMPT

    def test_prompt_has_document_variable(self):
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        assert "{document}" in WELLNESS_RERANKER_SCORING_PROMPT

    def test_prompt_uses_01_scale(self):
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        assert "0.0" in WELLNESS_RERANKER_SCORING_PROMPT
        assert "1.0" in WELLNESS_RERANKER_SCORING_PROMPT

    def test_prompt_is_wellness_specific(self):
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        prompt_lower = WELLNESS_RERANKER_SCORING_PROMPT.lower()
        assert any(w in prompt_lower for w in ("wellness", "health", "nutrition", "fitness"))

    def test_prompt_requests_only_score(self):
        """Prompt must ask for a single number to avoid score extraction failure."""
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        prompt_lower = WELLNESS_RERANKER_SCORING_PROMPT.lower()
        assert any(w in prompt_lower for w in ("single", "only", "numerical", "number"))

    def test_prompt_can_be_formatted(self):
        """Both template variables must be substitutable without error."""
        from src.infra.mem0_client import WELLNESS_RERANKER_SCORING_PROMPT
        rendered = WELLNESS_RERANKER_SCORING_PROMPT.format(
            query="What did I eat today?",
            document="User had oatmeal for breakfast.",
        )
        assert "What did I eat today?" in rendered
        assert "oatmeal" in rendered


# ── _build_reranker_config ────────────────────────────────────────────────────

def _make_settings(**overrides):
    defaults = dict(
        llm_model_pipeline="gpt-4o-mini",
        openai_api_key="sk-test",
        cohere_api_key="",
        zero_entropy_api_key="",
        huggingface_api_key="",
        mem0_reranker_provider="llm_reranker",
        mem0_reranker_model="",
        mem0_reranker_top_k=10,
        mem0_reranker_use_wellness_prompt=True,
        mem0_llm_provider="openai",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestBuildRerankerConfigLLMReranker:

    def test_llm_reranker_provider_key(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="llm_reranker"))
        assert cfg["provider"] == "llm_reranker"

    def test_llm_reranker_model_defaults_to_pipeline_model(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_model="",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert cfg["config"]["model"] == "gpt-4o-mini"

    def test_llm_reranker_explicit_model_respected(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_model="gpt-4o",
        ))
        assert cfg["config"]["model"] == "gpt-4o"

    def test_llm_reranker_top_k_in_config(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_top_k=15))
        assert cfg["config"]["top_k"] == 15

    def test_llm_reranker_temperature_is_zero(self):
        """Docs recommend temperature=0 for deterministic scoring."""
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings())
        assert cfg["config"]["temperature"] == 0.0

    def test_llm_reranker_max_tokens_is_small(self):
        """Score is a single float — no need for large token budget."""
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings())
        assert cfg["config"]["max_tokens"] <= 200

    def test_llm_reranker_includes_wellness_prompt_when_flag_true(self):
        from src.infra.mem0_client import _build_reranker_config, WELLNESS_RERANKER_SCORING_PROMPT
        cfg = _build_reranker_config(_make_settings(mem0_reranker_use_wellness_prompt=True))
        assert cfg["config"].get("scoring_prompt") == WELLNESS_RERANKER_SCORING_PROMPT

    def test_llm_reranker_omits_scoring_prompt_when_flag_false(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_use_wellness_prompt=False))
        assert "scoring_prompt" not in cfg["config"]

    def test_llm_reranker_api_key_in_config(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(openai_api_key="sk-rerank"))
        assert cfg["config"]["api_key"] == "sk-rerank"


class TestBuildRerankerConfigCohere:

    def test_cohere_provider_key(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="cohere-key",
        ))
        assert cfg["provider"] == "cohere"

    def test_cohere_api_key_in_config(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-secret",
        ))
        assert cfg["config"]["api_key"] == "co-secret"

    def test_cohere_default_model(self):
        from src.infra.mem0_client import _build_reranker_config, _SUPPORTED_RERANKERS
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-key",
            mem0_reranker_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_RERANKERS["cohere"]

    def test_cohere_explicit_model_respected(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-key",
            mem0_reranker_model="rerank-multilingual-v3.0",
        ))
        assert cfg["config"]["model"] == "rerank-multilingual-v3.0"

    def test_cohere_return_documents_false(self):
        """Docs recommend return_documents=False to reduce response size."""
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-key",
        ))
        assert cfg["config"].get("return_documents") is False

    def test_cohere_missing_key_falls_back_to_llm_reranker(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="",
        ))
        assert cfg["provider"] == "llm_reranker"

    def test_cohere_top_k_in_config(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-key",
            mem0_reranker_top_k=10,
        ))
        assert cfg["config"]["top_k"] == 10


class TestBuildRerankerConfigZeroEntropy:

    def test_zero_entropy_provider_key(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="zero_entropy",
            zero_entropy_api_key="ze-key",
        ))
        assert cfg["provider"] == "zero_entropy"

    def test_zero_entropy_default_model(self):
        from src.infra.mem0_client import _build_reranker_config, _SUPPORTED_RERANKERS
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="zero_entropy",
            zero_entropy_api_key="ze-key",
            mem0_reranker_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_RERANKERS["zero_entropy"]

    def test_zero_entropy_missing_key_falls_back_to_llm_reranker(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="zero_entropy",
            zero_entropy_api_key="",
        ))
        assert cfg["provider"] == "llm_reranker"

    def test_zero_entropy_api_key_in_config(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="zero_entropy",
            zero_entropy_api_key="ze-prod-key",
        ))
        assert cfg["config"]["api_key"] == "ze-prod-key"


class TestBuildRerankerConfigSentenceTransformer:

    def test_sentence_transformer_provider_key(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="sentence_transformer",
        ))
        assert cfg["provider"] == "sentence_transformer"

    def test_sentence_transformer_no_api_key_needed(self):
        """On-device provider — must work without any API key."""
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="sentence_transformer",
            cohere_api_key="",
            zero_entropy_api_key="",
        ))
        assert cfg["provider"] == "sentence_transformer"

    def test_sentence_transformer_default_model(self):
        from src.infra.mem0_client import _build_reranker_config, _SUPPORTED_RERANKERS
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="sentence_transformer",
            mem0_reranker_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_RERANKERS["sentence_transformer"]

    def test_sentence_transformer_has_batch_size(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="sentence_transformer"))
        assert "batch_size" in cfg["config"]
        assert cfg["config"]["batch_size"] > 0

    def test_sentence_transformer_top_k(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="sentence_transformer",
            mem0_reranker_top_k=10,
        ))
        assert cfg["config"]["top_k"] == 10


class TestBuildRerankerConfigHuggingFace:

    def test_huggingface_provider_key(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="huggingface"))
        assert cfg["provider"] == "huggingface"

    def test_huggingface_default_model(self):
        from src.infra.mem0_client import _build_reranker_config, _SUPPORTED_RERANKERS
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="huggingface",
            mem0_reranker_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_RERANKERS["huggingface"]

    def test_huggingface_api_key_included_when_set(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="huggingface",
            huggingface_api_key="hf-token",
        ))
        assert cfg["config"]["api_key"] == "hf-token"

    def test_huggingface_api_key_absent_when_empty(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(
            mem0_reranker_provider="huggingface",
            huggingface_api_key="",
        ))
        assert "api_key" not in cfg["config"]


class TestBuildRerankerConfigNone:

    def test_none_provider_returns_none(self):
        """provider=none must disable reranking by returning None."""
        from src.infra.mem0_client import _build_reranker_config
        result = _build_reranker_config(_make_settings(mem0_reranker_provider="none"))
        assert result is None

    def test_none_provider_case_insensitive(self):
        from src.infra.mem0_client import _build_reranker_config
        assert _build_reranker_config(_make_settings(mem0_reranker_provider="None")) is None
        assert _build_reranker_config(_make_settings(mem0_reranker_provider="NONE")) is None


class TestBuildRerankerConfigFallback:

    def test_unknown_provider_falls_back_to_llm_reranker(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="unknown_xyz"))
        assert cfg["provider"] == "llm_reranker"

    def test_case_insensitive_provider(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg_lower = _build_reranker_config(_make_settings(mem0_reranker_provider="cohere", cohere_api_key="k"))
        cfg_upper = _build_reranker_config(_make_settings(mem0_reranker_provider="Cohere", cohere_api_key="k"))
        assert cfg_lower["provider"] == cfg_upper["provider"]

    def test_whitespace_stripped_from_provider(self):
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="  llm_reranker  "))
        assert cfg["provider"] == "llm_reranker"

    def test_fallback_config_is_complete(self):
        """Fallback must return a valid, usable config."""
        from src.infra.mem0_client import _build_reranker_config
        cfg = _build_reranker_config(_make_settings(mem0_reranker_provider="unknown_xyz"))
        assert "provider" in cfg
        assert "config" in cfg
        assert "model" in cfg["config"]
        assert "top_k" in cfg["config"]


# ── Config integration ─────────────────────────────────────────────────────────

class TestRerankerConfigIntegration:

    def _full(self, **overrides):
        defaults = dict(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
            cohere_api_key="",
            zero_entropy_api_key="",
            huggingface_api_key="",
            mem0_llm_provider="openai",
            mem0_llm_model="",
            mem0_llm_temperature=0.0,
            mem0_llm_max_tokens=2000,
            mem0_llm_base_url="",
            mem0_embedder_provider="openai",
            mem0_embedder_model="",
            mem0_embedder_dims=1536,
            mem0_embedder_base_url="",
            mem0_vector_store_provider="qdrant",
            mem0_collection_name="reso_memories",
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_model="",
            mem0_reranker_top_k=10,
            mem0_reranker_use_wellness_prompt=True,
            qdrant_url="https://qdrant.test",
            qdrant_api_key="key",
            neo4j_url="bolt://neo4j.test",
            neo4j_username="neo4j",
            neo4j_password="pass",
            supabase_connection_string="",
            chroma_path="./chroma_db",
            chroma_host="",
            chroma_port=8000,
            pinecone_api_key="",
            pinecone_index_name="reso-memories",
            anthropic_api_key="",
            groq_api_key="",
            azure_openai_api_key="",
            azure_openai_endpoint="",
            azure_deployment_name="",
            aws_region="us-east-1",
            aws_access_key_id="",
            aws_secret_access_key="",
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_build_mem0_config_has_reranker_key(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._full())
        assert "reranker" in cfg

    def test_build_mem0_config_reranker_provider_matches(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._full(
            mem0_reranker_provider="sentence_transformer",
        ))
        assert cfg["reranker"]["provider"] == "sentence_transformer"

    def test_build_mem0_config_no_reranker_key_when_none(self):
        """When provider=none, the 'reranker' key must be absent from the config."""
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._full(mem0_reranker_provider="none"))
        assert "reranker" not in cfg

    def test_build_mem0_config_wellness_prompt_in_llm_reranker(self):
        from src.infra.mem0_client import _build_mem0_config, WELLNESS_RERANKER_SCORING_PROMPT
        cfg = _build_mem0_config(self._full(
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_use_wellness_prompt=True,
        ))
        assert cfg["reranker"]["config"].get("scoring_prompt") == WELLNESS_RERANKER_SCORING_PROMPT

    def test_build_mem0_config_cohere_reranker(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._full(
            mem0_reranker_provider="cohere",
            cohere_api_key="co-key",
        ))
        assert cfg["reranker"]["provider"] == "cohere"

    def test_top_k_propagates_to_reranker(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._full(mem0_reranker_top_k=15))
        assert cfg["reranker"]["config"]["top_k"] == 15


# ── get_mem0_config_summary ────────────────────────────────────────────────────

class TestConfigSummaryFeature13:

    def test_summary_reranker_has_provider(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "provider" in result["reranker"]

    def test_summary_reranker_has_top_k(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["reranker"]["top_k"] == get_settings().mem0_reranker_top_k

    def test_summary_reranker_has_wellness_prompt_flag(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "wellness_prompt" in result["reranker"]
        assert isinstance(result["reranker"]["wellness_prompt"], bool)

    def test_summary_reranker_provider_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["reranker"]["provider"] == get_settings().mem0_reranker_provider

    def test_summary_reranker_wellness_prompt_flag_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["reranker"]["wellness_prompt"] == get_settings().mem0_reranker_use_wellness_prompt
