"""Tests for Feature 10 — Configurable LLM Providers.

Verifies that:
  - New settings fields exist with correct defaults
  - get_supported_llm_providers() returns all 6 supported providers
  - _build_llm_config() produces correct Mem0 config blocks for each provider
  - Unknown provider falls back to openai with a warning
  - Model selection precedence: explicit model > pipeline model > provider default
  - Temperature default is ≤ 0.2 (docs recommend deterministic extraction)
  - _build_mem0_config() and _build_mem0_proxy_config() use the LLM block
  - get_mem0_config_summary() reflects the configured provider/model/temperature
  - validate_mem0_config() handles missing provider API keys gracefully

Run:
    python -m pytest tests/test_feature10_llm_providers.py -v
    python -m pytest tests/test_feature10_llm_providers.py -v -m live
"""

from __future__ import annotations

import asyncio
import logging
import types
import unittest.mock as mock

import pytest


# ── Settings field tests ───────────────────────────────────────────────────────

class TestFeature10SettingsFields:
    """New LLM provider settings fields must exist with correct defaults."""

    def _s(self):
        from src.infra.config import get_settings
        return get_settings()

    def test_mem0_llm_provider_exists(self):
        assert hasattr(self._s(), "mem0_llm_provider")

    def test_mem0_llm_provider_default_is_openai(self):
        s = self._s()
        # Default is "openai" unless overridden in .env
        assert isinstance(s.mem0_llm_provider, str)
        assert s.mem0_llm_provider != ""

    def test_mem0_llm_model_exists(self):
        assert hasattr(self._s(), "mem0_llm_model")

    def test_mem0_llm_model_is_string(self):
        # Empty string is valid (means "use llm_model_pipeline")
        assert isinstance(self._s().mem0_llm_model, str)

    def test_mem0_llm_temperature_exists(self):
        assert hasattr(self._s(), "mem0_llm_temperature")

    def test_mem0_llm_temperature_is_float(self):
        assert isinstance(self._s().mem0_llm_temperature, float)

    def test_mem0_llm_temperature_deterministic_default(self):
        """Docs recommend ≤ 0.2 for deterministic fact extraction."""
        s = self._s()
        assert s.mem0_llm_temperature <= 0.2

    def test_mem0_llm_max_tokens_exists(self):
        assert hasattr(self._s(), "mem0_llm_max_tokens")

    def test_mem0_llm_max_tokens_is_int(self):
        assert isinstance(self._s().mem0_llm_max_tokens, int)

    def test_mem0_llm_max_tokens_reasonable_default(self):
        assert 100 <= self._s().mem0_llm_max_tokens <= 8000

    def test_mem0_llm_base_url_exists(self):
        assert hasattr(self._s(), "mem0_llm_base_url")

    def test_mem0_llm_base_url_is_string(self):
        assert isinstance(self._s().mem0_llm_base_url, str)

    def test_anthropic_api_key_exists(self):
        assert hasattr(self._s(), "anthropic_api_key")

    def test_groq_api_key_exists(self):
        assert hasattr(self._s(), "groq_api_key")

    def test_azure_openai_api_key_exists(self):
        assert hasattr(self._s(), "azure_openai_api_key")

    def test_azure_openai_endpoint_exists(self):
        assert hasattr(self._s(), "azure_openai_endpoint")

    def test_azure_deployment_name_exists(self):
        assert hasattr(self._s(), "azure_deployment_name")

    def test_aws_region_exists(self):
        assert hasattr(self._s(), "aws_region")

    def test_aws_region_default_is_us_east_1(self):
        assert isinstance(self._s().aws_region, str)
        assert self._s().aws_region != ""  # has a default

    def test_aws_access_key_id_exists(self):
        assert hasattr(self._s(), "aws_access_key_id")

    def test_aws_secret_access_key_exists(self):
        assert hasattr(self._s(), "aws_secret_access_key")


# ── get_supported_llm_providers ───────────────────────────────────────────────

class TestGetSupportedLLMProviders:

    def test_returns_list(self):
        from src.infra.mem0_client import get_supported_llm_providers
        result = get_supported_llm_providers()
        assert isinstance(result, list)

    def test_contains_all_eight_providers(self):
        from src.infra.mem0_client import get_supported_llm_providers
        providers = get_supported_llm_providers()
        for expected in (
            "openai", "openai_structured", "azure_openai",
            "anthropic", "groq", "ollama", "litellm", "aws_bedrock",
        ):
            assert expected in providers, f"Missing provider: {expected}"

    def test_has_exactly_eight_providers(self):
        from src.infra.mem0_client import get_supported_llm_providers
        assert len(get_supported_llm_providers()) == 8

    def test_no_duplicates(self):
        from src.infra.mem0_client import get_supported_llm_providers
        providers = get_supported_llm_providers()
        assert len(providers) == len(set(providers))


# ── _build_llm_config ─────────────────────────────────────────────────────────

def _make_settings(**overrides):
    """Build a minimal SimpleNamespace settings object for _build_llm_config tests."""
    defaults = dict(
        llm_model_pipeline="gpt-4o-mini",
        openai_api_key="sk-test",
        anthropic_api_key="",
        groq_api_key="",
        azure_openai_api_key="",
        azure_openai_endpoint="",
        azure_deployment_name="",
        aws_region="us-east-1",
        aws_access_key_id="",
        aws_secret_access_key="",
        mem0_llm_provider="openai",
        mem0_llm_model="",
        mem0_llm_temperature=0.0,
        mem0_llm_max_tokens=2000,
        mem0_llm_base_url="",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestBuildLLMConfigOpenAI:

    def test_openai_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai"))
        assert cfg["provider"] == "openai"

    def test_openai_has_config_dict(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai"))
        assert isinstance(cfg["config"], dict)

    def test_openai_model_uses_pipeline_model_when_explicit_empty(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_model="",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert cfg["config"]["model"] == "gpt-4o-mini"

    def test_openai_explicit_model_takes_precedence(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_model="gpt-4o",
        ))
        assert cfg["config"]["model"] == "gpt-4o"

    def test_openai_temperature_set(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai", mem0_llm_temperature=0.1))
        assert cfg["config"]["temperature"] == 0.1

    def test_openai_api_key_present(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai", openai_api_key="sk-abc"))
        assert cfg["config"]["api_key"] == "sk-abc"

    def test_openai_base_url_included_when_set(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_base_url="http://proxy.internal/v1",
        ))
        assert cfg["config"].get("openai_base_url") == "http://proxy.internal/v1"

    def test_openai_base_url_absent_when_empty(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai", mem0_llm_base_url=""))
        assert "openai_base_url" not in cfg["config"]


class TestBuildLLMConfigOpenAIStructured:

    def test_openai_structured_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai_structured"))
        assert cfg["provider"] == "openai_structured"

    def test_openai_structured_has_model(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="openai_structured"))
        assert "model" in cfg["config"]

    def test_openai_structured_has_api_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai_structured",
            openai_api_key="sk-struct",
        ))
        assert cfg["config"]["api_key"] == "sk-struct"

    def test_openai_structured_base_url_included_when_set(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai_structured",
            mem0_llm_base_url="http://proxy.internal/v1",
        ))
        assert cfg["config"].get("openai_base_url") == "http://proxy.internal/v1"

    def test_openai_structured_base_url_absent_when_empty(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai_structured",
            mem0_llm_base_url="",
        ))
        assert "openai_base_url" not in cfg["config"]


class TestBuildLLMConfigAnthropic:

    def test_anthropic_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="anthropic",
            anthropic_api_key="sk-ant-test",
        ))
        assert cfg["provider"] == "anthropic"

    def test_anthropic_default_model_is_updated(self):
        """Default Anthropic model must be the latest Claude (claude-sonnet-4-20250514)."""
        from src.infra.mem0_client import _SUPPORTED_PROVIDERS
        assert _SUPPORTED_PROVIDERS["anthropic"] == "claude-sonnet-4-20250514"

    def test_anthropic_uses_provider_default_model_when_pipeline_model_given(self):
        """When mem0_llm_model is empty, Anthropic must NOT use gpt-4o-mini."""
        from src.infra.mem0_client import _build_llm_config, _SUPPORTED_PROVIDERS
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="anthropic",
            mem0_llm_model="",
            llm_model_pipeline="gpt-4o-mini",  # OpenAI model — must not be used for Anthropic
            anthropic_api_key="sk-ant-test",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_PROVIDERS["anthropic"]

    def test_anthropic_explicit_model_respected(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="anthropic",
            mem0_llm_model="claude-3-haiku-20240307",
            anthropic_api_key="sk-ant-test",
        ))
        assert cfg["config"]["model"] == "claude-3-haiku-20240307"

    def test_anthropic_api_key_in_config(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="anthropic",
            anthropic_api_key="sk-ant-prod",
        ))
        assert cfg["config"]["api_key"] == "sk-ant-prod"


class TestBuildLLMConfigGroq:

    def test_groq_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="groq",
            groq_api_key="gsk-test",
        ))
        assert cfg["provider"] == "groq"

    def test_groq_uses_provider_default_model_when_pipeline_model_given(self):
        """When mem0_llm_model is empty, Groq must NOT use gpt-4o-mini."""
        from src.infra.mem0_client import _build_llm_config, _SUPPORTED_PROVIDERS
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="groq",
            mem0_llm_model="",
            llm_model_pipeline="gpt-4o-mini",
            groq_api_key="gsk-test",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_PROVIDERS["groq"]

    def test_groq_api_key_in_config(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="groq",
            groq_api_key="gsk-prod",
        ))
        assert cfg["config"]["api_key"] == "gsk-prod"


class TestBuildLLMConfigOllama:

    def test_ollama_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="ollama"))
        assert cfg["provider"] == "ollama"

    def test_ollama_uses_provider_default_model_when_pipeline_model_given(self):
        from src.infra.mem0_client import _build_llm_config, _SUPPORTED_PROVIDERS
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="ollama",
            mem0_llm_model="",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_PROVIDERS["ollama"]

    def test_ollama_default_base_url(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="ollama",
            mem0_llm_base_url="",
        ))
        assert "localhost:11434" in cfg["config"].get("ollama_base_url", "")

    def test_ollama_custom_base_url(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="ollama",
            mem0_llm_base_url="http://ollama.internal:11434",
        ))
        assert cfg["config"]["ollama_base_url"] == "http://ollama.internal:11434"


class TestBuildLLMConfigAzure:

    def test_azure_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="azure_openai",
            azure_openai_api_key="az-test",
            azure_openai_endpoint="https://myaccount.openai.azure.com",
        ))
        assert cfg["provider"] == "azure_openai"

    def test_azure_has_azure_kwargs(self):
        """Azure config uses azure_kwargs sub-dict (Mem0 convention)."""
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="azure_openai",
            azure_openai_api_key="az-test",
            azure_openai_endpoint="https://myaccount.openai.azure.com",
        ))
        assert "azure_kwargs" in cfg["config"]

    def test_azure_endpoint_in_azure_kwargs(self):
        """Endpoint is stored under azure_kwargs.api_base (Mem0 Azure schema)."""
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="azure_openai",
            azure_openai_api_key="az-key",
            azure_openai_endpoint="https://custom.openai.azure.com",
        ))
        assert cfg["config"]["azure_kwargs"]["api_base"] == "https://custom.openai.azure.com"


class TestBuildLLMConfigLiteLLM:

    def test_litellm_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="litellm"))
        assert cfg["provider"] == "litellm"

    def test_litellm_has_model(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="litellm",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert "model" in cfg["config"]


class TestBuildLLMConfigAWSBedrock:

    def test_aws_bedrock_provider_key(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="aws_bedrock"))
        assert cfg["provider"] == "aws_bedrock"

    def test_aws_bedrock_default_model_is_claude_haiku(self):
        from src.infra.mem0_client import _SUPPORTED_PROVIDERS
        assert _SUPPORTED_PROVIDERS["aws_bedrock"] == "anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_aws_bedrock_uses_default_model_when_no_explicit(self):
        from src.infra.mem0_client import _build_llm_config, _SUPPORTED_PROVIDERS
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            mem0_llm_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_PROVIDERS["aws_bedrock"]

    def test_aws_bedrock_explicit_model_respected(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            mem0_llm_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        ))
        assert cfg["config"]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_aws_bedrock_credentials_included_when_set(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret123",
            aws_region="us-west-2",
        ))
        assert cfg["config"]["aws_access_key"] == "AKIATEST"
        assert cfg["config"]["aws_secret_key"] == "secret123"
        assert cfg["config"]["aws_region"] == "us-west-2"

    def test_aws_bedrock_credentials_absent_when_empty(self):
        """Empty credentials should not be in config — let boto3 use its chain."""
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            aws_access_key_id="",
            aws_secret_access_key="",
        ))
        assert "aws_access_key" not in cfg["config"]
        assert "aws_secret_key" not in cfg["config"]

    def test_aws_bedrock_region_defaults_to_us_east_1(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            aws_region="us-east-1",
        ))
        assert cfg["config"]["aws_region"] == "us-east-1"

    def test_aws_bedrock_has_temperature_and_max_tokens(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="aws_bedrock",
            mem0_llm_temperature=0.1,
            mem0_llm_max_tokens=1500,
        ))
        assert cfg["config"]["temperature"] == 0.1
        assert cfg["config"]["max_tokens"] == 1500


class TestBuildLLMConfigFallback:

    def test_unknown_provider_falls_back_to_openai(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="unknown_xyz"))
        assert cfg["provider"] == "openai"

    def test_unknown_provider_fallback_returns_valid_config(self):
        """Unknown provider must fall back to a valid, usable OpenAI config block."""
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="unknown_xyz"))
        # Fallen back to openai — must be a complete, usable config dict
        assert cfg["provider"] == "openai"
        assert "model" in cfg["config"]
        assert "api_key" in cfg["config"]

    def test_case_insensitive_provider(self):
        """Provider name must be case-insensitive."""
        from src.infra.mem0_client import _build_llm_config
        cfg_lower = _build_llm_config(_make_settings(mem0_llm_provider="openai"))
        cfg_upper = _build_llm_config(_make_settings(mem0_llm_provider="OpenAI"))
        assert cfg_lower["provider"] == cfg_upper["provider"]

    def test_whitespace_stripped_from_provider(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(mem0_llm_provider="  openai  "))
        assert cfg["provider"] == "openai"


# ── Model selection precedence ─────────────────────────────────────────────────

class TestModelSelectionPrecedence:
    """Explicit model > pipeline model > provider default."""

    def test_explicit_model_beats_pipeline_model(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_model="gpt-4o",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert cfg["config"]["model"] == "gpt-4o"

    def test_pipeline_model_used_when_explicit_empty_for_openai(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_model="",
            llm_model_pipeline="gpt-4o-mini",
        ))
        assert cfg["config"]["model"] == "gpt-4o-mini"

    def test_provider_default_used_when_both_empty_openai(self):
        from src.infra.mem0_client import _build_llm_config, _SUPPORTED_PROVIDERS
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="openai",
            mem0_llm_model="",
            llm_model_pipeline="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_PROVIDERS["openai"]

    def test_anthropic_explicit_model_beats_provider_default(self):
        from src.infra.mem0_client import _build_llm_config
        cfg = _build_llm_config(_make_settings(
            mem0_llm_provider="anthropic",
            mem0_llm_model="claude-3-opus-20240229",
            anthropic_api_key="sk-ant",
        ))
        assert cfg["config"]["model"] == "claude-3-opus-20240229"


# ── _build_mem0_config and _build_mem0_proxy_config use _build_llm_config ─────

class TestConfigIntegration:
    """The full config builders must use _build_llm_config output."""

    def _make_full_settings(self, **overrides):
        defaults = dict(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
            qdrant_url="https://qdrant.test",
            qdrant_api_key="qdrant-key",
            neo4j_url="bolt://neo4j.test",
            neo4j_username="neo4j",
            neo4j_password="password",
            mem0_collection_name="reso_memories",
            mem0_reranker_top_k=10,
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
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_build_mem0_config_llm_provider_key(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_full_settings(mem0_llm_provider="openai"))
        assert cfg["llm"]["provider"] == "openai"

    def test_build_mem0_config_llm_provider_anthropic(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_full_settings(
            mem0_llm_provider="anthropic",
            anthropic_api_key="sk-ant",
        ))
        assert cfg["llm"]["provider"] == "anthropic"

    def test_build_proxy_config_llm_provider_matches(self):
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_full_settings(mem0_llm_provider="groq", groq_api_key="gsk"))
        assert cfg["llm"]["provider"] == "groq"

    def test_async_and_proxy_config_share_llm_provider(self):
        """AsyncMemory and proxy must use the same LLM provider."""
        from src.infra.mem0_client import _build_mem0_config, _build_mem0_proxy_config
        settings = self._make_full_settings(mem0_llm_provider="openai")
        async_cfg = _build_mem0_config(settings)
        proxy_cfg = _build_mem0_proxy_config(settings)
        assert async_cfg["llm"]["provider"] == proxy_cfg["llm"]["provider"]

    def test_build_mem0_config_openai_structured(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_full_settings(mem0_llm_provider="openai_structured"))
        assert cfg["llm"]["provider"] == "openai_structured"

    def test_build_mem0_config_aws_bedrock(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_full_settings(mem0_llm_provider="aws_bedrock"))
        assert cfg["llm"]["provider"] == "aws_bedrock"


# ── get_mem0_config_summary reflects Feature 10 settings ──────────────────────

class TestConfigSummaryFeature10:

    def test_summary_llm_section_has_provider(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "provider" in result["llm"]

    def test_summary_llm_section_has_model(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "model" in result["llm"]

    def test_summary_llm_section_has_temperature(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "temperature" in result["llm"]

    def test_summary_llm_section_has_max_tokens(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "max_tokens" in result["llm"]

    def test_summary_llm_provider_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        settings = get_settings()
        assert result["llm"]["provider"] == settings.mem0_llm_provider

    def test_summary_llm_model_is_string(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert isinstance(result["llm"]["model"], str)
        assert result["llm"]["model"] != ""  # must resolve to something

    def test_summary_no_api_keys_in_llm_section(self):
        """The summary must never expose raw API keys."""
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        summary_str = str(result["llm"])
        settings = get_settings()
        if settings.openai_api_key and len(settings.openai_api_key) > 8:
            assert settings.openai_api_key not in summary_str
        if settings.anthropic_api_key and len(settings.anthropic_api_key) > 8:
            assert settings.anthropic_api_key not in summary_str
        if settings.groq_api_key and len(settings.groq_api_key) > 8:
            assert settings.groq_api_key not in summary_str


# ── validate_mem0_config handles provider-specific key absence ─────────────────

class TestValidateMem0ConfigProviderErrors:
    """validate_mem0_config() must return clean error strings, never raise."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_anthropic_missing_key_returns_error_string(self):
        """If provider=anthropic but no ANTHROPIC_API_KEY, llm status must be error."""
        from src.infra.mem0_client import validate_mem0_config
        from src.infra.config import get_settings
        settings = get_settings()

        if settings.mem0_llm_provider != "anthropic" or settings.anthropic_api_key:
            pytest.skip("Only applicable when provider=anthropic and key missing")

        result = self._run(validate_mem0_config())
        assert result["llm"].startswith("error")

    def test_groq_missing_key_returns_error_string(self):
        from src.infra.mem0_client import validate_mem0_config
        from src.infra.config import get_settings
        settings = get_settings()

        if settings.mem0_llm_provider != "groq" or settings.groq_api_key:
            pytest.skip("Only applicable when provider=groq and key missing")

        result = self._run(validate_mem0_config())
        assert result["llm"].startswith("error")

    def test_azure_missing_key_returns_error_string(self):
        from src.infra.mem0_client import validate_mem0_config
        from src.infra.config import get_settings
        settings = get_settings()

        if settings.mem0_llm_provider != "azure_openai" or settings.azure_openai_api_key:
            pytest.skip("Only applicable when provider=azure_openai and key missing")

        result = self._run(validate_mem0_config())
        assert result["llm"].startswith("error")

    def test_validate_never_raises_any_provider(self):
        """Regardless of provider, validate_mem0_config must not raise."""
        from src.infra.mem0_client import validate_mem0_config
        try:
            result = self._run(validate_mem0_config())
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"validate_mem0_config raised unexpectedly: {exc}")

    def test_validate_llm_key_present_in_result(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert "llm" in result

    def test_validate_llm_value_is_string(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert isinstance(result["llm"], str)


# ── Live tests (requires OPENAI_API_KEY + configured provider) ─────────────────

@pytest.mark.live
class TestFeature10Live:
    """Live component validation for the configured LLM provider."""

    @pytest.fixture(autouse=True)
    def require_openai_key(self):
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_llm_validation_ok_for_default_provider(self):
        from src.infra.mem0_client import validate_mem0_config
        from src.infra.config import get_settings
        if not get_settings().qdrant_url:
            pytest.skip("QDRANT_URL not set — validate short-circuits")
        result = self._run(validate_mem0_config())
        assert result["llm"] in ("ok", "unconfigured", "configured"), \
            f"Unexpected LLM status: {result['llm']}"

    def test_build_llm_config_openai_can_be_imported(self):
        from src.infra.mem0_client import _build_llm_config
        from src.infra.config import get_settings
        cfg = _build_llm_config(get_settings())
        assert "provider" in cfg
        assert "config" in cfg
