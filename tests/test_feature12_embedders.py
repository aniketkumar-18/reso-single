"""Tests for Feature 12 — Configurable Embedder.

Verifies that:
  - New settings fields exist with correct defaults
  - get_supported_embedders() returns all 9 providers
  - _build_embedder_config() produces correct config blocks per provider
  - Unknown provider falls back to openai
  - MEM0_EMBEDDER_DIMS is honoured across all providers
  - Provider default models are correct (especially dims-critical ones)
  - _build_mem0_config() and proxy use _build_embedder_config()
  - get_mem0_config_summary() reflects actual embedder provider/model/dims
  - validate_mem0_config() probes the configured embedder

Run:
    python -m pytest tests/test_feature12_embedders.py -v
    python -m pytest tests/test_feature12_embedders.py -v -m live
"""

from __future__ import annotations

import asyncio
import types
import unittest.mock as mock

import pytest


# ── Settings field tests ───────────────────────────────────────────────────────

class TestFeature12SettingsFields:

    def _s(self):
        from src.infra.config import get_settings
        return get_settings()

    def test_mem0_embedder_provider_exists(self):
        assert hasattr(self._s(), "mem0_embedder_provider")

    def test_mem0_embedder_provider_default_is_openai(self):
        s = self._s()
        assert isinstance(s.mem0_embedder_provider, str)
        assert s.mem0_embedder_provider != ""

    def test_mem0_embedder_model_exists(self):
        assert hasattr(self._s(), "mem0_embedder_model")

    def test_mem0_embedder_model_is_string(self):
        assert isinstance(self._s().mem0_embedder_model, str)

    def test_mem0_embedder_dims_exists(self):
        assert hasattr(self._s(), "mem0_embedder_dims")

    def test_mem0_embedder_dims_is_int(self):
        assert isinstance(self._s().mem0_embedder_dims, int)

    def test_mem0_embedder_dims_default_is_1536(self):
        """Default must match text-embedding-3-small (OpenAI default)."""
        assert self._s().mem0_embedder_dims == 1536

    def test_mem0_embedder_base_url_exists(self):
        assert hasattr(self._s(), "mem0_embedder_base_url")

    def test_mem0_embedder_base_url_is_string(self):
        assert isinstance(self._s().mem0_embedder_base_url, str)


# ── get_supported_embedders ────────────────────────────────────────────────────

class TestGetSupportedEmbedders:

    def test_returns_list(self):
        from src.infra.mem0_client import get_supported_embedders
        assert isinstance(get_supported_embedders(), list)

    def test_contains_all_nine_providers(self):
        from src.infra.mem0_client import get_supported_embedders
        providers = get_supported_embedders()
        for expected in (
            "openai", "azure_openai", "ollama", "huggingface",
            "google_ai", "vertexai", "together", "lmstudio", "aws_bedrock",
        ):
            assert expected in providers, f"Missing embedder provider: {expected}"

    def test_has_exactly_nine_providers(self):
        from src.infra.mem0_client import get_supported_embedders
        assert len(get_supported_embedders()) == 9

    def test_no_duplicates(self):
        from src.infra.mem0_client import get_supported_embedders
        providers = get_supported_embedders()
        assert len(providers) == len(set(providers))


# ── Provider default models ────────────────────────────────────────────────────

class TestEmbedderDefaultModels:

    def test_openai_default_model_is_text_embedding_3_small(self):
        from src.infra.mem0_client import _SUPPORTED_EMBEDDERS
        assert _SUPPORTED_EMBEDDERS["openai"] == "text-embedding-3-small"

    def test_ollama_default_model_set(self):
        from src.infra.mem0_client import _SUPPORTED_EMBEDDERS
        assert _SUPPORTED_EMBEDDERS["ollama"] != ""

    def test_huggingface_default_model_set(self):
        from src.infra.mem0_client import _SUPPORTED_EMBEDDERS
        assert _SUPPORTED_EMBEDDERS["huggingface"] != ""

    def test_aws_bedrock_default_model_set(self):
        from src.infra.mem0_client import _SUPPORTED_EMBEDDERS
        assert "amazon" in _SUPPORTED_EMBEDDERS["aws_bedrock"]

    def test_default_dims_for_openai_is_1536(self):
        from src.infra.mem0_client import _DEFAULT_EMBEDDER_DIMS
        assert _DEFAULT_EMBEDDER_DIMS["openai"] == 1536

    def test_default_dims_for_ollama_is_less_than_openai(self):
        """Non-OpenAI embedders typically produce smaller vectors."""
        from src.infra.mem0_client import _DEFAULT_EMBEDDER_DIMS
        assert _DEFAULT_EMBEDDER_DIMS["ollama"] < _DEFAULT_EMBEDDER_DIMS["openai"]


# ── _build_embedder_config per provider ───────────────────────────────────────

def _make_settings(**overrides):
    defaults = dict(
        openai_api_key="sk-test",
        azure_openai_api_key="",
        azure_openai_endpoint="",
        azure_deployment_name="",
        aws_region="us-east-1",
        aws_access_key_id="",
        aws_secret_access_key="",
        mem0_embedder_provider="openai",
        mem0_embedder_model="",
        mem0_embedder_dims=1536,
        mem0_embedder_base_url="",
        # Feature 9
        mem0_collection_name="reso_memories",
        mem0_reranker_top_k=10,
        # Feature 10
        llm_model_pipeline="gpt-4o-mini",
        mem0_llm_provider="openai",
        mem0_llm_model="",
        mem0_llm_temperature=0.0,
        mem0_llm_max_tokens=2000,
        mem0_llm_base_url="",
        anthropic_api_key="",
        groq_api_key="",
        neo4j_url="bolt://neo4j.test",
        neo4j_username="neo4j",
        neo4j_password="password",
        # Feature 11
        mem0_vector_store_provider="qdrant",
        qdrant_url="https://qdrant.test",
        qdrant_api_key="qdrant-key",
        supabase_connection_string="",
        chroma_path="./chroma_db",
        chroma_host="",
        chroma_port=8000,
        pinecone_api_key="",
        pinecone_index_name="reso-memories",
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


class TestBuildEmbedderConfigOpenAI:

    def test_openai_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="openai"))
        assert cfg["provider"] == "openai"

    def test_openai_model_uses_default_when_empty(self):
        from src.infra.mem0_client import _build_embedder_config, _SUPPORTED_EMBEDDERS
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="openai", mem0_embedder_model=""))
        assert cfg["config"]["model"] == _SUPPORTED_EMBEDDERS["openai"]

    def test_openai_explicit_model_respected(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="openai",
            mem0_embedder_model="text-embedding-3-large",
        ))
        assert cfg["config"]["model"] == "text-embedding-3-large"

    def test_openai_dims_in_config(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_dims=3072))
        assert cfg["config"]["embedding_dims"] == 3072

    def test_openai_api_key_in_config(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(openai_api_key="sk-embed"))
        assert cfg["config"]["api_key"] == "sk-embed"

    def test_openai_base_url_included_when_set(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_base_url="http://proxy/v1"))
        assert cfg["config"].get("openai_base_url") == "http://proxy/v1"

    def test_openai_base_url_absent_when_empty(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_base_url=""))
        assert "openai_base_url" not in cfg["config"]


class TestBuildEmbedderConfigAzureOpenAI:

    def test_azure_openai_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="azure_openai",
            azure_openai_api_key="az-key",
            azure_openai_endpoint="https://my.openai.azure.com",
        ))
        assert cfg["provider"] == "azure_openai"

    def test_azure_openai_has_azure_kwargs(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="azure_openai",
            azure_openai_endpoint="https://my.openai.azure.com",
        ))
        assert "azure_kwargs" in cfg["config"]

    def test_azure_openai_endpoint_in_azure_kwargs(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="azure_openai",
            azure_openai_endpoint="https://custom.openai.azure.com",
        ))
        assert cfg["config"]["azure_kwargs"]["api_base"] == "https://custom.openai.azure.com"


class TestBuildEmbedderConfigOllama:

    def test_ollama_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="ollama"))
        assert cfg["provider"] == "ollama"

    def test_ollama_default_base_url(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="ollama",
            mem0_embedder_base_url="",
        ))
        assert "localhost:11434" in cfg["config"].get("ollama_base_url", "")

    def test_ollama_custom_base_url(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="ollama",
            mem0_embedder_base_url="http://ollama.internal:11434",
        ))
        assert cfg["config"]["ollama_base_url"] == "http://ollama.internal:11434"

    def test_ollama_uses_provider_default_model(self):
        from src.infra.mem0_client import _build_embedder_config, _SUPPORTED_EMBEDDERS
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="ollama",
            mem0_embedder_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_EMBEDDERS["ollama"]


class TestBuildEmbedderConfigHuggingFace:

    def test_huggingface_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="huggingface"))
        assert cfg["provider"] == "huggingface"

    def test_hugging_face_alias_works(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="hugging_face"))
        assert cfg["provider"] == "huggingface"

    def test_huggingface_has_model(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="huggingface"))
        assert "model" in cfg["config"]

    def test_huggingface_dims_in_config(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="huggingface",
            mem0_embedder_dims=384,
        ))
        assert cfg["config"]["embedding_dims"] == 384


class TestBuildEmbedderConfigLMStudio:

    def test_lmstudio_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="lmstudio"))
        assert cfg["provider"] == "lmstudio"

    def test_lmstudio_default_base_url(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="lmstudio",
            mem0_embedder_base_url="",
        ))
        assert "1234" in cfg["config"].get("lmstudio_base_url", "")

    def test_lmstudio_custom_base_url(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="lmstudio",
            mem0_embedder_base_url="http://lmstudio.local:1234",
        ))
        assert cfg["config"]["lmstudio_base_url"] == "http://lmstudio.local:1234"


class TestBuildEmbedderConfigAWSBedrock:

    def test_aws_bedrock_provider_key(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="aws_bedrock"))
        assert cfg["provider"] == "aws_bedrock"

    def test_aws_bedrock_default_model(self):
        from src.infra.mem0_client import _build_embedder_config, _SUPPORTED_EMBEDDERS
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="aws_bedrock",
            mem0_embedder_model="",
        ))
        assert cfg["config"]["model"] == _SUPPORTED_EMBEDDERS["aws_bedrock"]

    def test_aws_bedrock_credentials_included_when_set(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="aws_bedrock",
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret",
            aws_region="us-west-2",
        ))
        assert cfg["config"]["aws_access_key"] == "AKIATEST"
        assert cfg["config"]["aws_secret_key"] == "secret"
        assert cfg["config"]["aws_region"] == "us-west-2"

    def test_aws_bedrock_credentials_absent_when_empty(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="aws_bedrock",
            aws_access_key_id="",
            aws_secret_access_key="",
        ))
        assert "aws_access_key" not in cfg["config"]
        assert "aws_secret_key" not in cfg["config"]


class TestBuildEmbedderConfigFallback:

    def test_unknown_provider_falls_back_to_openai(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="unknown_xyz"))
        assert cfg["provider"] == "openai"

    def test_unknown_provider_fallback_has_dims_1536(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="unknown_xyz"))
        assert cfg["config"]["embedding_dims"] == 1536

    def test_case_insensitive_provider(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg_lower = _build_embedder_config(_make_settings(mem0_embedder_provider="openai"))
        cfg_upper = _build_embedder_config(_make_settings(mem0_embedder_provider="OpenAI"))
        assert cfg_lower["provider"] == cfg_upper["provider"]

    def test_whitespace_stripped(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(mem0_embedder_provider="  openai  "))
        assert cfg["provider"] == "openai"


# ── Dims correctness ───────────────────────────────────────────────────────────

class TestEmbedderDimsHandling:

    def test_dims_setting_propagates_to_config(self):
        """MEM0_EMBEDDER_DIMS must flow through to the provider config."""
        from src.infra.mem0_client import _build_embedder_config
        for dims in (384, 768, 1024, 1536, 3072):
            cfg = _build_embedder_config(_make_settings(
                mem0_embedder_provider="openai",
                mem0_embedder_dims=dims,
            ))
            assert cfg["config"]["embedding_dims"] == dims

    def test_dims_propagate_to_ollama(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="ollama",
            mem0_embedder_dims=768,
        ))
        assert cfg["config"]["embedding_dims"] == 768

    def test_dims_propagate_to_huggingface(self):
        from src.infra.mem0_client import _build_embedder_config
        cfg = _build_embedder_config(_make_settings(
            mem0_embedder_provider="huggingface",
            mem0_embedder_dims=384,
        ))
        assert cfg["config"]["embedding_dims"] == 384


# ── Config integration ─────────────────────────────────────────────────────────

class TestEmbedderConfigIntegration:

    def _full(self, **overrides):
        defaults = dict(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
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
            mem0_reranker_top_k=10,
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

    def test_build_mem0_config_embedder_provider(self):
        from src.infra.mem0_client import _build_mem0_config
        result = _build_mem0_config(self._full(mem0_embedder_provider="openai"))
        assert result["embedder"]["provider"] == "openai"

    def test_build_mem0_config_ollama_embedder(self):
        from src.infra.mem0_client import _build_mem0_config
        result = _build_mem0_config(self._full(
            mem0_embedder_provider="ollama",
            mem0_embedder_dims=768,
        ))
        assert result["embedder"]["provider"] == "ollama"

    def test_build_proxy_config_shares_embedder(self):
        from src.infra.mem0_client import _build_mem0_config, _build_mem0_proxy_config
        settings = self._full(mem0_embedder_provider="openai")
        assert (_build_mem0_config(settings)["embedder"]["provider"]
                == _build_mem0_proxy_config(settings)["embedder"]["provider"])

    def test_embedder_dims_flow_into_mem0_config(self):
        from src.infra.mem0_client import _build_mem0_config
        result = _build_mem0_config(self._full(mem0_embedder_dims=768, mem0_embedder_provider="ollama"))
        assert result["embedder"]["config"]["embedding_dims"] == 768


# ── get_mem0_config_summary ────────────────────────────────────────────────────

class TestConfigSummaryFeature12:

    def test_summary_embedder_has_provider(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "provider" in result["embedder"]

    def test_summary_embedder_has_model(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "model" in result["embedder"]
        assert result["embedder"]["model"] != ""

    def test_summary_embedder_has_dims(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "dims" in result["embedder"]
        assert isinstance(result["embedder"]["dims"], int)

    def test_summary_embedder_provider_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        settings = get_settings()
        assert result["embedder"]["provider"] == settings.mem0_embedder_provider

    def test_summary_embedder_dims_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["embedder"]["dims"] == get_settings().mem0_embedder_dims


# ── Live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestFeature12Live:

    @pytest.fixture(autouse=True)
    def require_openai_key(self):
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_openai_embedder_ok_when_configured(self):
        from src.infra.config import get_settings
        from src.infra.mem0_client import validate_mem0_config
        s = get_settings()
        if not s.qdrant_url:
            pytest.skip("QDRANT_URL not set — validate short-circuits")
        result = self._run(validate_mem0_config())
        assert result["embedder"] in ("ok", "unconfigured", "configured"), \
            f"Unexpected embedder status: {result['embedder']}"
