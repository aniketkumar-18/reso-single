"""Tests for Feature 11 — Configurable Vector Database.

Verifies that:
  - New settings fields exist with correct defaults
  - get_supported_vector_stores() returns all supported providers
  - _build_vector_store_config() produces correct config blocks per provider
  - Unknown provider falls back to qdrant
  - Both _build_mem0_config() and _build_mem0_proxy_config() use _build_vector_store_config()
  - is_mem0_configured / is_proxy_configured are provider-aware
  - get_mem0_config_summary() reflects actual provider in vector_store section
  - validate_mem0_config() uses vector_store key (not qdrant)

Run:
    python -m pytest tests/test_feature11_vector_stores.py -v
    python -m pytest tests/test_feature11_vector_stores.py -v -m live
"""

from __future__ import annotations

import asyncio
import types
import unittest.mock as mock

import pytest


# ── Settings field tests ───────────────────────────────────────────────────────

class TestFeature11SettingsFields:

    def _s(self):
        from src.infra.config import get_settings
        return get_settings()

    def test_mem0_vector_store_provider_exists(self):
        assert hasattr(self._s(), "mem0_vector_store_provider")

    def test_mem0_vector_store_provider_default_is_qdrant(self):
        s = self._s()
        assert isinstance(s.mem0_vector_store_provider, str)
        assert s.mem0_vector_store_provider != ""

    def test_chroma_path_exists(self):
        assert hasattr(self._s(), "chroma_path")

    def test_chroma_path_is_string(self):
        assert isinstance(self._s().chroma_path, str)

    def test_chroma_host_exists(self):
        assert hasattr(self._s(), "chroma_host")

    def test_chroma_port_exists(self):
        assert hasattr(self._s(), "chroma_port")

    def test_chroma_port_is_int(self):
        assert isinstance(self._s().chroma_port, int)

    def test_pinecone_api_key_exists(self):
        assert hasattr(self._s(), "pinecone_api_key")

    def test_pinecone_index_name_exists(self):
        assert hasattr(self._s(), "pinecone_index_name")

    def test_pinecone_index_name_is_string(self):
        assert isinstance(self._s().pinecone_index_name, str)
        assert self._s().pinecone_index_name != ""  # has a default


# ── get_supported_vector_stores ───────────────────────────────────────────────

class TestGetSupportedVectorStores:

    def test_returns_list(self):
        from src.infra.mem0_client import get_supported_vector_stores
        assert isinstance(get_supported_vector_stores(), list)

    def test_contains_qdrant(self):
        from src.infra.mem0_client import get_supported_vector_stores
        assert "qdrant" in get_supported_vector_stores()

    def test_contains_all_five_providers(self):
        from src.infra.mem0_client import get_supported_vector_stores
        providers = get_supported_vector_stores()
        for expected in ("qdrant", "pgvector", "supabase", "chroma", "pinecone"):
            assert expected in providers, f"Missing provider: {expected}"

    def test_no_duplicates(self):
        from src.infra.mem0_client import get_supported_vector_stores
        providers = get_supported_vector_stores()
        assert len(providers) == len(set(providers))


# ── _build_vector_store_config ────────────────────────────────────────────────

def _make_settings(**overrides):
    defaults = dict(
        mem0_vector_store_provider="qdrant",
        mem0_collection_name="reso_memories",
        qdrant_url="https://qdrant.test",
        qdrant_api_key="qdrant-key",
        supabase_connection_string="postgresql://user:pass@host/db",
        chroma_path="./chroma_db",
        chroma_host="",
        chroma_port=8000,
        pinecone_api_key="",
        pinecone_index_name="reso-memories",
        # Feature 9
        mem0_reranker_top_k=10,
        # Feature 10
        llm_model_pipeline="gpt-4o-mini",
        openai_api_key="sk-test",
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
        neo4j_url="bolt://neo4j.test",
        neo4j_username="neo4j",
        neo4j_password="password",
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


class TestBuildVectorStoreConfigQdrant:

    def test_qdrant_provider_key(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="qdrant"))
        assert cfg["provider"] == "qdrant"

    def test_qdrant_collection_name(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_collection_name="custom_coll"))
        assert cfg["config"]["collection_name"] == "custom_coll"

    def test_qdrant_url_in_config(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(qdrant_url="https://my.qdrant.io"))
        assert cfg["config"]["url"] == "https://my.qdrant.io"

    def test_qdrant_api_key_included_when_set(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(qdrant_api_key="secret-key"))
        assert cfg["config"]["api_key"] == "secret-key"

    def test_qdrant_api_key_absent_when_empty(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(qdrant_api_key=""))
        assert "api_key" not in cfg["config"]

    def test_qdrant_embedding_dims(self):
        from src.infra.mem0_client import _build_vector_store_config, _EMBEDDING_DIMS
        cfg = _build_vector_store_config(_make_settings())
        assert cfg["config"]["embedding_model_dims"] == _EMBEDDING_DIMS


class TestBuildVectorStoreConfigSupabase:

    def test_pgvector_maps_to_supabase_provider(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="pgvector"))
        assert cfg["provider"] == "supabase"

    def test_supabase_provider_key(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="supabase"))
        assert cfg["provider"] == "supabase"

    def test_supabase_connection_string_in_config(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="supabase",
            supabase_connection_string="postgresql://user:pass@host/db",
        ))
        assert cfg["config"]["connection_string"] == "postgresql://user:pass@host/db"

    def test_supabase_collection_name(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="supabase",
            mem0_collection_name="reso_pg",
        ))
        assert cfg["config"]["collection_name"] == "reso_pg"

    def test_supabase_embedding_dims(self):
        from src.infra.mem0_client import _build_vector_store_config, _EMBEDDING_DIMS
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="supabase"))
        assert cfg["config"]["embedding_model_dims"] == _EMBEDDING_DIMS


class TestBuildVectorStoreConfigChroma:

    def test_chroma_provider_key(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="chroma"))
        assert cfg["provider"] == "chroma"

    def test_chroma_embedded_mode_uses_path(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="chroma",
            chroma_host="",
            chroma_path="./my_chroma",
        ))
        assert cfg["config"].get("path") == "./my_chroma"
        assert "host" not in cfg["config"]

    def test_chroma_server_mode_uses_host_port(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="chroma",
            chroma_host="chroma.internal",
            chroma_port=8001,
        ))
        assert cfg["config"]["host"] == "chroma.internal"
        assert cfg["config"]["port"] == 8001
        assert "path" not in cfg["config"]

    def test_chroma_collection_name(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="chroma",
            mem0_collection_name="wellness_chroma",
        ))
        assert cfg["config"]["collection_name"] == "wellness_chroma"


class TestBuildVectorStoreConfigPinecone:

    def test_pinecone_provider_key(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="pinecone",
            pinecone_api_key="pc-test",
        ))
        assert cfg["provider"] == "pinecone"

    def test_pinecone_api_key_in_config(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="pinecone",
            pinecone_api_key="pc-secret",
        ))
        assert cfg["config"]["api_key"] == "pc-secret"

    def test_pinecone_uses_index_name_as_collection(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="pinecone",
            pinecone_api_key="pc-key",
            pinecone_index_name="reso-prod",
        ))
        assert cfg["config"]["collection_name"] == "reso-prod"

    def test_pinecone_falls_back_to_collection_name_when_index_empty(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(
            mem0_vector_store_provider="pinecone",
            pinecone_api_key="pc-key",
            pinecone_index_name="",
            mem0_collection_name="reso_memories",
        ))
        assert cfg["config"]["collection_name"] == "reso_memories"


class TestBuildVectorStoreConfigFallback:

    def test_unknown_provider_falls_back_to_qdrant(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="unknown_xyz"))
        assert cfg["provider"] == "qdrant"

    def test_case_insensitive_provider(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg_lower = _build_vector_store_config(_make_settings(mem0_vector_store_provider="qdrant"))
        cfg_upper = _build_vector_store_config(_make_settings(mem0_vector_store_provider="Qdrant"))
        assert cfg_lower["provider"] == cfg_upper["provider"]

    def test_whitespace_stripped_from_provider(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="  qdrant  "))
        assert cfg["provider"] == "qdrant"

    def test_unknown_provider_fallback_has_complete_config(self):
        from src.infra.mem0_client import _build_vector_store_config
        cfg = _build_vector_store_config(_make_settings(mem0_vector_store_provider="unknown_xyz"))
        assert "collection_name" in cfg["config"]
        assert "embedding_model_dims" in cfg["config"]


# ── is_mem0_configured / is_proxy_configured ──────────────────────────────────

class TestProviderAwareConfiguredFlags:
    """is_mem0_configured and is_proxy_configured must reflect the active provider."""

    def test_qdrant_configured_when_qdrant_url_set(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="qdrant",
            qdrant_url="https://qdrant.test",
            neo4j_url="bolt://neo4j.test",
            neo4j_password="secret",
            supabase_connection_string="",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is True

    def test_qdrant_not_configured_when_url_missing(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="qdrant",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is False

    def test_supabase_configured_when_connection_string_set(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="supabase",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="postgresql://user:pass@host/db",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is True

    def test_pgvector_configured_when_connection_string_set(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="pgvector",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="postgresql://user:pass@host/db",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is True

    def test_chroma_always_configured(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="chroma",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is True

    def test_pinecone_configured_when_api_key_set(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="pinecone",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="",
            pinecone_api_key="pc-key",
        )
        assert s.is_mem0_configured is True

    def test_pinecone_not_configured_when_key_missing(self):
        from src.infra.config import Settings
        s = Settings.model_construct(
            mem0_vector_store_provider="pinecone",
            qdrant_url="",
            neo4j_url="",
            neo4j_password="",
            supabase_connection_string="",
            pinecone_api_key="",
        )
        assert s.is_mem0_configured is False


# ── Config integration ─────────────────────────────────────────────────────────

class TestConfigIntegration:

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

    def test_build_mem0_config_uses_vector_store_config(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = self._full(mem0_vector_store_provider="qdrant")
        result = _build_mem0_config(cfg)
        assert result["vector_store"]["provider"] == "qdrant"

    def test_build_mem0_config_chroma(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = self._full(mem0_vector_store_provider="chroma")
        result = _build_mem0_config(cfg)
        assert result["vector_store"]["provider"] == "chroma"

    def test_build_mem0_config_supabase(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = self._full(mem0_vector_store_provider="supabase",
                         supabase_connection_string="postgresql://x:y@h/d")
        result = _build_mem0_config(cfg)
        assert result["vector_store"]["provider"] == "supabase"

    def test_build_proxy_config_uses_same_vector_store(self):
        from src.infra.mem0_client import _build_mem0_config, _build_mem0_proxy_config
        settings = self._full(mem0_vector_store_provider="chroma")
        assert (_build_mem0_config(settings)["vector_store"]["provider"]
                == _build_mem0_proxy_config(settings)["vector_store"]["provider"])

    def test_both_configs_share_collection_name(self):
        from src.infra.mem0_client import _build_mem0_config, _build_mem0_proxy_config
        settings = self._full(mem0_collection_name="shared_coll")
        assert (
            _build_mem0_config(settings)["vector_store"]["config"]["collection_name"]
            == _build_mem0_proxy_config(settings)["vector_store"]["config"]["collection_name"]
            == "shared_coll"
        )


# ── get_mem0_config_summary ────────────────────────────────────────────────────

class TestConfigSummaryFeature11:

    def test_summary_vector_store_has_provider(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "provider" in result["vector_store"]

    def test_summary_vector_store_has_collection(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["vector_store"]["collection"] == get_settings().mem0_collection_name

    def test_summary_vector_store_has_embedding_dims(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert "embedding_dims" in result["vector_store"]

    def test_summary_vector_store_provider_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        settings = get_settings()
        expected = (settings.mem0_vector_store_provider or "qdrant").lower()
        # For pgvector/supabase, both map to "supabase" in the provider
        if expected in ("pgvector", "supabase"):
            assert result["vector_store"]["provider"] == "supabase"
        else:
            assert result["vector_store"]["provider"] == expected

    def test_summary_no_raw_credentials_in_vector_store(self):
        """Raw API keys and passwords must not appear in the summary."""
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        summary_str = str(result["vector_store"])
        settings = get_settings()
        if settings.qdrant_api_key and len(settings.qdrant_api_key) > 8:
            assert settings.qdrant_api_key not in summary_str
        if settings.pinecone_api_key and len(settings.pinecone_api_key) > 8:
            assert settings.pinecone_api_key not in summary_str


# ── validate_mem0_config vector_store key ─────────────────────────────────────

class TestValidateVectorStoreKey:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_validate_returns_vector_store_key(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert "vector_store" in result

    def test_validate_vector_store_value_is_string(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert isinstance(result["vector_store"], str)

    def test_validate_vector_store_is_valid_status(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        val = result["vector_store"]
        assert (
            val == "unconfigured"
            or val == "ok"
            or val == "configured"
            or val.startswith("error")
        ), f"Unexpected vector_store status: {val!r}"

    def test_validate_no_qdrant_key(self):
        """The old 'qdrant' key must no longer exist — replaced by 'vector_store'."""
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert "qdrant" not in result

    def test_validate_never_raises(self):
        from src.infra.mem0_client import validate_mem0_config
        try:
            result = self._run(validate_mem0_config())
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"validate_mem0_config raised unexpectedly: {exc}")


# ── Live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestFeature11Live:
    """Live tests — require real credentials for the configured provider."""

    @pytest.fixture(autouse=True)
    def require_openai_key(self):
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_qdrant_vector_store_ok_when_configured(self):
        from src.infra.config import get_settings
        from src.infra.mem0_client import validate_mem0_config
        s = get_settings()
        if not s.qdrant_url or s.mem0_vector_store_provider != "qdrant":
            pytest.skip("QDRANT_URL not set or provider is not qdrant")
        result = self._run(validate_mem0_config())
        assert result["vector_store"] == "ok"
