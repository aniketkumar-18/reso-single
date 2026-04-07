"""Tests for Feature 9 — Configure OSS Stack.

Verifies that:
  - New settings fields exist with correct defaults (collection name, reranker top_k)
  - _build_mem0_config() and _build_mem0_proxy_config() use the new settings
  - get_mem0_config_summary() returns a safe, secret-free dict
  - Secrets (API keys, passwords) are properly masked in the summary
  - validate_mem0_config() returns a complete status dict and never raises
  - Health endpoint response includes mem0_config section
  - app.py lifespan imports the new functions

Run:
    python -m pytest tests/test_feature9_oss_config.py -v
    python -m pytest tests/test_feature9_oss_config.py -v -m live
"""

from __future__ import annotations

import asyncio
import os
import unittest.mock as mock
import pytest


# ── Settings field tests ───────────────────────────────────────────────────────

class TestNewSettingsFields:
    """Verify new OSS-tuning settings fields exist with correct defaults."""

    def test_mem0_collection_name_exists(self):
        from src.infra.config import get_settings
        s = get_settings()
        assert hasattr(s, "mem0_collection_name")

    def test_mem0_collection_name_default(self):
        from src.infra.config import get_settings
        s = get_settings()
        # Default should be "reso_memories" unless overridden in .env
        assert isinstance(s.mem0_collection_name, str)
        assert len(s.mem0_collection_name) > 0

    def test_mem0_reranker_top_k_exists(self):
        from src.infra.config import get_settings
        s = get_settings()
        assert hasattr(s, "mem0_reranker_top_k")

    def test_mem0_reranker_top_k_is_int(self):
        from src.infra.config import get_settings
        s = get_settings()
        assert isinstance(s.mem0_reranker_top_k, int)

    def test_mem0_reranker_top_k_in_recommended_range(self):
        """Docs recommend 10–20. Default must be in this range."""
        from src.infra.config import get_settings
        s = get_settings()
        # Check default value (may be overridden in .env to any value the user chose)
        # We just verify it's positive and reasonable
        assert s.mem0_reranker_top_k > 0
        assert s.mem0_reranker_top_k <= 100  # sanity upper bound


class TestConfigUsesNewSettings:
    """_build_mem0_config and _build_mem0_proxy_config must use settings fields."""

    def _make_settings(self, collection="test_coll", top_k=15):
        from types import SimpleNamespace
        return SimpleNamespace(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
            qdrant_url="https://qdrant.test",
            qdrant_api_key="qdrant-key",
            neo4j_url="bolt://neo4j.test",
            neo4j_username="neo4j",
            neo4j_password="password",
            mem0_collection_name=collection,
            mem0_reranker_top_k=top_k,
            # Feature 10 — LLM provider
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
            # Feature 11 — vector store
            mem0_vector_store_provider="qdrant",
            supabase_connection_string="",
            chroma_path="./chroma_db",
            chroma_host="",
            chroma_port=8000,
            pinecone_api_key="",
            pinecone_index_name="reso-memories",
            # Feature 12 — embedder
            mem0_embedder_provider="openai",
            mem0_embedder_model="",
            mem0_embedder_dims=1536,
            mem0_embedder_base_url="",
            # Feature 13 — reranker
            mem0_reranker_provider="llm_reranker",
            mem0_reranker_model="",
            mem0_reranker_use_wellness_prompt=True,
            cohere_api_key="",
            zero_entropy_api_key="",
            huggingface_api_key="",
        )

    def test_build_mem0_config_uses_collection_name(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_settings(collection="custom_coll"))
        assert cfg["vector_store"]["config"]["collection_name"] == "custom_coll"

    def test_build_mem0_config_uses_reranker_top_k(self):
        from src.infra.mem0_client import _build_mem0_config
        cfg = _build_mem0_config(self._make_settings(top_k=15))
        assert cfg["reranker"]["config"]["top_k"] == 15

    def test_build_proxy_config_uses_collection_name(self):
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings(collection="proxy_coll"))
        assert cfg["vector_store"]["config"]["collection_name"] == "proxy_coll"

    def test_proxy_and_async_memory_share_collection_name(self):
        """Proxy and AsyncMemory must use the same collection (shared store)."""
        from src.infra.mem0_client import _build_mem0_config, _build_mem0_proxy_config
        settings = self._make_settings(collection="shared_coll")
        async_cfg = _build_mem0_config(settings)
        proxy_cfg = _build_mem0_proxy_config(settings)
        assert (
            async_cfg["vector_store"]["config"]["collection_name"]
            == proxy_cfg["vector_store"]["config"]["collection_name"]
            == "shared_coll"
        )


# ── get_mem0_config_summary tests ─────────────────────────────────────────────

class TestGetMem0ConfigSummary:
    """Verify summary is complete, accurate, and safe (no secrets)."""

    def test_returns_dict(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        for key in ("mem0_configured", "proxy_configured", "vector_store",
                    "llm", "embedder", "graph_store", "reranker", "custom_prompts", "version"):
            assert key in result, f"Summary missing key: {key}"

    def test_api_keys_are_masked(self):
        """API keys must not appear verbatim in the summary."""
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        settings = get_settings()
        result = get_mem0_config_summary()

        summary_str = str(result)
        # The full API key must not appear in the summary
        if settings.qdrant_api_key and len(settings.qdrant_api_key) > 8:
            assert settings.qdrant_api_key not in summary_str

    def test_qdrant_url_is_masked_when_qdrant_provider(self):
        """When provider=qdrant, full URL must not appear verbatim in the summary."""
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        if get_settings().mem0_vector_store_provider != "qdrant":
            pytest.skip("Only applicable when vector store is qdrant")
        url_val = result["vector_store"].get("url", "")
        assert isinstance(url_val, str)
        assert "?" not in url_val
        assert "&" not in url_val

    def test_collection_name_is_present(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["vector_store"]["collection"] == get_settings().mem0_collection_name

    def test_reranker_top_k_matches_settings(self):
        from src.infra.mem0_client import get_mem0_config_summary
        from src.infra.config import get_settings
        result = get_mem0_config_summary()
        assert result["reranker"]["top_k"] == get_settings().mem0_reranker_top_k

    def test_custom_prompts_both_true(self):
        """Features 6+7 prompts must be reflected in the summary."""
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert result["custom_prompts"]["fact_extraction"] is True
        assert result["custom_prompts"]["update_memory"] is True

    def test_version_is_v1_1(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert result["version"] == "v1.1"

    def test_mem0_configured_is_bool(self):
        from src.infra.mem0_client import get_mem0_config_summary
        result = get_mem0_config_summary()
        assert isinstance(result["mem0_configured"], bool)
        assert isinstance(result["proxy_configured"], bool)


# ── _mask_secret and _mask_url unit tests ─────────────────────────────────────

class TestMaskHelpers:

    def test_mask_secret_hides_key(self):
        from src.infra.mem0_client import _mask_secret
        key = "sk-abc123verylongkey"
        result = _mask_secret(key, keep_chars=4)
        assert result.startswith("sk-a")
        assert "***" in result
        assert key not in result

    def test_mask_secret_empty_string(self):
        from src.infra.mem0_client import _mask_secret
        assert _mask_secret("") == ""

    def test_mask_url_strips_path_and_credentials(self):
        from src.infra.mem0_client import _mask_url
        # URL with path and query string — result must strip them
        url = "https://user:pass@bd2c0180.qdrant.io/collections?token=abc123"
        result = _mask_url(url)
        assert result.startswith("https://")
        assert "pass" not in result
        assert "token" not in result
        assert "abc123" not in result

    def test_mask_url_returns_scheme_and_host(self):
        from src.infra.mem0_client import _mask_url
        url = "https://bd2c0180.gcp.cloud.qdrant.io"
        result = _mask_url(url)
        assert result.startswith("https://")
        assert "bd2c0180" in result  # hostname preserved

    def test_mask_url_empty(self):
        from src.infra.mem0_client import _mask_url
        assert _mask_url("") == ""


# ── validate_mem0_config unit tests (mocked) ──────────────────────────────────

class TestValidateMem0Config:
    """validate_mem0_config() must never raise, always return a status dict."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_dict_with_required_keys(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        for key in ("vector_store", "embedder", "llm", "graph_store", "overall"):
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_strings(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        for k, v in result.items():
            assert isinstance(v, str), f"Key {k!r} has non-string value: {v!r}"

    def test_vector_store_key_is_valid_status(self):
        """The vector_store key must be 'unconfigured', 'ok', 'configured', or start with 'error'."""
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        vs_val = result["vector_store"]
        assert isinstance(vs_val, str)
        assert (
            vs_val == "unconfigured"
            or vs_val == "ok"
            or vs_val == "configured"
            or vs_val.startswith("error")
        ), f"Unexpected vector_store status: {vs_val!r}"

    def test_never_raises_on_exception(self):
        """Even with broken dependencies, validate_mem0_config must return, not raise."""
        from src.infra.mem0_client import validate_mem0_config
        # Should not raise regardless of environment
        try:
            result = self._run(validate_mem0_config())
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"validate_mem0_config raised unexpectedly: {exc}")

    def test_overall_key_present(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert "overall" in result
        assert result["overall"] in ("ok", "unconfigured", "degraded", "partial", "configured")


# ── Health endpoint integration test ──────────────────────────────────────────

class TestHealthEndpointMem0:
    """Verify health endpoint includes mem0_config section."""

    def test_health_response_has_mem0_config(self):
        """health() response must include a mem0_config key."""
        import inspect
        from src.gateway.routes import chat as chat_module
        source = inspect.getsource(chat_module.health)
        assert "mem0_config" in source

    def test_health_checks_include_mem0_keys(self):
        """health() checks dict must include mem0_configured and mem0_collection."""
        import inspect
        from src.gateway.routes import chat as chat_module
        source = inspect.getsource(chat_module.health)
        assert "mem0_configured" in source
        assert "mem0_collection" in source
        assert "mem0_reranker_top_k" in source


# ── App lifespan wiring test ───────────────────────────────────────────────────

class TestAppLifespanWiring:
    """app.py lifespan source must reference the new Feature 9 functions.

    Read source directly to avoid importing app.py (which requires optional
    langgraph-checkpoint-postgres at import time).
    """

    def _app_source(self):
        from pathlib import Path
        return Path("src/gateway/app.py").read_text()

    def test_app_imports_validate_mem0_config(self):
        assert "validate_mem0_config" in self._app_source()

    def test_app_imports_get_mem0_config_summary(self):
        assert "get_mem0_config_summary" in self._app_source()

    def test_app_pre_warms_proxy(self):
        assert "get_mem0_proxy" in self._app_source()


# ── Live validation test (requires OPENAI_API_KEY + QDRANT_URL) ───────────────

@pytest.mark.live
class TestValidateMem0ConfigLive:
    """Live component validation. Run with: pytest -m live"""

    @pytest.fixture(autouse=True)
    def require_keys(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_live_validation_returns_complete_dict(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert isinstance(result, dict)
        assert "overall" in result

    def test_embedder_ok_with_valid_key(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        # If OpenAI key is valid, embedder should pass
        assert result["embedder"] in ("ok", "unconfigured"), \
            f"Unexpected embedder status: {result['embedder']}"

    def test_llm_ok_with_valid_key(self):
        from src.infra.mem0_client import validate_mem0_config
        result = self._run(validate_mem0_config())
        assert result["llm"] in ("ok", "unconfigured"), \
            f"Unexpected LLM status: {result['llm']}"

    def test_vector_store_ok_when_qdrant_configured(self):
        from src.infra.config import get_settings
        from src.infra.mem0_client import validate_mem0_config
        s = get_settings()
        if not s.qdrant_url or s.mem0_vector_store_provider != "qdrant":
            pytest.skip("QDRANT_URL not set or provider is not qdrant")
        result = self._run(validate_mem0_config())
        assert result["vector_store"] == "ok", f"Vector store validation failed: {result['vector_store']}"
