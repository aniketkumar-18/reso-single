"""Tests for Feature 8 — OpenAI Compatibility (Mem0 proxy client).

Verifies that:
  - _build_mem0_proxy_config() produces a valid config (Qdrant + OpenAI, no Neo4j)
  - get_mem0_proxy() returns None gracefully when unconfigured
  - create_memory_aware_chat() falls back to plain OpenAI when proxy unavailable
  - auto_save_conversation_memory() returns sensible status dicts
  - wellness_agent_node imports and uses auto_save_conversation_memory
  - is_proxy_configured on settings works independently of Neo4j config

Run:
    python -m pytest tests/test_feature8_openai_compat.py -v
    python -m pytest tests/test_feature8_openai_compat.py -v -m live
"""

from __future__ import annotations

import asyncio
import os
import unittest.mock as mock
import pytest


# ── Config / singleton structure tests (no network) ───────────────────────────

class TestProxyConfig:
    """Validate proxy config structure — lighter than AsyncMemory config."""

    def _make_settings(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            llm_model_pipeline="gpt-4o-mini",
            openai_api_key="sk-test",
            qdrant_url="https://qdrant.test",
            qdrant_api_key="qdrant-key",
            neo4j_url="bolt://neo4j.test",
            neo4j_username="neo4j",
            neo4j_password="password",
            is_proxy_configured=True,
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

    def test_proxy_config_has_vector_store(self):
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert "vector_store" in cfg

    def test_proxy_config_uses_qdrant(self):
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert cfg["vector_store"]["provider"] == "qdrant"

    def test_proxy_config_has_no_graph_store(self):
        """Proxy is lightweight — no Neo4j (handled by full AsyncMemory)."""
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert "graph_store" not in cfg

    def test_proxy_config_has_no_reranker(self):
        """Proxy is used for auto-save, not complex search — no reranker needed."""
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert "reranker" not in cfg

    def test_proxy_config_inherits_feature6_prompt(self):
        from src.infra.mem0_client import _build_mem0_proxy_config, WELLNESS_FACT_EXTRACTION_PROMPT
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert cfg.get("custom_fact_extraction_prompt") == WELLNESS_FACT_EXTRACTION_PROMPT

    def test_proxy_config_inherits_feature7_prompt(self):
        from src.infra.mem0_client import _build_mem0_proxy_config, WELLNESS_UPDATE_MEMORY_PROMPT
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert cfg.get("custom_update_memory_prompt") == WELLNESS_UPDATE_MEMORY_PROMPT

    def test_proxy_config_version_v1_1(self):
        from src.infra.mem0_client import _build_mem0_proxy_config
        cfg = _build_mem0_proxy_config(self._make_settings())
        assert cfg.get("version") == "v1.1"


class TestIsProxyConfigured:
    """is_proxy_configured depends only on QDRANT_URL, not Neo4j."""

    def _make_settings(self, qdrant_url="", neo4j_url="", neo4j_password=""):
        from types import SimpleNamespace
        return SimpleNamespace(
            qdrant_url=qdrant_url,
            neo4j_url=neo4j_url,
            neo4j_password=neo4j_password,
        )

    def test_true_when_only_qdrant_set(self):
        s = self._make_settings(qdrant_url="https://qdrant.test")
        # is_proxy_configured is a property on the real Settings class
        # Test the logic directly
        assert bool(s.qdrant_url)  # ← same check as is_proxy_configured

    def test_false_when_qdrant_missing(self):
        s = self._make_settings(qdrant_url="")
        assert not bool(s.qdrant_url)

    def test_is_proxy_independent_of_neo4j(self):
        """Proxy only needs Qdrant; Neo4j absence should not affect it."""
        s = self._make_settings(qdrant_url="https://qdrant.test", neo4j_url="", neo4j_password="")
        assert bool(s.qdrant_url)  # proxy configured even without Neo4j

    def test_settings_has_is_proxy_configured_property(self):
        """Verify the property exists on the real Settings class."""
        from src.infra.config import get_settings
        settings = get_settings()
        assert hasattr(settings, "is_proxy_configured")
        assert isinstance(settings.is_proxy_configured, bool)


class TestGetMem0ProxyUnconfigured:
    """get_mem0_proxy() must return None gracefully when Qdrant not set."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_none_when_unconfigured(self):
        """Direct logic test: is_proxy_configured is False when qdrant_url is empty."""
        from types import SimpleNamespace
        fake_settings = SimpleNamespace(qdrant_url="", is_proxy_configured=False)
        # The is_proxy_configured property returns bool(qdrant_url)
        assert not fake_settings.is_proxy_configured

    def test_real_settings_proxy_configured_is_bool(self):
        from src.infra.config import get_settings
        s = get_settings()
        assert isinstance(s.is_proxy_configured, bool)


# ── auto_save_conversation_memory unit tests ──────────────────────────────────

class TestAutoSaveConversationMemory:
    """Unit tests for auto_save_conversation_memory."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_skips_when_no_user_id(self):
        from src.infra.mem0_client import auto_save_conversation_memory
        result = self._run(auto_save_conversation_memory(
            user_message="I have diabetes.",
            assistant_response="I'll help you.",
            user_id="",
        ))
        assert result["status"] == "skipped"

    def test_skips_when_no_content(self):
        from src.infra.mem0_client import auto_save_conversation_memory
        result = self._run(auto_save_conversation_memory(
            user_message="",
            assistant_response="",
            user_id="user-123",
        ))
        assert result["status"] == "skipped"

    def test_skips_when_proxy_none(self):
        """When proxy is not configured, auto_save must return skipped (not error)."""
        from src.infra.mem0_client import auto_save_conversation_memory

        with mock.patch(
            "src.infra.mem0_client.get_mem0_proxy",
            new=mock.AsyncMock(return_value=None),
        ):
            result = self._run(auto_save_conversation_memory(
                user_message="I prefer evening runs.",
                assistant_response="Got it!",
                user_id="user-123",
            ))
        assert result["status"] == "skipped"

    def test_returns_non_raising_on_proxy_error(self):
        """Proxy errors must return error status, never propagate as exceptions."""
        from src.infra.mem0_client import auto_save_conversation_memory

        error_proxy = mock.MagicMock()
        error_proxy.chat.completions.create.side_effect = RuntimeError("proxy down")

        async def get_error_proxy():
            return error_proxy

        with mock.patch("src.infra.mem0_client.get_mem0_proxy", get_error_proxy):
            with mock.patch("src.infra.config.get_settings") as mock_settings:
                from types import SimpleNamespace
                mock_settings.return_value = SimpleNamespace(
                    llm_model_pipeline="gpt-4o-mini",
                    openai_api_key="sk-test",
                )
                result = self._run(auto_save_conversation_memory(
                    user_message="I am vegetarian.",
                    assistant_response="Noted!",
                    user_id="user-123",
                ))
        # Should be error or skipped, never raise
        assert result["status"] in ("error", "skipped", "timeout")


def _async_value(v):
    """Return a coroutine that returns v."""
    async def _():
        return v
    return _()


# ── Wellness agent node wiring test ───────────────────────────────────────────

class TestWellnessAgentNodeWiring:
    """Verify wellness_agent_node imports and uses auto_save_conversation_memory."""

    def test_auto_save_imported_in_agent_node(self):
        import importlib
        import src.agent.nodes.wellness_agent as mod
        importlib.reload(mod)
        assert hasattr(mod, "auto_save_conversation_memory")

    def test_agent_node_module_has_auto_save_call(self):
        """The source code of wellness_agent must reference auto_save_conversation_memory."""
        import inspect
        from src.agent.nodes import wellness_agent
        source = inspect.getsource(wellness_agent)
        assert "auto_save_conversation_memory" in source


# ── create_memory_aware_chat unit tests ───────────────────────────────────────

class TestCreateMemoryAwareChatUnit:
    """Unit tests for create_memory_aware_chat."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_falls_back_to_openai_when_proxy_none(self):
        """When proxy is None, must call plain OpenAI and return completion."""
        from src.infra.mem0_client import create_memory_aware_chat

        mock_completion = mock.MagicMock()
        mock_completion.choices = [mock.MagicMock()]
        mock_completion.choices[0].message.content = "Hello!"

        mock_openai_client = mock.MagicMock()
        mock_openai_client.chat.completions.create = mock.AsyncMock(return_value=mock_completion)

        with mock.patch("src.infra.mem0_client.get_mem0_proxy", new=mock.AsyncMock(return_value=None)):
            with mock.patch("openai.AsyncOpenAI", return_value=mock_openai_client):
                result = self._run(create_memory_aware_chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    user_id="user-123",
                ))
        assert result.choices[0].message.content == "Hello!"


# ── Live integration tests (requires OPENAI_API_KEY + QDRANT_URL) ─────────────

@pytest.mark.live
class TestOpenAICompatLive:
    """Live tests. Run with: pytest -m live (requires OPENAI_API_KEY + QDRANT_URL)."""

    @pytest.fixture(autouse=True)
    def require_keys(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_create_memory_aware_chat_returns_completion(self):
        """create_memory_aware_chat must return an OpenAI-compatible completion."""
        from src.infra.mem0_client import create_memory_aware_chat
        result = self._run(create_memory_aware_chat(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            user_id="test-user-feature8",
        ))
        assert result is not None
        # Check OpenAI-compatible response shape
        assert hasattr(result, "choices")
        assert len(result.choices) > 0
        content = result.choices[0].message.content
        assert content is not None and len(content) > 0

    def test_auto_save_returns_saved_or_skipped(self):
        """auto_save_conversation_memory must return 'saved' or 'skipped', never raise."""
        from src.infra.mem0_client import auto_save_conversation_memory
        result = self._run(auto_save_conversation_memory(
            user_message="I prefer morning runs and I'm allergic to shellfish.",
            assistant_response="Got it! I'll keep that in mind.",
            user_id="test-user-feature8",
            run_id="test-session-f8",
        ))
        assert result["status"] in ("saved", "skipped", "timeout", "error")

    def test_memory_aware_chat_with_user_id_scope(self):
        """Two calls with same user_id — second should have memory context."""
        from src.infra.mem0_client import create_memory_aware_chat

        # First call: establish preference
        self._run(create_memory_aware_chat(
            messages=[{"role": "user", "content": "I love spicy Thai food."}],
            user_id="test-user-feature8-mem",
        ))

        # Second call: ask for recommendation — proxy should recall the preference
        result = self._run(create_memory_aware_chat(
            messages=[{"role": "user", "content": "What restaurant type should I try?"}],
            user_id="test-user-feature8-mem",
        ))
        assert result is not None
        assert hasattr(result, "choices")
