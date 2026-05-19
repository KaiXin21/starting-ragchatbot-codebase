import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, ValidationError
from typing import List, Optional


# ---------------------------------------------------------------------------
# Inline Pydantic models — mirrors app.py without triggering module-level
# RAGSystem instantiation (which would try to connect to ChromaDB)
# ---------------------------------------------------------------------------

class SourceItem(BaseModel):
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


# ---------------------------------------------------------------------------
# RAGSystem fixture — patches all external dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def rag_and_mocks():
    with patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.DocumentProcessor"), \
         patch("rag_system.SessionManager") as MockSM:

        cfg = MagicMock()
        cfg.ANTHROPIC_API_KEY = "test-key"
        cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
        cfg.CHROMA_PATH = "/tmp/fake_chroma"
        cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        cfg.MAX_RESULTS = 5
        cfg.MAX_HISTORY = 2
        cfg.CHUNK_SIZE = 800
        cfg.CHUNK_OVERLAP = 100

        from rag_system import RAGSystem
        system = RAGSystem(cfg)

        mock_ai = MockAI.return_value
        mock_vs = MockVS.return_value
        mock_sm = MockSM.return_value

        yield system, mock_ai, mock_vs, mock_sm


# ---------------------------------------------------------------------------
# Group 1 — query() return contract
# ---------------------------------------------------------------------------

class TestRAGSystemQueryContract:

    def test_returns_two_element_tuple(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "The answer"
        result = system.query("What is RAG?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_string(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "The answer"
        answer, _ = system.query("What is RAG?")
        assert isinstance(answer, str)
        assert answer == "The answer"

    def test_second_element_is_list(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"
        _, sources = system.query("What is RAG?")
        assert isinstance(sources, list)

    def test_sources_populated_from_search_tool(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"
        # Simulate a tool call having populated last_sources
        system.search_tool.last_sources = [{"text": "MCP Course - Lesson 1", "url": "https://x.com"}]

        _, sources = system.query("What is MCP?")

        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"
        assert sources[0]["url"] == "https://x.com"

    def test_sources_reset_after_query(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"
        system.search_tool.last_sources = [{"text": "X", "url": None}]

        system.query("What is MCP?")

        assert system.search_tool.last_sources == []

    def test_prompt_embeds_original_query(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("Explain transformers")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "Explain transformers" in call_kwargs["query"]

    def test_tools_passed_to_ai_generator(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("q")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) > 0

    def test_tool_manager_passed_to_ai_generator(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("q")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["tool_manager"] is system.tool_manager


# ---------------------------------------------------------------------------
# Group 2 — session handling
# ---------------------------------------------------------------------------

class TestRAGSystemSessionHandling:

    def test_with_session_id_fetches_conversation_history(self, rag_and_mocks):
        system, mock_ai, _, mock_sm = rag_and_mocks
        mock_ai.generate_response.return_value = "answer"
        mock_sm.get_conversation_history.return_value = "User: Hi\nAssistant: Hello"

        system.query("Follow-up?", session_id="session_1")

        mock_sm.get_conversation_history.assert_called_once_with("session_1")

    def test_with_session_id_forwards_history_to_ai(self, rag_and_mocks):
        system, mock_ai, _, mock_sm = rag_and_mocks
        mock_ai.generate_response.return_value = "answer"
        mock_sm.get_conversation_history.return_value = "past history"

        system.query("Follow-up?", session_id="session_1")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "past history"

    def test_with_session_id_saves_exchange(self, rag_and_mocks):
        system, mock_ai, _, mock_sm = rag_and_mocks
        mock_ai.generate_response.return_value = "My answer"

        system.query("My question", session_id="session_1")

        mock_sm.add_exchange.assert_called_once_with("session_1", "My question", "My answer")

    def test_without_session_id_no_history_lookup(self, rag_and_mocks):
        system, mock_ai, _, mock_sm = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("General question")

        mock_sm.get_conversation_history.assert_not_called()

    def test_without_session_id_no_exchange_saved(self, rag_and_mocks):
        system, mock_ai, _, mock_sm = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("General question")

        mock_sm.add_exchange.assert_not_called()

    def test_without_session_id_history_is_none(self, rag_and_mocks):
        system, mock_ai, _, _ = rag_and_mocks
        mock_ai.generate_response.return_value = "ok"

        system.query("General question")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None


# ---------------------------------------------------------------------------
# Group 3 — Pydantic contract: sources dict shape accepted by QueryResponse
# ---------------------------------------------------------------------------

class TestPydanticResponseContract:

    def test_sources_dict_with_url_coerced_to_source_item(self):
        resp = QueryResponse(
            answer="Some answer",
            sources=[{"text": "MCP Course - Lesson 1", "url": "https://example.com/lesson1"}],
            session_id="session_1",
        )
        assert resp.sources[0].text == "MCP Course - Lesson 1"
        assert resp.sources[0].url == "https://example.com/lesson1"

    def test_sources_dict_with_url_none_accepted(self):
        resp = QueryResponse(
            answer="Some answer",
            sources=[{"text": "MCP Course - Lesson 1", "url": None}],
            session_id="session_1",
        )
        assert resp.sources[0].url is None

    def test_empty_sources_list_accepted(self):
        resp = QueryResponse(answer="answer", sources=[], session_id="s")
        assert resp.sources == []

    def test_source_item_requires_text_field(self):
        with pytest.raises(ValidationError):
            SourceItem(url="https://example.com")  # text is required

    def test_source_item_text_only_is_valid(self):
        item = SourceItem(text="Course - Lesson 1")
        assert item.text == "Course - Lesson 1"
        assert item.url is None

    def test_multiple_sources_coerced_correctly(self):
        resp = QueryResponse(
            answer="a",
            sources=[
                {"text": "Course A - Lesson 1", "url": "https://a.com"},
                {"text": "Course B - Lesson 2", "url": None},
            ],
            session_id="s",
        )
        assert len(resp.sources) == 2
        assert resp.sources[1].text == "Course B - Lesson 2"
        assert resp.sources[1].url is None
