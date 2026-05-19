import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator
from helpers import make_text_block, make_tool_use_block, make_api_response


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_gen_and_client():
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        mock_client = MockAnthropic.return_value
        gen = AIGenerator(api_key="test-key", model="claude-sonnet-4-6")
        yield gen, mock_client


# ---------------------------------------------------------------------------
# Group 1 — generate_response() direct (no tool use)
# ---------------------------------------------------------------------------

class TestGenerateResponseDirect:

    def test_returns_text_from_content_block(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Hello world")], stop_reason="end_turn"
        )
        result = gen.generate_response(query="What is 2+2?")
        assert result == "Hello world"

    def test_exactly_one_api_call_for_direct_response(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Answer")], stop_reason="end_turn"
        )
        gen.generate_response(query="General question?")
        assert mock_client.messages.create.call_count == 1

    def test_conversation_history_appended_to_system_prompt(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("ok")], stop_reason="end_turn"
        )
        gen.generate_response(query="Follow-up?", conversation_history="User: Hi\nAssistant: Hello")
        system_arg = mock_client.messages.create.call_args[1]["system"]
        assert "Previous conversation:" in system_arg
        assert "User: Hi" in system_arg

    def test_no_history_system_prompt_has_no_previous_section(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("ok")], stop_reason="end_turn"
        )
        gen.generate_response(query="What is Python?", conversation_history=None)
        system_arg = mock_client.messages.create.call_args[1]["system"]
        assert "Previous conversation:" not in system_arg

    def test_tools_and_tool_choice_added_when_tools_provided(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("ok")], stop_reason="end_turn"
        )
        tools = [{"name": "search_course_content", "description": "search"}]
        gen.generate_response(query="q", tools=tools, tool_manager=MagicMock())
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_choice" in call_kwargs

    def test_no_tools_in_params_when_none_provided(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("ok")], stop_reason="end_turn"
        )
        gen.generate_response(query="q", tools=None, tool_manager=None)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# Group 2 — generate_response() tool-use path
# ---------------------------------------------------------------------------

class TestGenerateResponseToolUse:

    def test_one_tool_round_makes_two_api_calls(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block = make_tool_use_block("tu_1", "search_course_content", {"query": "MCP"})
        first_response = make_api_response([tool_block], stop_reason="tool_use")
        second_response = make_api_response([make_text_block("Final answer")], stop_reason="end_turn")
        mock_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search results"

        result = gen.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1
        assert result == "Final answer"

    def test_end_turn_skips_tool_execution(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        mock_client.messages.create.return_value = make_api_response(
            [make_text_block("Direct answer")], stop_reason="end_turn"
        )
        tool_manager = MagicMock()

        gen.generate_response(
            query="General question?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 1
        tool_manager.execute_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Group 3 — sequential tool calling (agentic loop)
# ---------------------------------------------------------------------------

class TestSequentialToolCalling:

    def test_two_tool_rounds_makes_three_api_calls(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block_1 = make_tool_use_block("tu_1", "get_course_outline", {"course_title": "X"})
        tool_block_2 = make_tool_use_block("tu_2", "search_course_content", {"query": "topic"})
        mock_client.messages.create.side_effect = [
            make_api_response([tool_block_1], stop_reason="tool_use"),
            make_api_response([tool_block_2], stop_reason="tool_use"),
            make_api_response([make_text_block("Final answer")], stop_reason="end_turn"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        result = gen.generate_response(
            query="q", tools=[{"name": "get_course_outline"}], tool_manager=tool_manager
        )

        assert mock_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        assert result == "Final answer"

    def test_loop_exits_early_when_second_response_has_no_tool_use(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block = make_tool_use_block("tu_1", "search_course_content", {"query": "RAG"})
        mock_client.messages.create.side_effect = [
            make_api_response([tool_block], stop_reason="tool_use"),
            make_api_response([make_text_block("Done")], stop_reason="end_turn"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "results"

        result = gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert mock_client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1
        assert result == "Done"

    def test_loop_stops_at_max_two_rounds(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block_1 = make_tool_use_block("tu_1", "search_course_content", {"query": "a"})
        tool_block_2 = make_tool_use_block("tu_2", "search_course_content", {"query": "b"})
        mock_client.messages.create.side_effect = [
            make_api_response([tool_block_1], stop_reason="tool_use"),
            make_api_response([tool_block_2], stop_reason="tool_use"),
            make_api_response([make_text_block("Answer")], stop_reason="end_turn"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert mock_client.messages.create.call_count == 3

    def test_message_sequence_correct_after_two_rounds(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block_1 = make_tool_use_block("tu_1", "get_course_outline", {"course_title": "X"})
        tool_block_2 = make_tool_use_block("tu_2", "search_course_content", {"query": "topic"})
        r1 = make_api_response([tool_block_1], stop_reason="tool_use")
        r2 = make_api_response([tool_block_2], stop_reason="tool_use")
        r3 = make_api_response([make_text_block("Final")], stop_reason="end_turn")
        mock_client.messages.create.side_effect = [r1, r2, r3]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "output"

        gen.generate_response(
            query="original question",
            tools=[{"name": "get_course_outline"}],
            tool_manager=tool_manager,
        )

        messages = mock_client.messages.create.call_args_list[2][1]["messages"]
        assert len(messages) == 5
        assert messages[0] == {"role": "user", "content": "original question"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == r1.content
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["tool_use_id"] == "tu_1"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == r2.content
        assert messages[4]["role"] == "user"
        assert messages[4]["content"][0]["tool_use_id"] == "tu_2"

    def test_tools_included_in_all_api_calls_during_loop(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block_1 = make_tool_use_block("tu_1", "search_course_content", {"query": "a"})
        tool_block_2 = make_tool_use_block("tu_2", "search_course_content", {"query": "b"})
        mock_client.messages.create.side_effect = [
            make_api_response([tool_block_1], stop_reason="tool_use"),
            make_api_response([tool_block_2], stop_reason="tool_use"),
            make_api_response([make_text_block("ok")], stop_reason="end_turn"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"
        tools = [{"name": "search_course_content"}]

        gen.generate_response(query="q", tools=tools, tool_manager=tool_manager)

        for call in mock_client.messages.create.call_args_list:
            kwargs = call[1]
            assert "tools" in kwargs
            assert "tool_choice" in kwargs

    def test_second_tool_correct_name_and_kwargs(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block_1 = make_tool_use_block("tu_1", "get_course_outline", {"course_title": "X"})
        tool_block_2 = make_tool_use_block(
            "tu_2", "search_course_content", {"query": "attention", "course_name": "Y"}
        )
        mock_client.messages.create.side_effect = [
            make_api_response([tool_block_1], stop_reason="tool_use"),
            make_api_response([tool_block_2], stop_reason="tool_use"),
            make_api_response([make_text_block("ok")], stop_reason="end_turn"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q", tools=[{"name": "get_course_outline"}], tool_manager=tool_manager
        )

        calls = tool_manager.execute_tool.call_args_list
        assert calls[0] == (("get_course_outline",), {"course_title": "X"})
        assert calls[1] == (("search_course_content",), {"query": "attention", "course_name": "Y"})

    def test_tool_exception_propagates(self, ai_gen_and_client):
        gen, mock_client = ai_gen_and_client
        tool_block = make_tool_use_block("tu_1", "search_course_content", {"query": "q"})
        mock_client.messages.create.return_value = make_api_response(
            [tool_block], stop_reason="tool_use"
        )
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("DB down")

        import pytest
        with pytest.raises(RuntimeError, match="DB down"):
            gen.generate_response(
                query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
            )


# ---------------------------------------------------------------------------
# Group 4 — bug documentation + model ID validation
# ---------------------------------------------------------------------------

class TestBugsAndModelValidation:

    def test_empty_content_list_returns_empty_string(self, ai_gen_and_client):
        """Bug 2 fix: empty content list no longer raises IndexError — returns empty string."""
        gen, mock_client = ai_gen_and_client
        tool_block = make_tool_use_block("tu_1", "search_course_content", {"query": "q"})
        first_response = make_api_response([tool_block], stop_reason="tool_use")
        empty_response = make_api_response(content_blocks=[], stop_reason="end_turn")
        mock_client.messages.create.side_effect = [first_response, empty_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "results"

        result = gen.generate_response(
            query="What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )
        assert result == ""

    def test_validate_model_id_is_current(self):
        """Bug 1: config uses an invalid/deprecated model ID.
        This test fails on the unfixed codebase and passes after the fix."""
        from config import config

        valid_model_ids = {
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        }
        assert config.ANTHROPIC_MODEL in valid_model_ids, (
            f"Model '{config.ANTHROPIC_MODEL}' is not a recognised Claude 4 model ID. "
            f"Expected one of: {valid_model_ids}"
        )
