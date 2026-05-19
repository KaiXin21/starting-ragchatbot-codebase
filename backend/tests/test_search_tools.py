import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, call

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from helpers import make_search_results


# ---------------------------------------------------------------------------
# Group 1 — execute() return values
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecuteReturnValues:

    def test_happy_path_formats_header_and_body(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["RAG explained here"],
            metas=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        mock_vector_store.get_lesson_link.return_value = "https://x.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="what is MCP?")

        assert "[MCP Course - Lesson 1]" in result
        assert "RAG explained here" in result

    def test_empty_results_no_filters(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything")

        assert result == "No relevant content found."

    def test_empty_results_with_course_filter(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything", course_name="MCP")

        assert result == "No relevant content found in course 'MCP'."

    def test_empty_results_with_lesson_filter(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything", lesson_number=3)

        assert result == "No relevant content found in lesson 3."

    def test_error_result_returned_verbatim(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            [], [], error="Search error: Invalid collection"
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything")

        assert result == "Search error: Invalid collection"

    def test_execute_passes_all_kwargs_to_store(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q", course_name="MCP", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="q", course_name="MCP", lesson_number=2
        )

    def test_multiple_docs_joined_by_blank_lines(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["First doc", "Second doc"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="q")

        assert "First doc" in result
        assert "Second doc" in result
        assert "[Course A - Lesson 1]" in result
        assert "[Course A - Lesson 2]" in result


# ---------------------------------------------------------------------------
# Group 2 — last_sources tracking
# ---------------------------------------------------------------------------

class TestCourseSearchToolLastSources:

    def test_sources_populated_with_url(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        mock_vector_store.get_lesson_link.return_value = "https://x.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q")

        assert tool.last_sources == [
            {"text": "MCP Course - Lesson 1", "url": "https://x.com/lesson1"}
        ]

    def test_sources_url_none_when_no_link(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q")

        assert tool.last_sources[0]["url"] is None

    def test_sources_empty_after_empty_results(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results([], [])
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q")
        assert tool.last_sources == []

    def test_sources_empty_after_error_results(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            [], [], error="Some error"
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q")
        assert tool.last_sources == []

    def test_sources_replaced_on_second_call(self, mock_vector_store):
        mock_vector_store.search.side_effect = [
            make_search_results(
                docs=["doc"],
                metas=[{"course_title": "A", "lesson_number": 1}],
            ),
            make_search_results([], []),
        ]
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="first")
        assert len(tool.last_sources) == 1

        tool.execute(query="second")
        assert tool.last_sources == []

    def test_sources_no_lesson_number(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "RAG Basics"}],  # no lesson_number key
        )
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="q")

        assert len(tool.last_sources) == 1
        assert "Lesson" not in tool.last_sources[0]["text"]
        assert tool.last_sources[0]["url"] is None
        mock_vector_store.get_lesson_link.assert_not_called()


# ---------------------------------------------------------------------------
# Group 3 — ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:

    def _make_tool_manager(self, mock_vector_store):
        mock_vector_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "Course", "lesson_number": 1}],
        )
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)
        return tm, tool

    def test_execute_dispatches_to_search_tool(self, mock_vector_store):
        tm, _ = self._make_tool_manager(mock_vector_store)
        result = tm.execute_tool("search_course_content", query="test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_last_sources_returns_tool_sources(self, mock_vector_store):
        tm, tool = self._make_tool_manager(mock_vector_store)
        tm.execute_tool("search_course_content", query="test")
        assert tm.get_last_sources() is tool.last_sources

    def test_reset_sources_clears_all(self, mock_vector_store):
        tm, tool = self._make_tool_manager(mock_vector_store)
        tm.execute_tool("search_course_content", query="test")
        assert len(tm.get_last_sources()) > 0

        tm.reset_sources()
        assert tm.get_last_sources() == []
        assert tool.last_sources == []

    def test_unknown_tool_returns_error_string(self, mock_vector_store):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent_tool", query="q")
        assert "not found" in result.lower()
