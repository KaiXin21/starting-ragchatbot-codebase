import pytest


# ---------------------------------------------------------------------------
# Group 1 — POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_valid_query_returns_200(self, test_client, mock_rag_system):
        response = test_client.post("/api/query", json={"query": "What is RAG?"})
        assert response.status_code == 200

    def test_response_contains_answer(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("RAG is retrieval-augmented generation", [])
        response = test_client.post("/api/query", json={"query": "What is RAG?"})
        assert response.json()["answer"] == "RAG is retrieval-augmented generation"

    def test_response_contains_session_id_string(self, test_client, mock_rag_system):
        response = test_client.post("/api/query", json={"query": "hello"})
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_response_contains_sources_list(self, test_client, mock_rag_system):
        response = test_client.post("/api/query", json={"query": "hello"})
        assert isinstance(response.json()["sources"], list)

    def test_auto_creates_session_when_none_provided(self, test_client, mock_rag_system):
        response = test_client.post("/api/query", json={"query": "hello"})
        assert response.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_auto_created_session_id_in_response(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "auto-session-99"
        response = test_client.post("/api/query", json={"query": "hello"})
        assert response.json()["session_id"] == "auto-session-99"

    def test_uses_provided_session_id(self, test_client, mock_rag_system):
        response = test_client.post(
            "/api/query",
            json={"query": "hello", "session_id": "my-session-42"},
        )
        assert response.json()["session_id"] == "my-session-42"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_provided_session_id_forwarded_to_rag(self, test_client, mock_rag_system):
        test_client.post("/api/query", json={"query": "hello", "session_id": "sess-7"})
        mock_rag_system.query.assert_called_once_with("hello", "sess-7")

    def test_sources_with_url_in_response(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "answer",
            [{"text": "MCP Course - Lesson 1", "url": "https://example.com/lesson1"}],
        )
        response = test_client.post("/api/query", json={"query": "MCP?"})
        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"

    def test_sources_with_null_url_in_response(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "answer",
            [{"text": "MCP Course - Lesson 1", "url": None}],
        )
        response = test_client.post("/api/query", json={"query": "MCP?"})
        assert response.json()["sources"][0]["url"] is None

    def test_empty_sources_list_in_response(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("answer", [])
        response = test_client.post("/api/query", json={"query": "q"})
        assert response.json()["sources"] == []

    def test_rag_error_returns_500(self, test_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB connection failed")
        response = test_client.post("/api/query", json={"query": "hello"})
        assert response.status_code == 500

    def test_missing_query_field_returns_422(self, test_client, mock_rag_system):
        response = test_client.post("/api/query", json={"session_id": "s1"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Group 2 — GET /api/courses
# ---------------------------------------------------------------------------

class TestCourseStatsEndpoint:

    def test_returns_200(self, test_client, mock_rag_system):
        response = test_client.get("/api/courses")
        assert response.status_code == 200

    def test_returns_total_courses_count(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        response = test_client.get("/api/courses")
        assert response.json()["total_courses"] == 5

    def test_returns_course_titles_list(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["RAG Basics", "MCP Course"],
        }
        response = test_client.get("/api/courses")
        assert response.json()["course_titles"] == ["RAG Basics", "MCP Course"]

    def test_empty_catalog_returns_zero_and_empty_list(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        response = test_client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_error_returns_500(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("Store offline")
        response = test_client.get("/api/courses")
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Group 3 — DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200(self, test_client, mock_rag_system):
        response = test_client.delete("/api/session/any-session")
        assert response.status_code == 200

    def test_existing_session_returns_existed_true(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.delete_session.return_value = True
        response = test_client.delete("/api/session/session_42")
        assert response.json()["existed"] is True

    def test_nonexistent_session_returns_existed_false(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.delete_session.return_value = False
        response = test_client.delete("/api/session/ghost-session")
        assert response.json()["existed"] is False

    def test_response_contains_status_ok(self, test_client, mock_rag_system):
        response = test_client.delete("/api/session/any-session")
        assert response.json()["status"] == "ok"

    def test_correct_session_id_forwarded_to_manager(self, test_client, mock_rag_system):
        test_client.delete("/api/session/target-session-99")
        mock_rag_system.session_manager.delete_session.assert_called_once_with(
            "target-session-99"
        )
