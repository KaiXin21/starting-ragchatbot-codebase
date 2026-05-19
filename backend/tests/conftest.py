import sys
import os

# Make both backend/ and backend/tests/ importable
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_tests_dir)
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _tests_dir)

import pytest
from unittest.mock import MagicMock
from helpers import make_search_results


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.get_lesson_link.return_value = None
    store.search.return_value = make_search_results([], [])
    return store


@pytest.fixture
def mock_rag_system():
    """Fully-mocked RAGSystem with sensible defaults for API endpoint tests."""
    mock = MagicMock()
    mock.query.return_value = ("Test answer", [])
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    mock.session_manager.create_session.return_value = "session_auto_1"
    mock.session_manager.delete_session.return_value = True
    return mock


@pytest.fixture
def test_client(mock_rag_system):
    """TestClient backed by a minimal FastAPI app that mirrors the real endpoints.

    Defined here rather than importing app.py to avoid its module-level
    RAGSystem instantiation and StaticFiles mount (which requires ../frontend).
    Route handlers close over mock_rag_system so tests can reconfigure it
    between fixture setup and the HTTP call.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from typing import List, Optional

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    _app = FastAPI()

    @_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(**analytics)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        existed = mock_rag_system.session_manager.delete_session(session_id)
        return {"status": "ok", "existed": existed}

    return TestClient(_app)
