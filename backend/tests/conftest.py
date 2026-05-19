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
