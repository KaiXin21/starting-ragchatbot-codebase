"""Shared mock object builders for the test suite."""
import sys
import os

# Ensure backend/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
from vector_store import SearchResults


def make_text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_tool_use_block(tool_id: str, name: str, input_dict: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_dict
    return b


def make_api_response(content_blocks, stop_reason: str = "end_turn"):
    r = MagicMock()
    r.content = content_blocks
    r.stop_reason = stop_reason
    return r


def make_search_results(docs, metas, distances=None, error=None):
    if error:
        return SearchResults.empty(error)
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances if distances is not None else [0.1] * len(docs),
    )
