# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install/manage dependencies:**
```bash
uv sync        # install dependencies
uv add <pkg>   # add a new dependency
uv remove <pkg> # remove a dependency
```

Always use `uv` to manage dependencies and run Python — never use `pip` or `python` directly.

**Environment setup** — create `.env` in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

The app is served at `http://localhost:8000`. The FastAPI docs are at `http://localhost:8000/docs`.

## Architecture

This is a full-stack RAG chatbot. The backend is a single FastAPI process that serves both the API and the static frontend.

**Request flow for a user query:**
1. `frontend/script.js` sends `POST /api/query` with `{query, session_id}`
2. `backend/app.py` routes to `RAGSystem.query()`
3. `RAGSystem` fetches conversation history from `SessionManager`, then calls `AIGenerator.generate_response()`
4. `AIGenerator` makes a first Claude API call with the `search_course_content` tool available
5. If Claude calls the tool, `ToolManager` dispatches to `CourseSearchTool`, which runs a semantic search in ChromaDB (`VectorStore`)
6. Results are fed back to Claude in a second API call; the final answer text is returned
7. `RAGSystem` saves the exchange to `SessionManager` and returns `(answer, sources)` to the endpoint

**Document ingestion** (runs at startup via `app.py:startup_event`):
- `DocumentProcessor.process_course_document()` parses `.txt` files from `docs/` expecting the format: `Course Title:`, `Course Link:`, `Course Instructor:` in the first three lines, then `Lesson N: Title` markers
- Text is split into sentence-based chunks (800 chars, 100-char overlap)
- `VectorStore` embeds chunks with `all-MiniLM-L6-v2` (via `sentence-transformers`) and stores them in ChromaDB at `backend/chroma_db/`
- ChromaDB persists across restarts; `clear_existing=False` means re-running startup won't re-index already-stored documents

**Key configuration** (`backend/config.py`):
- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2`
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100
- `MAX_RESULTS`: 5 (search results returned to Claude)
- `MAX_HISTORY`: 2 (conversation exchanges kept per session)
- `CHROMA_PATH`: `./chroma_db` (relative to `backend/`)

**Sessions** are in-memory only — they are lost on server restart.
