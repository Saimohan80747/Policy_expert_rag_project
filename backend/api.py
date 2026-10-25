from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion import ingest_file
from rag import answer_question
from retrieve_docs import retrieve_documents
from supabase_.chat.message import get_chat_history
from supabase_.chat.service import process_chat
from supabase_.chat.session import delete_session, get_all_sessions, get_session
from supabase_.delete_documents import delete_document
from supabase_.file_download import download_file
from supabase_.file_preview import preview_file
from zip_file_breaker import zip_file_breaker

app = FastAPI(
    title="Company Docs RAG API",
    description="Document ingestion, hybrid retrieval and question answering",
    version="1.2.0",
)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Return unexpected errors as JSON *with* CORS headers.

    Starlette runs this outside the CORS middleware, so without the header the browser
    reports a misleading "blocked by CORS policy" instead of the real error.
    """
    headers = {}
    origin = request.headers.get("origin")
    if origin in ALLOWED_ORIGINS:
        headers = {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true"}
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers=headers,
    )


class PromptRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)


def _server_error(exc):
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/pipeline")
def pipeline(file: UploadFile = File(...)):
    """Ingest one document, or every document inside a ZIP."""
    if (file.filename or "").lower().endswith(".zip"):
        return zip_file_breaker(file, ingest_file)
    return ingest_file(file)


@app.get("/search")
def search(q: str, top_k: int = 8, alpha: float = 0.6, current_only: bool = True):
    """Debug endpoint: hybrid retrieval only (no LLM)."""
    try:
        return {"query": q, "results": retrieve_documents(q, top_k, alpha, current_only)}
    except Exception as e:
        raise _server_error(e)


@app.post("/ask")
def ask(req: QuestionRequest):
    """Answer one question without chat history or saving."""
    try:
        return answer_question(req.question)
    except Exception as e:
        raise _server_error(e)


@app.post("/prompt")
def prompt(request: PromptRequest):
    """Answer a question inside a chat session (creates the session on the first message)."""
    try:
        history = get_chat_history(request.session_id) if request.session_id else []
        result = answer_question(request.question, history)
    except Exception as e:
        raise _server_error(e)

    session_id = process_chat(
        session_id=request.session_id or None,
        question=request.question,
        answer=result["answer"],
        file_name=result["file_name"],
        file_url=result["file_url"],
        sources=result["sources"],
    )

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "file_name": result["file_name"],
        "file_url": result["file_url"],
        "sources": result["sources"],
        "route": result["route"],
        "route_used": result["route_used"],
        "standalone_question": result["standalone_question"],
        "graph_debug": result.get("graph_debug"),
    }


@app.get("/history")
def get_history():
    return {"sessions": get_all_sessions()}


@app.get("/history/{session_id}")
def get_session_by_id(session_id: str):
    session = get_session(session_id)
    if not session:
        return {"error": f"No session found with ID: {session_id}"}
    return {"session": session}


@app.get("/sessions/{session_id}")
def get_messages_by_id(session_id: str):
    session = get_chat_history(session_id)
    if not session:
        return {"error": f"No session found with ID: {session_id}"}
    return {"session": session}


@app.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str):
    if not delete_session(session_id):
        return {"status": "error", "message": "Session not found."}
    return {"status": "success", "message": "Chat session deleted successfully."}


@app.get("/download/{file_name}")
def download(file_name: str):
    return download_file(file_name)


@app.get("/preview/{file_name}")
def preview(file_name: str):
    return preview_file(file_name)


@app.get("/graph/stats")
def graph_stats_endpoint():
    """Node, relationship and alias counts of the loaded company graph."""
    try:
        from knowledge_graph.stats import graph_stats

        return graph_stats()
    except Exception as e:
        raise _server_error(e)


@app.delete("/delete-document/{file_name}")
def delete_document_endpoint(file_name: str):
    try:
        return delete_document(file_name)
    except Exception as e:
        raise _server_error(e)
