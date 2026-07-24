
from pydantic import BaseModel
from fastapi import FastAPI,UploadFile, File

from  zip_file_breaker import zip_file_breaker
from  data_cleaning import clean_text
from  chunking.text import *
from  file_reader import extract_text
from  embeddings import embedding_chunks
from  pipeline import Pipeline
from  retrieve_docs import retrieve_documents
from  chat_cohere import chat_with_cohere

from  supabase_.file_upload import upload_file
from  supabase_.file_download import download_file
from  supabase_.file_preview import preview_file
from  supabase_.chat.session import get_all_sessions,get_session,delete_session
from  supabase_.chat.message import save_chat,get_chat_history
from  supabase_.chat.service import process_chat

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Chunking API",
    description="API for different text chunking strategies",
    version="1.0.0"
)

class PromptRequest(BaseModel):
    question: str
    session_id: str

@app.post("/pipeline")
def pipeline(file: UploadFile = File(...)):
    pipeline = Pipeline(steps=[upload_file,extract_text,clean_text,recursive_chunks,embedding_chunks])
    if(file.filename.endswith('.zip')):
        return zip_file_breaker(file,pipeline)
    result = pipeline.run(file)
    return result

@app.post("/prompt")
def prompt(request: PromptRequest):

    pipeline = Pipeline(
        steps=[
            retrieve_documents,
            chat_with_cohere
        ]
    )

    result = pipeline.run(request.question)
    
    session_id = process_chat(
        session_id=request.session_id,
        question=request.question,
        answer=result["answer"],
        file_name=result.get("file_name"),
        file_url=result.get("file_url"),
    )

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "file_name": result.get("file_name"),
        "file_url": result.get("file_url"),
    }


@app.get("/history")
def get_history():

    sessions = get_all_sessions()

    return {
        "sessions": sessions
    }



@app.get("/history/{session_id}")
def get_session_by_id(session_id: str):
    session = get_session(session_id)

    if not session:
        return {
            "error": f"No session found with ID: {session_id}"
        }

    return {
        "session": session
    }

@app.get("/sessions/{session_id}")
def get_messages_by_id(session_id: str):
    session = get_chat_history(session_id)

    if not session:
        return {
            "error": f"No session found with ID: {session_id}"
        }

    return {
        "session": session
    }


@app.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str):

    session = delete_session(session_id)

    if not session:
        return {
            "status": "error",
            "message": "Session not found."
        }

    return {
        "status": "success",
        "message": "Chat session deleted successfully."
    }

@app.get("/download/{file_name}")
def download(file_name: str):
    return download_file(file_name)

@app.get("/preview/{file_name}")
def preview(file_name: str):
    return preview_file(file_name)
