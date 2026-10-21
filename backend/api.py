
from pydantic import BaseModel,Field
from fastapi import FastAPI, UploadFile, File, HTTPException,APIRouter
from  fastapi.middleware.cors import CORSMiddleware

from  zip_file_breaker import zip_file_breaker
from  data_cleaning import clean_text
from  chunking.text import *
from  file_reader import extract_text
from  embeddings import embedding_chunks
from  pipeline import Pipeline
from  retrieve_docs import retrieve_documents
from  chat_cohere import chat_with_cohere
from  ner import extract_entities

from  supabase_.file_upload import upload_file
from  supabase_.file_download import download_file
from  supabase_.file_preview import preview_file
from  supabase_.chat.session import get_all_sessions,get_session,delete_session
from  supabase_.chat.message import save_chat,get_chat_history
from  supabase_.chat.service import process_chat
from  supabase_.delete_documents  import delete_document

from knowledge_graph.retrive_subgraph_context import graph_answer
from knowledge_graph.db import create_driver

driver=create_driver()

import spacy
# nlp = spacy.load("en_core_web_sm")
app = FastAPI(
    title="Chunking API",
    description="API for different text chunking strategies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    question: str
    session_id: str

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, example="Which films did Aamir Khan and AR Rahman work on together?")
    top_k:    int = Field(default=3, ge=1, le=10, description="Number of starting nodes for graph traversal")
    hops:     int = Field(default=2, ge=1, le=4,  description="Traversal depth from each starting node")


class QuestionResponse(BaseModel):
    question:        str
    answer:          str
    retrieved_nodes: list[dict]
    context_preview: str          # First 500 chars of context, for transparency


class CypherRequest(BaseModel):
    query: str = Field(..., example="MATCH (m:Movie) RETURN m.title LIMIT 5")

def _public_upload_result(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"uploaded_file", "file_bytes"}
    }

@app.post("/ask")
def ask(req: QuestionRequest):
    """
    Answer a natural language question about Bollywood using GraphRAG.

    The pipeline:
    1. Embed the question and find the top_k most relevant graph nodes
    2. Traverse each node's neighbourhood up to `hops` relationship steps
    3. Assemble the subgraph as structured text context
    4. Call GPT-4o to generate a natural language answer from that context
    """
    try:
        result = graph_answer(
            driver,req.question,2
        )
        data={
            "query": req.question,
            "knowledge_context": result,
        }
        return chat_with_cohere(data)
        # return QuestionResponse(
        #     question=req.question,
        #     answer=result["answer"],
        #     retrieved_nodes=result["retrieved_nodes"],
        #     context_preview=result["context"][:500],
        # )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
def pipeline(file: UploadFile = File(...)):
    pipeline_runner = Pipeline(steps=[upload_file,extract_text])
    if(file.filename.endswith('.zip')):
        return zip_file_breaker(file,pipeline)
    result = pipeline_runner.run(file)
    return result

@app.post("/prompt")
def prompt(request: PromptRequest):
    try:
            result = graph_answer(
                driver,request.question,2
            )
            data={
                "query": request.question,
                "knowledge_context": result,
            }
            result = chat_with_cohere(data)
            # return QuestionResponse(
            #     question=req.question,
            #     answer=result["answer"],
            #     retrieved_nodes=result["retrieved_nodes"],
            #     context_preview=result["context"][:500],
            # )
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # documents = retrieve_documents(request.question)
    # memory=get_chat_history(request.session_id)
    # result = chat_with_cohere({
    #     "query": request.question,
    #     "documents": documents,
    #     "memory": memory
    # })
    # # pipeline = Pipeline(
    # #     steps=[
    # #         retrieve_documents,
            
    # #         chat_with_cohere
    # #     ]
    # # )

    # result = pipeline.run(request.question)
    
    session_id = process_chat(
        session_id=request.session_id,
        question=request.question,
        answer=result["answer"],
        file_name=result.get("file_name",""),
        file_url=result.get("file_url",""),
    )

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "file_name": result.get("file_name",""),
        "file_url": result.get("file_url",""),
    }
# @app.post("/build_graph")
# def build_graph(request: PromptRequest):


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

@app.delete("/delete-document/{file_name}")
def delete_document_endpoint(file_name: str):
    return delete_document(file_name)


# @app.post("/pipeline/upload")
# def pipeline_upload(file: UploadFile = File(...)):
#     return _public_upload_result(upload_file(file))


# @app.post("/pipeline/extract")
# def pipeline_extract(file: UploadFile = File(...)):
#     file_bytes = file.file.read()
#     return extract_text({
#         "uploaded_file": file,
#         "file_bytes": file_bytes,
#         "file_name": file.filename or "",
#     })


# @app.post("/pipeline/clean")
# def pipeline_clean(payload: dict):
#     return clean_text(payload)


# @app.post("/pipeline/chunk")
# def pipeline_chunk(payload: dict, chunk_size: int = 2000):
#     if chunk_size <= 0:
#         raise HTTPException(status_code=400, detail="chunk_size must be greater than zero")
#     return recursive_chunks(payload, chunk_size=chunk_size)


# @app.post("/pipeline/embed")
# def pipeline_embed(payload: dict):
#     return embedding_chunks(payload)
# @app.get("/extract")
# def extract(text: str):
#     doc = nlp(text)

#     entities = set()

#     for ent in doc.ents:
#         entities.add((ent.text, ent.label_))

#     return {
#         "entities": [
#             {"text": text, "label": label}
#             for text, label in entities
#         ]
#     }

# @app.post("/upload_file")
# def upload(file: UploadFile = File(...)):
#     return extract_text(file)