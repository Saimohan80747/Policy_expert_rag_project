
from typing import Any
from contextlib import asynccontextmanager
import io

import pandas as pd
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
from router import router_decision

from  supabase_.file_upload import upload_file
from  supabase_.file_download import download_file
from  supabase_.file_preview import preview_file
from  supabase_.chat.session import get_all_sessions,get_session,delete_session
from  supabase_.chat.message import save_chat,get_chat_history
from  supabase_.chat.service import process_chat
from  supabase_.delete_documents  import delete_document

from knowledge_graph.retrive_subgraph_context import graph_answer
from knowledge_graph.db import create_driver
from config import API_CORS_ORIGINS
from knowledge_graph.loader import (
    load_constraints,

    # Node loaders
    load_employee_data,
    load_department_data,
    load_project_data,
    load_intern_data,
    load_subdepartment_data,
    load_teams_data,
    load_policy_data,

    # Relationship loaders
    load_department_subdepartment_relationship,
    load_subdepartment_team_relationship,
    load_team_employee_relationship,
    load_employee_employee_relationship,
    load_intern_employee_relationship,
    load_policy_employee_relationship,
    load_teammate_relationship,
    load_team_project_relationship,

    # # DataFrames
    # employee_data,
    # department_data,
    # project_data,
    # intern_data,
    # subdepartment_data,
    # teams_data,
    # policy_data,

    # department_subdepartment_data,
    # subdepartment_team_data,
    # team_employee_data,
    # employee_employee_data,
    # intern_employee_data,
    # policy_employee_data,
    # teammate_data,
    # team_project_data
)


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
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DataRequest(BaseModel):
    data: list[dict[str, Any]]


class AllNodesRequest(BaseModel):
    employees: list[dict[str, Any]] = Field(default_factory=list)
    departments: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    interns: list[dict[str, Any]] = Field(default_factory=list)
    subdepartments: list[dict[str, Any]] = Field(default_factory=list)
    teams: list[dict[str, Any]] = Field(default_factory=list)
    policies: list[dict[str, Any]] = Field(default_factory=list)


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
    context_preview: str




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
    pipeline_runner = Pipeline(steps=[upload_file,extract_text,clean_text,recursive_chunks,embedding_chunks])
    if(file.filename.endswith('.zip')):
        return zip_file_breaker(file, pipeline_runner)
    result = pipeline_runner.run(file)
    return result


@app.post("/prompt")
def prompt(request: PromptRequest):
    try:
        data = {}

        router_result = router_decision(request.question)
        print(router_result)
        route = router_result["route"]

        if route == "knowledge_graph":
            result = graph_answer(
                driver, request.question, 2
            )

            data.update({
                "query": request.question,
                "knowledge_context": result,
            })

        elif route == "document_retrieval":
            documents = retrieve_documents(request.question)
            memory = get_chat_history(request.session_id)

            data.update({
                "query": request.question,
                "documents": documents,
                "memory": memory
            })

        elif route == "both":
            result = graph_answer(
                driver, request.question, 2
            )

            documents = retrieve_documents(request.question)
            memory = get_chat_history(request.session_id)

            data.update({
                "query": request.question,
                "knowledge_context": result,
                "documents": documents,
                "memory": memory
            })

        else:
            raise ValueError(f"Invalid route: {route}")
        print("Data to be sent to chat_with_cohere:")
        result = chat_with_cohere(data)
        print("Chat result:")
        session_id = process_chat(
            session_id=request.session_id,
            question=request.question,
            answer=result["answer"],
            file_name=result.get("file_name", ""),
            file_url=result.get("file_url", ""),
        )

        return {
            "session_id": session_id,
            "answer": result["answer"],
            "file_name": result.get("file_name", ""),
            "file_url": result.get("file_url", ""),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


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

async def read_csv_file(file: UploadFile):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} is not a CSV file"
        )

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
        return df
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )


# ==================================================
#                    NODE ENDPOINTS
# ==================================================


# 1. EMPLOYEE
@app.post("/nodes/employee")
async def upload_employee(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_employee_data(driver, data)

    return {
        "message": "Employee data uploaded successfully",
        "rows": len(data)
    }


# 2. DEPARTMENT
@app.post("/nodes/department")
async def upload_department(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_department_data(driver, data)

    return {
        "message": "Department data uploaded successfully",
        "rows": len(data)
    }


# 3. PROJECT
@app.post("/nodes/project")
async def upload_project(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_project_data(driver, data)

    return {
        "message": "Project data uploaded successfully",
        "rows": len(data)
    }


# 4. INTERN
@app.post("/nodes/intern")
async def upload_intern(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_intern_data(driver, data)

    return {
        "message": "Intern data uploaded successfully",
        "rows": len(data)
    }


# 5. SUBDEPARTMENT
@app.post("/nodes/subdepartment")
async def upload_subdepartment(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_subdepartment_data(driver, data)

    return {
        "message": "SubDepartment data uploaded successfully",
        "rows": len(data)
    }


# 6. TEAM
@app.post("/nodes/team")
async def upload_team(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_teams_data(driver, data)

    return {
        "message": "Team data uploaded successfully",
        "rows": len(data)
    }


# 7. POLICY
@app.post("/nodes/policy")
async def upload_policy(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_policy_data(driver, data)

    return {
        "message": "Policy data uploaded successfully",
        "rows": len(data)
    }


# # 8. ALL NODES
# @app.post("/nodes/all")
# async def upload_all_nodes(
#     employee_file: UploadFile = File(...),
#     department_file: UploadFile = File(...),
#     project_file: UploadFile = File(...),
#     intern_file: UploadFile = File(...),
#     subdepartment_file: UploadFile = File(...),
#     team_file: UploadFile = File(...),
#     policy_file: UploadFile = File(...)
# ):
#     employee_data = await read_csv_file(employee_file)
#     department_data = await read_csv_file(department_file)
#     project_data = await read_csv_file(project_file)
#     intern_data = await read_csv_file(intern_file)
#     subdepartment_data = await read_csv_file(subdepartment_file)
#     team_data = await read_csv_file(team_file)
#     policy_data = await read_csv_file(policy_file)

#     load_employee_data(driver, employee_data)
#     load_department_data(driver, department_data)
#     load_project_data(driver, project_data)
#     load_intern_data(driver, intern_data)
#     load_subdepartment_data(driver, subdepartment_data)
#     load_teams_data(driver, team_data)
#     load_policy_data(driver, policy_data)

#     return {
#         "message": "All node data uploaded successfully",
#         "employee_rows": len(employee_data),
#         "department_rows": len(department_data),
#         "project_rows": len(project_data),
#         "intern_rows": len(intern_data),
#         "subdepartment_rows": len(subdepartment_data),
#         "team_rows": len(team_data),
#         "policy_rows": len(policy_data)
#     }




@app.post("/relationships/department-subdepartment")
async def upload_department_subdepartment(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_department_subdepartment_relationship(driver, data)

    return {
        "message": "Department-SubDepartment relationships uploaded",
        "rows": len(data)
    }


# 2. SUBDEPARTMENT -> TEAM
@app.post("/relationships/subdepartment-team")
async def upload_subdepartment_team(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_subdepartment_team_relationship(driver, data)

    return {
        "message": "SubDepartment-Team relationships uploaded",
        "rows": len(data)
    }


# 3. TEAM -> EMPLOYEE
@app.post("/relationships/team-employee")
async def upload_team_employee(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_team_employee_relationship(driver, data)

    return {
        "message": "Team-Employee relationships uploaded",
        "rows": len(data)
    }


# 4. EMPLOYEE -> EMPLOYEE
@app.post("/relationships/employee-employee")
async def upload_employee_employee(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_employee_employee_relationship(driver, data)

    return {
        "message": "Employee-Employee relationships uploaded",
        "rows": len(data)
    }


# 5. INTERN -> EMPLOYEE
@app.post("/relationships/intern-employee")
async def upload_intern_employee(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_intern_employee_relationship(driver, data)

    return {
        "message": "Intern-Employee relationships uploaded",
        "rows": len(data)
    }


# 6. POLICY -> EMPLOYEE
@app.post("/relationships/policy-employee")
async def upload_policy_employee(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_policy_employee_relationship(driver, data)

    return {
        "message": "Policy-Employee relationships uploaded",
        "rows": len(data)
    }


# 7. EMPLOYEE -> EMPLOYEE TEAMMATE
@app.post("/relationships/teammate")
async def upload_teammate(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_teammate_relationship(driver, data)

    return {
        "message": "Teammate relationships uploaded",
        "rows": len(data)
    }


# 8. TEAM -> PROJECT
@app.post("/relationships/team-project")
async def upload_team_project(
    file: UploadFile = File(...)
):
    data = await read_csv_file(file)

    load_team_project_relationship(driver, data)

    return {
        "message": "Team-Project relationships uploaded",
        "rows": len(data)
    }