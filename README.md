# First RAG Project

A small RAG app with a FastAPI backend and a Streamlit frontend. It lets you upload a document, ask questions about it, view chat history, and open or download the source file used in the answer.

## Features

- Upload PDF, DOCX, TXT, CSV, XLSX files
- Run the processing pipeline on uploaded files
- Ask questions against the retrieved document context
- View previous chat sessions
- Open or download the source file from an answer
- Preview uploaded files through the backend

## Project Structure

```text
first_rag_project/
├── backend/
│   ├── api.py
│   ├── chunking/
│   ├── data_cleaning.py
│   ├── embeddings.py
│   ├── file_reader.py
│   ├── pipeline.py
│   ├── retrieve_docs.py
│   └── supabase_/
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- A single `backend/.env` file with the required credentials
- A Pinecone API key
- A Cohere API key

## Install

Create and activate a virtual environment, then install the packages for each side:

```bash
cd backend
pip install -r requirements.txt
```

```bash
cd ../frontend
pip install -r requirements.txt
```

## Run

Start the backend:

```bash
cd backend
uvicorn api:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

Backend: `http://127.0.0.1:8000`

Frontend: `http://localhost:8501`

## API Endpoints

- `POST /pipeline` - upload and process a file
- `POST /prompt` - ask a question
- `GET /history` - list chat sessions
- `GET /history/{session_id}` - get session details
- `GET /sessions/{session_id}` - get chat messages for a session
- `DELETE /sessions/{session_id}` - delete a session
- `GET /download/{file_name}` - download a source file
- `GET /preview/{file_name}` - preview a source file

## Notes

- The frontend loads chat messages from `GET /sessions/{session_id}` when you click a session.
- Source files can be opened with `Open source` or downloaded with `Download source`.
