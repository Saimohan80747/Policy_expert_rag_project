# Policy Expert RAG

A document-question-answering application with a FastAPI backend and a React/Vite frontend. Upload a supported document, process it into the retrieval pipeline, and ask questions through the chat interface. Chat sessions and source-file actions are backed by Supabase, while the knowledge-graph flow uses Neo4j.

## Features

- Upload PDF, DOCX, TXT, CSV, XLSX, and ZIP files
- Process documents through the backend pipeline
- Ask questions using retrieved document and knowledge-graph context
- View and delete previous chat sessions
- Preview and download source files
- Expose the backend API through FastAPI's interactive documentation

## Project structure

```text
first_rag_project/
├── backend/
│   ├── api.py
│   ├── chunking/
│   ├── knowledge_graph/
│   ├── supabase_/
│   ├── embeddings.py
│   ├── file_reader.py
│   ├── pipeline.py
│   └── retrieve_docs.py
├── frontend/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
└── README.md
```

## Prerequisites

Install the following before setting up the project:

- Python 3.10 or newer
- Node.js 18 or newer and npm
- A running or hosted Neo4j database
- Accounts and API credentials for Cohere, Pinecone, and Supabase

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Saimohan80747/Policy_expert_rag_project.git
cd Policy_expert_rag_project
```

### 2. Create a Python environment

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following once for the current user and then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install backend dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create `backend\.env` from the example file:

```powershell
Copy-Item backend\.env.example backend\.env
```

Open `backend\.env` and provide values for all of the following variables:

```dotenv
COHERE_API_KEY=your_cohere_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
PINECONE_API_KEY=your_pinecone_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Keep `backend\.env` private. It is ignored by Git. The backend loads this file when it starts, and it verifies the Neo4j connection during startup.

### 5. Install frontend dependencies

In a new terminal, from the repository root:

```powershell
cd frontend
npm install
```

## Run the application

Use two terminals from the repository root.

### Terminal 1: start the backend

```powershell
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn api:app --reload
```

The API is available at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` to view the Swagger UI.

### Terminal 2: start the frontend

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

The frontend is configured to call the backend at `http://127.0.0.1:8000`. If the backend runs on a different host or port, update `BASE_URL` in `frontend\src\api.js`.

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/pipeline` | Upload and process a document |
| `POST` | `/ask` | Ask a knowledge-graph question |
| `POST` | `/prompt` | Ask a question in a chat session |
| `GET` | `/history` | List chat sessions |
| `GET` | `/history/{session_id}` | Get a session |
| `GET` | `/sessions/{session_id}` | Get messages for a session |
| `DELETE` | `/sessions/{session_id}` | Delete a chat session |
| `GET` | `/download/{file_name}` | Download a source file |
| `GET` | `/preview/{file_name}` | Preview a source file |
| `DELETE` | `/delete-document/{file_name}` | Delete a source document |

## Troubleshooting

- **`Missing Neo4j credentials in .env`**: confirm that `backend\.env` exists and contains `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.
- **Neo4j connectivity errors**: make sure the database is running, the URI is reachable, and the credentials are correct.
- **Frontend cannot reach the API**: start the backend first and confirm that it is listening on port `8000`.
- **Missing Python module**: activate `.venv` and run `python -m pip install -r requirements.txt` from the repository root.
- **Frontend dependency errors**: remove `frontend\node_modules` only if needed, then run `npm install` again.

## Security

Never commit API keys, database passwords, uploaded documents, or other secrets. Use `backend\.env` for local credentials and keep it out of version control.
