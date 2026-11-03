import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _env(name, default=None):
    return os.getenv(name, default)


COHERE_API_KEY = _env("COHERE_API_KEY")
COHERE_MODEL = _env("COHERE_MODEL", "command-a-03-2025")

SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_KEY = _env("SUPABASE_KEY")
SUPABASE_BUCKET = _env("SUPABASE_BUCKET", "documents")
SUPABASE_DOCUMENTS_TABLE = _env("SUPABASE_DOCUMENTS_TABLE", "documents")

PINECONE_API_KEY = _env("PINECONE_API_KEY")
PINECONE_UPLOAD_INDEX = _env("PINECONE_UPLOAD_INDEX", "sai-vector-index")
PINECONE_RETRIEVAL_INDEX = _env("PINECONE_RETRIEVAL_INDEX", "sai-vector-index")
PINECONE_NAMESPACE = _env("PINECONE_NAMESPACE", "first-namespace")
PINECONE_DENSE_MODEL = _env("PINECONE_DENSE_MODEL", "llama-text-embed-v2")
PINECONE_SPARSE_MODEL = _env(
    "PINECONE_SPARSE_MODEL", "pinecone-sparse-english-v0"
)
PINECONE_DIMENSION = int(_env("PINECONE_DIMENSION", "1024"))
PINECONE_CLOUD = _env("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env("PINECONE_REGION", "us-east-1")

NEO4J_URI = _env("NEO4J_URI")
NEO4J_USER = _env("NEO4J_USER")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD")
NEO4J_VECTOR_INDEX = _env("NEO4J_VECTOR_INDEX", "all_nodes_embedding_index")
NEO4J_VECTOR_DIMENSION = int(_env("NEO4J_VECTOR_DIMENSION", "384"))
NEO4J_MAX_HOPS = int(_env("NEO4J_MAX_HOPS", "2"))
NEO4J_TOP_K = int(_env("NEO4J_TOP_K", "3"))

CHUNK_SIZE = int(_env("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "300"))
RETRIEVAL_TOP_K = int(_env("RETRIEVAL_TOP_K", "5"))

API_CORS_ORIGINS = [
    origin.strip()
    for origin in _env(
        "API_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]