"""Central configuration. Values can be overridden from backend/.env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --- Pinecone ---------------------------------------------------------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "company-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "docs")
DENSE_MODEL = "llama-text-embed-v2"          # 1024-d
SPARSE_MODEL = "pinecone-sparse-english-v0"
EMBED_DIM = 1024
EMBED_BATCH = 96      # max inputs per inference.embed call
UPSERT_BATCH = 100
DELETE_BATCH = 1000

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# --- Ingestion --------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}
STORAGE_BUCKET = "documents"
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
MAX_ZIP_DEPTH = 3

# --- LLM (Gemini) -----------------------------------------------------------
# Cheap/fast model for question analysis and chat titles; stronger one for answers.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-3.8-flash")

# --- Chat -------------------------------------------------------------------
RETRIEVE_TOP_K = 8
HISTORY_TURNS = 4          # previous Q/A pairs used to resolve follow-up questions
HISTORY_ANSWER_CHARS = 600
