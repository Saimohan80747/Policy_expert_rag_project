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

# --- Graph retrieval ---------------------------------------------------------
CYPHER_MODEL = os.getenv("GEMINI_CYPHER_MODEL", GEMINI_ANSWER_MODEL)
GRAPH_MAX_ROWS = 50
GRAPH_QUERY_TIMEOUT = 15      # seconds
GRAPH_MAX_ATTEMPTS = 3        # query generation attempts (each retry sees the previous error)

# --- Retrieval upgrade ---------------------------------------------------------
GRAPH_CHUNK_CANDIDATES = 10   # chunks that mention the linked entities, added to the vector candidates
GRAPH_CHUNK_WEIGHT = 0.5      # RRF weight of that list relative to vector search (1.0)
RERANK_CANDIDATES = 20        # vector candidates to rerank (only when a Cohere key is set)
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
RERANK_MIN_SCORE = 0.05       # drop reranked chunks scoring below this (the top 2 are always kept)
NEIGHBOUR_SEEDS = 2           # top chunks whose previous/next chunks are added for context
NEIGHBOUR_MAX = 4
RERANK_PER_MINUTE = 8         # Cohere trial allows 10/min; stay under it
RERANK_MONTHLY_LIMIT = 900    # trial allows ~1,000 API calls/month

# --- Chat -------------------------------------------------------------------
RETRIEVE_TOP_K = 8
HISTORY_TURNS = 4          # previous Q/A pairs used to resolve follow-up questions
HISTORY_ANSWER_CHARS = 600

# --- In-document section pull ----------------------------------------------------
DOC_EXPAND_DOCS = 3           # look inside the best N documents
DOC_EXPAND_MAX = 3            # add at most this many extra chunks
DOC_SCAN_CHUNKS = 14          # how many chunks of a document to scan
