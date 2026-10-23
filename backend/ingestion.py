"""Document ingestion: upload -> extract -> clean -> chunk -> embed -> store."""
import embeddings
from app_config import CHUNK_OVERLAP, CHUNK_SIZE
from chunking.text import recursive_chunks
from data_cleaning import clean_text
from embeddings import embedding_chunks
from file_reader import extract_text
from pipeline import Pipeline, fail
from supabase_.file_upload import finalize_upload, upload_file


def chunk_text(state):
    state = recursive_chunks(state, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    state["chunks"] = [c.strip() for c in state["chunks"] if c.strip()]
    if not state["chunks"]:
        return fail(state, "Text produced no chunks.", "chunk_text")
    return state


def store_document(state):
    """Record the document, then hide older versions from search.

    If recording fails, the vectors just written are removed so search never
    returns chunks of a document that is not registered.
    """
    try:
        finalize_upload(state)
    except Exception:
        embeddings.cleanup_version(state["doc_id"], state["version"])
        raise

    if state.get("supersedes"):
        try:
            embeddings.retire_older_versions(state["doc_id"], state["version"])
        except Exception as exc:  # noqa: BLE001 - document is stored; surface as a warning
            state.setdefault("warnings", []).append(
                f"Could not retire older version vectors: {exc}"
            )
    return state


INGEST_STEPS = [
    upload_file,
    extract_text,
    clean_text,
    chunk_text,
    embedding_chunks,
    store_document,
]


def public_result(state):
    """Small JSON-safe summary of an ingestion run (no bytes, text or chunks)."""
    result = {
        "status": state.get("status"),
        "file_name": state.get("file_name"),
        "doc_id": state.get("doc_id"),
        "version": state.get("version"),
    }
    for key in ("supersedes", "duplicate_of", "hash", "url", "step", "error", "warnings"):
        if state.get(key):
            result[key] = state[key]
    if state.get("embedded_chunks") is not None:
        result["chunks"] = state["embedded_chunks"]
    return result


def ingest_file(uploaded_file):
    state = Pipeline(INGEST_STEPS).run(uploaded_file)
    if state.get("status") == "pending":
        state["status"] = "ingested"
    return public_result(state)
