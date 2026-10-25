import hashlib
import mimetypes
import os
import re

from app_config import STORAGE_BUCKET, SUPPORTED_EXTENSIONS
from pipeline import fail

from .supabase_db import supabase_var

BUCKET = STORAGE_BUCKET


def safe_filename(name):
    """Storage-safe file name (ASCII letters, digits, . _ -)."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    if len(base) > 120:  # shorten the name, never the extension
        stem, ext = os.path.splitext(base)
        base = stem[: 120 - len(ext)] + ext
    return base or "unnamed"


def make_doc_id(file_name):
    """Stable identity of a document across versions: same file name = same doc."""
    return hashlib.sha1(file_name.lower().encode("utf-8")).hexdigest()[:12]


def _find_by_hash(file_hash):
    rows = (
        supabase_var.table("documents")
        .select("file_name,doc_id,version")
        .eq("file_hash", file_hash)
        .limit(1)
        .execute()
        .data
    )
    return rows[0] if rows else None


def _latest_version(doc_id):
    rows = (
        supabase_var.table("documents")
        .select("version")
        .eq("doc_id", doc_id)
        .order("version", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return int(rows[0]["version"]) if rows else 0


def upload_file(uploaded_file):
    """Step 1: identify the file. No side effects.

    Outcomes: 'failed' (unsupported/empty), 'duplicate' (same bytes already
    ingested), otherwise a pending state with doc_id and version. A file with
    the same name but different bytes becomes the next version of that doc.
    """
    if uploaded_file is None:
        return {"status": "failed", "error": "No file uploaded."}

    original_name = uploaded_file.filename or ""
    file_name = safe_filename(original_name)
    extension = os.path.splitext(file_name)[1].lower()

    uploaded_file.file.seek(0)
    file_bytes = uploaded_file.file.read()
    uploaded_file.file.seek(0)

    state = {
        "file_name": file_name,
        "original_name": original_name,
        "extension": extension,
        "file_size": len(file_bytes),
    }

    if extension not in SUPPORTED_EXTENSIONS:
        return fail(state, f"Unsupported file type '{extension or 'none'}'.", "upload_file")
    if not file_bytes:
        return fail(state, "File is empty.", "upload_file")

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    state["hash"] = file_hash

    existing = _find_by_hash(file_hash)
    if existing:
        state.update(
            status="duplicate",
            duplicate_of=existing["file_name"],
            doc_id=existing.get("doc_id"),
            version=existing.get("version"),
        )
        return state

    doc_id = make_doc_id(file_name)
    latest = _latest_version(doc_id)
    state.update(
        file_bytes=file_bytes,
        doc_id=doc_id,
        version=latest + 1,
        supersedes=latest or None,
        status="pending",
        url=supabase_var.storage.from_(BUCKET).get_public_url(file_name),
    )
    return state


def finalize_upload(state):
    """Last step: store the original file, record metadata, retire old version rows."""
    file_name = state["file_name"]
    mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

    supabase_var.storage.from_(BUCKET).upload(
        path=file_name,
        file=state["file_bytes"],
        file_options={"content-type": mime, "upsert": "true"},
    )
    url = supabase_var.storage.from_(BUCKET).get_public_url(file_name)

    supabase_var.table("documents").insert(
        {
            "file_name": file_name,
            "file_hash": state["hash"],
            "file_url": url,
            "file_size": state["file_size"],
            "doc_id": state["doc_id"],
            "version": state["version"],
            "is_current": True,
            "doc_type": state["extension"].lstrip("."),
            "chunk_count": len(state.get("chunks", [])),
        }
    ).execute()

    if state.get("supersedes"):
        (
            supabase_var.table("documents")
            .update({"is_current": False})
            .eq("doc_id", state["doc_id"])
            .lt("version", state["version"])
            .execute()
        )

    state["url"] = url
    return state
