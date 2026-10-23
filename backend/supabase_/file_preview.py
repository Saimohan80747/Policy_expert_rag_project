import os

from fastapi import HTTPException

from file_reader import extract_text

from .supabase_db import supabase_var

BUCKET = "documents"


def preview_file(file_name: str):
    """Text preview of a stored document, extracted the same way as for indexing."""
    try:
        file_bytes = supabase_var.storage.from_(BUCKET).download(file_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Unable to load file: {exc}")

    state = {
        "file_name": file_name,
        "extension": os.path.splitext(file_name)[1].lower(),
        "file_bytes": file_bytes,
    }
    try:
        result = extract_text(state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}")
    if result.get("status") == "failed":
        raise HTTPException(status_code=422, detail=result["error"])

    return {"file_name": file_name, "text": result["text"]}
