try:  # newer PyMuPDF exposes `pymupdf`; older exposes `fitz`
    import pymupdf as fitz
except ImportError:
    import fitz
from .supabase_db import supabase_var
BUCKET = "documents"
import os
import tempfile


def preview_file(file_name: str):
    file_bytes = (
        supabase_var.storage
        .from_(BUCKET)
        .download(file_name)
    )

    temp_path = os.path.join(tempfile.gettempdir(), file_name)

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    doc = fitz.open(temp_path)

    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()

    return {
        "file_name": file_name,
        "text": text
    }