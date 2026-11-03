import hashlib
from .supabase_db import supabase_var
from config import SUPABASE_BUCKET, SUPABASE_DOCUMENTS_TABLE
import re


def upload_file(uploaded_file):
    if uploaded_file is None:
        return {
            "status": "error",
            "message": "No file uploaded."
        }
    file_bytes = uploaded_file.file.read()
    uploaded_file.file.seek(0)

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # check duplicate
    existing = (
        supabase_var.table(SUPABASE_DOCUMENTS_TABLE)
        .select("*")
        .eq("file_hash", file_hash)
        .execute()
    )

    if existing.data:
        return {
            "status": "duplicate",
            "file": existing.data[0]
        }
    
    filename = uploaded_file.filename
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
   
  
    path = filename

    # upload to storage
    supabase_var.storage.from_(SUPABASE_BUCKET).upload(
        path=path,
        file=file_bytes
    )

    url = supabase_var.storage.from_(SUPABASE_BUCKET).get_public_url(path)

    # save metadata
    supabase_var.table(SUPABASE_DOCUMENTS_TABLE).insert({

        "file_name": filename,

        "file_hash": file_hash,

        "file_url": url,

        "file_size": len(file_bytes)

    }).execute()

    return {

        "uploaded_file": uploaded_file,
        "file_bytes": file_bytes,
      
        "status": "uploaded",

        "url": url,
        "file_name": filename,
        "hash": file_hash

    }
