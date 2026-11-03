
from .supabase_db import supabase_var
from config import SUPABASE_BUCKET, SUPABASE_DOCUMENTS_TABLE


def delete_document(file_name):
    storage_result = supabase_var.storage.from_(SUPABASE_BUCKET).remove([file_name])

    metadata_result = (
        supabase_var.table(SUPABASE_DOCUMENTS_TABLE)
        .delete()
        .eq("file_name", file_name)
        .execute()
    )

    return {
        "status": "deleted",
        "file_name": file_name,
        "storage": storage_result,
        "metadata": metadata_result.data,
    }
