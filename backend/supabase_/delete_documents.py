
from .supabase_db import supabase_var


def delete_document(file_name):
    storage_result = supabase_var.storage.from_("documents").remove([file_name])

    metadata_result = (
        supabase_var.table("documents")
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

