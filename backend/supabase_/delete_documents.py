from .supabase_db import supabase_var


def delete_document(file_name):
    """Remove a document everywhere: stored file, metadata rows (all versions) and vectors."""
    from embeddings import delete_doc_vectors  # imported lazily: needs Pinecone credentials

    rows = (
        supabase_var.table("documents")
        .select("doc_id")
        .eq("file_name", file_name)
        .execute()
        .data
        or []
    )
    doc_ids = {row["doc_id"] for row in rows if row.get("doc_id")}

    vectors_deleted = sum(delete_doc_vectors(doc_id) for doc_id in doc_ids)
    storage_result = supabase_var.storage.from_("documents").remove([file_name])
    metadata_result = (
        supabase_var.table("documents").delete().eq("file_name", file_name).execute()
    )

    return {
        "status": "deleted" if rows else "not_found",
        "file_name": file_name,
        "versions_deleted": len(rows),
        "vectors_deleted": vectors_deleted,
        "storage": storage_result,
        "metadata": metadata_result.data,
    }
