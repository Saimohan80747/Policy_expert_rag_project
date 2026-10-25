from .supabase_db import supabase_var


def _delete_from_graph(doc_ids):
    """Best effort: the graph is optional, so problems are reported, not raised."""
    from knowledge_graph import lexical
    from knowledge_graph.db import GraphNotConfigured

    try:
        for doc_id in doc_ids:
            lexical.delete_source_docs(doc_id)
        return "deleted"
    except GraphNotConfigured:
        return "skipped"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {type(exc).__name__}: {exc}"


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
    graph_status = _delete_from_graph(doc_ids)
    storage_result = supabase_var.storage.from_("documents").remove([file_name])
    metadata_result = (
        supabase_var.table("documents").delete().eq("file_name", file_name).execute()
    )

    return {
        "status": "deleted" if rows else "not_found",
        "file_name": file_name,
        "versions_deleted": len(rows),
        "vectors_deleted": vectors_deleted,
        "graph": graph_status,
        "storage": storage_result,
        "metadata": metadata_result.data,
    }
