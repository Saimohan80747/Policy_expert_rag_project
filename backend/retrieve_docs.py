import logging

from app_config import PINECONE_NAMESPACE
from embeddings import embed_dense, embed_sparse, get_index
from supabase_.supabase_db import supabase_var

logger = logging.getLogger(__name__)


def hybrid_scale(dense, sparse, alpha):
    """Weight dense vs sparse (alpha=1 -> dense only, 0 -> sparse only) for dotproduct indexes."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return (
        [v * alpha for v in dense],
        {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]},
    )


def _drop_superseded(results):
    """Remove chunks of non-current versions.

    Pinecone applies metadata updates asynchronously, so a just-superseded
    version can still carry is_current=true for a short while. The documents
    table is the source of truth and is consistent immediately.
    """
    doc_ids = sorted({r["doc_id"] for r in results if r.get("doc_id")})
    if not doc_ids:
        return results
    try:
        rows = (
            supabase_var.table("documents")
            .select("doc_id,version")
            .in_("doc_id", doc_ids)
            .eq("is_current", True)
            .execute()
            .data
            or []
        )
    except Exception:  # noqa: BLE001 - never fail a search because of this check
        logger.exception("Could not verify current versions; returning unfiltered results")
        return results
    current = {(row["doc_id"], int(row["version"])) for row in rows}
    return [r for r in results if (r["doc_id"], r["version"]) in current]


def retrieve_documents(query: str, top_k: int = 8, alpha: float = 0.6, current_only: bool = True):
    """Hybrid (dense + sparse) search over chunks. Only current document versions by default."""
    dense = embed_dense([query], input_type="query")[0]
    sparse = embed_sparse([query], input_type="query")[0]
    dense, sparse = hybrid_scale(dense, sparse, alpha)

    response = get_index().query(
        namespace=PINECONE_NAMESPACE,
        top_k=top_k,
        vector=dense,
        sparse_vector=sparse,
        include_metadata=True,
        filter={"is_current": {"$eq": True}} if current_only else None,
    )

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        version = meta.get("version")
        results.append(
            {
                "id": match.id,
                "score": match.score,
                "text": meta.get("text", ""),
                "file_name": meta.get("file_name"),
                "file_url": meta.get("file_url"),
                "doc_id": meta.get("doc_id"),
                "version": int(version) if version is not None else None,
                "chunk_index": meta.get("chunk_index"),
            }
        )
    return _drop_superseded(results) if current_only else results
