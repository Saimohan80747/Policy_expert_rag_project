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


def chunk_from_metadata(vec_id, score, meta):
    version, index = meta.get("version"), meta.get("chunk_index")
    return {
        "id": vec_id,
        "score": score,
        "text": meta.get("text", ""),
        "file_name": meta.get("file_name"),
        "file_url": meta.get("file_url"),
        "doc_id": meta.get("doc_id"),
        "version": int(version) if version is not None else None,
        "chunk_index": int(index) if index is not None else None,
    }


def fetch_chunks(ids):
    """Fetch specific chunks by id (used for neighbours of a hit). Missing ids are skipped."""
    if not ids:
        return []
    response = get_index().fetch(ids=list(ids), namespace=PINECONE_NAMESPACE)
    vectors = getattr(response, "vectors", None) or {}
    return [chunk_from_metadata(vid, 0.0, getattr(v, "metadata", None) or {}) for vid, v in vectors.items()]


def _search_filter(current_only, include_superseded):
    if not current_only:
        return None
    current = {"is_current": {"$eq": True}}
    if include_superseded:
        return current
    return {"$and": [current, {"superseded": {"$ne": True}}]}


def retrieve_documents(query: str, top_k: int = 8, alpha: float = 0.6, current_only: bool = True,
                       include_superseded: bool = False):
    """Hybrid (dense + sparse) search over chunks.

    Only current versions by default, and not documents flagged as replaced by a newer document
    (tools/mark_superseded.py) unless include_superseded is set.
    """
    dense = embed_dense([query], input_type="query")[0]
    sparse = embed_sparse([query], input_type="query")[0]
    dense, sparse = hybrid_scale(dense, sparse, alpha)

    response = get_index().query(
        namespace=PINECONE_NAMESPACE,
        top_k=top_k,
        vector=dense,
        sparse_vector=sparse,
        include_metadata=True,
        filter=_search_filter(current_only, include_superseded),
    )

    results = [chunk_from_metadata(m.id, m.score, m.metadata or {}) for m in response.matches]
    return _drop_superseded(results) if current_only else results
