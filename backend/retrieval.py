"""Document retrieval: vector search + entity-anchored graph chunks, fused with RRF, optionally reranked."""
import logging

from app_config import (
    GRAPH_CHUNK_CANDIDATES,
    GRAPH_CHUNK_WEIGHT,
    RERANK_CANDIDATES,
    RETRIEVE_TOP_K,
)
from knowledge_graph import db
from rerank import rerank, rerank_enabled
from retrieve_docs import retrieve_documents

logger = logging.getLogger(__name__)
RRF_K = 60

ENTITY_CHUNKS_QUERY = """
MATCH (c:Chunk {is_current: true})-[m:MENTIONS]->(e:Entity)
WHERE e.uid IN $uids
WITH c, count(DISTINCT e) AS hits, sum(m.count) AS total
WHERE hits >= $min_hits
MATCH (s:SourceDoc)-[:HAS_CHUNK]->(c)
RETURN c.cid AS cid, c.text AS text, c.index AS idx, s.doc_id AS doc_id, s.version AS version,
       s.file_name AS file_name, s.file_url AS file_url
ORDER BY hits DESC, total DESC
LIMIT $limit
"""


def chunk_key(chunk):
    index = chunk.get("chunk_index")
    return (chunk.get("doc_id"), chunk.get("version"), int(index) if index is not None else None)


def entity_chunks(linked, limit=GRAPH_CHUNK_CANDIDATES) -> list[dict]:
    """Current chunks that mention the linked entities (all of them when two or more are linked)."""
    uids = [e["uid"] for e in linked or [] if e.get("how") in ("alias", "exact")]
    if not uids:
        return []
    try:
        rows = db.run(ENTITY_CHUNKS_QUERY, uids=uids, min_hits=min(2, len(uids)), limit=limit)
    except Exception as exc:  # noqa: BLE001 - graph chunks are a bonus, never a blocker
        logger.warning("Entity-anchored chunk lookup failed: %s: %s", type(exc).__name__, exc)
        return []
    return [
        {"id": r["cid"], "score": 0.0, "text": r["text"] or "", "file_name": r["file_name"],
         "file_url": r["file_url"], "doc_id": r["doc_id"], "version": r["version"],
         "chunk_index": r["idx"], "via_graph": True}
        for r in rows
    ]


def fuse(vector, graph, k=RRF_K, graph_weight=GRAPH_CHUNK_WEIGHT) -> list[dict]:
    """Reciprocal rank fusion. A chunk found by both lists is boosted and keeps the vector hit's data."""
    scores, items = {}, {}
    for weight, ranked in ((1.0, vector), (graph_weight, graph)):
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            if key not in items:
                items[key] = dict(chunk)
            elif chunk.get("via_graph"):
                items[key]["via_graph"] = True
    return [items[key] for key in sorted(scores, key=scores.get, reverse=True)]


def retrieve_context(question: str, linked=None, top_k: int = RETRIEVE_TOP_K) -> list[dict]:
    wide = RERANK_CANDIDATES if rerank_enabled() else top_k
    vector = retrieve_documents(question, top_k=wide)
    graph = entity_chunks(linked) if linked else []
    fused = fuse(vector, graph) if graph else vector
    chunks, _ = rerank(question, fused[: max(wide, top_k)], top_k)
    return chunks
