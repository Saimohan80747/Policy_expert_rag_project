"""Document retrieval: vector search + entity-anchored graph chunks, fused with RRF, optionally reranked."""
import logging

from app_config import (
    GRAPH_CHUNK_CANDIDATES,
    GRAPH_CHUNK_WEIGHT,
    NEIGHBOUR_MAX,
    NEIGHBOUR_SEEDS,
    RERANK_MIN_SCORE,
    RERANK_CANDIDATES,
    RETRIEVE_TOP_K,
)
from knowledge_graph import db
from rerank import rerank, rerank_enabled
from retrieve_docs import fetch_chunks, retrieve_documents

logger = logging.getLogger(__name__)
RRF_K = 60

ENTITY_CHUNKS_QUERY = """
MATCH (c:Chunk {is_current: true})-[m:MENTIONS]->(e:Entity)
WHERE e.uid IN $uids
WITH c, count(DISTINCT e) AS hits, sum(m.count) AS total
WHERE hits >= $min_hits AND ($history OR coalesce(c.superseded, false) = false)
MATCH (s:SourceDoc)-[:HAS_CHUNK]->(c)
RETURN c.cid AS cid, c.text AS text, c.index AS idx, s.doc_id AS doc_id, s.version AS version,
       s.file_name AS file_name, s.file_url AS file_url
ORDER BY hits DESC, total DESC
LIMIT $limit
"""


def chunk_key(chunk):
    index = chunk.get("chunk_index")
    return (chunk.get("doc_id"), chunk.get("version"), int(index) if index is not None else None)


def entity_chunks(linked, limit=GRAPH_CHUNK_CANDIDATES, include_history=False) -> list[dict]:
    """Current chunks that mention the linked entities (all of them when two or more are linked)."""
    uids = [e["uid"] for e in linked or [] if e.get("how") in ("alias", "exact")]
    if not uids:
        return []
    try:
        rows = db.run(ENTITY_CHUNKS_QUERY, uids=uids, min_hits=min(2, len(uids)), limit=limit, history=include_history)
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


def retrieve_context(question: str, linked=None, top_k: int = RETRIEVE_TOP_K, include_history: bool = False) -> list[dict]:
    wide = RERANK_CANDIDATES if rerank_enabled() else top_k
    vector = retrieve_documents(question, top_k=wide, include_superseded=include_history)
    if not vector and not include_history:  # only superseded documents match: better an old answer than none
        vector = retrieve_documents(question, top_k=wide, include_superseded=True)
    graph = entity_chunks(linked, include_history=include_history) if linked else []
    fused = fuse(vector, graph) if graph else vector
    chunks, reranked = rerank(question, fused[: max(wide, top_k)], top_k)
    if reranked:  # scores are meaningful: drop clearly irrelevant passages, but keep the best two
        chunks = [c for i, c in enumerate(chunks) if i < 2 or c["score"] >= RERANK_MIN_SCORE]
    return chunks + neighbours(chunks)


def neighbours(chunks, seeds=NEIGHBOUR_SEEDS, limit=NEIGHBOUR_MAX) -> list[dict]:
    """The chunks just before and after the best hits: answers often sit in the next section."""
    have = {chunk_key(c) for c in chunks}
    wanted = []
    for c in chunks[:seeds]:
        index = c.get("chunk_index")
        if c.get("doc_id") is None or c.get("version") is None or index is None:
            continue
        for step in (1, -1):
            if index + step >= 0 and (c["doc_id"], c["version"], index + step) not in have:
                have.add((c["doc_id"], c["version"], index + step))
                wanted.append(f"{c['doc_id']}:v{c['version']}:{index + step}")
    try:
        found = fetch_chunks(wanted[:limit])
    except Exception as exc:  # noqa: BLE001 - context expansion is a bonus
        logger.warning("Neighbour chunk fetch failed: %s: %s", type(exc).__name__, exc)
        return []
    found = [c for c in found if c["text"]]
    return sorted(found, key=lambda c: (c["doc_id"], c["chunk_index"]))
