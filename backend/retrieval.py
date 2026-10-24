"""Document retrieval: vector search + entity-anchored graph chunks, fused with RRF, optionally reranked."""
import logging
import math
import re

from app_config import (
    DOC_EXPAND_DOCS,
    DOC_EXPAND_MAX,
    DOC_SCAN_CHUNKS,
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


STOPWORDS = frozenset(
    "a an and are as at be by can did do does for from has have how if in is it its of on or that the this to was "
    "were what when where which who whom whose why will with under about stated value according state say says".split())


def tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", (text or "").lower()):
        if t in STOPWORDS or len(t) < 2:
            continue
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t)
    return out


def expand_documents(question, chunks, docs=DOC_EXPAND_DOCS, limit=DOC_EXPAND_MAX, scan=DOC_SCAN_CHUNKS) -> list[dict]:
    """Pull the sections of the best documents that match the question's rarer words.

    Search finds the right document but not always the right section of a long one (a value in clause 6.2 while
    the best-scoring chunk is the header). Inside each document, words that few of its chunks contain count more.
    """
    wanted_terms = set(tokens(question))
    if not wanted_terms:
        return []
    have = {chunk_key(c) for c in chunks}
    top, ids = [], []
    for c in chunks:
        ref = (c.get("doc_id"), c.get("version"))
        if None in ref or ref in top:
            continue
        top.append(ref)
        if len(top) >= docs:
            break
    for doc_id, version in top:
        ids += [f"{doc_id}:v{version}:{i}" for i in range(scan) if (doc_id, version, i) not in have]
    if not ids:
        return []
    try:
        found = fetch_chunks(ids)
    except Exception as exc:  # noqa: BLE001 - context expansion is a bonus
        logger.warning("Document expansion fetch failed: %s: %s", type(exc).__name__, exc)
        return []
    scored = []
    for doc_id, version in top:
        pool = [c for c in found + [c for c in chunks if chunk_key(c)[:2] == (doc_id, version)]
                if chunk_key(c)[:2] == (doc_id, version) and c.get("text")]
        pool = list({chunk_key(c): c for c in pool}.values())
        bags = {chunk_key(c): set(tokens(c["text"])) for c in pool}
        df = {t: sum(t in b for b in bags.values()) for t in wanted_terms}
        for c in pool:
            key = chunk_key(c)
            if key in have:
                continue
            hit = [t for t in wanted_terms if t in bags[key]]
            weight = sum(math.log(1 + len(pool) / df[t]) for t in hit)
            rare = any(df[t] <= 2 for t in hit)
            if hit and (len(hit) >= 2 or rare):
                scored.append((weight * (1.0 if rare else 0.5), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(c, expanded=True) for _, c in scored[:limit]]


def retrieve_context(question: str, linked=None, top_k: int = RETRIEVE_TOP_K, include_history: bool = False,
                     anchored: bool = True, use_neighbours: bool = True, use_rerank: bool = True,
                     use_expand: bool = True) -> list[dict]:
    """The three switches exist so the evaluation can ablate each step; the app uses the defaults."""
    use_rerank = use_rerank and rerank_enabled()
    wide = RERANK_CANDIDATES if use_rerank else top_k
    vector = retrieve_documents(question, top_k=wide, include_superseded=include_history)
    if not vector and not include_history:  # only superseded documents match: better an old answer than none
        vector = retrieve_documents(question, top_k=wide, include_superseded=True)
    graph = entity_chunks(linked, include_history=include_history) if (linked and anchored) else []
    fused = fuse(vector, graph) if graph else vector
    if use_rerank:
        chunks, reranked = rerank(question, fused[: max(wide, top_k)], top_k)
    else:
        chunks, reranked = fused[:top_k], False
    if reranked:  # scores are meaningful: drop clearly irrelevant passages, but keep the best two
        chunks = [c for i, c in enumerate(chunks) if i < 2 or c["score"] >= RERANK_MIN_SCORE]
    extra = expand_documents(question, chunks) if use_expand else []
    near = neighbours(chunks) if use_neighbours else []
    seen = {chunk_key(c) for c in chunks}
    out = list(chunks)
    for c in extra + near:
        if chunk_key(c) not in seen:
            seen.add(chunk_key(c))
            out.append(c)
    return out


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
