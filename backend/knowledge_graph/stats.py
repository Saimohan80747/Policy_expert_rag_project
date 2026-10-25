"""Read-only graph summary used by GET /graph/stats."""
from .db import run


def graph_stats() -> dict:
    labels = run("MATCH (n:Entity) UNWIND labels(n) AS l WITH l WHERE l <> 'Entity' "
                 "RETURN l AS label, count(*) AS count ORDER BY count DESC")
    rels = run("MATCH (:Entity)-[r]->(:Entity) RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC")
    aliases = run("MATCH (a:Alias) RETURN count(a) AS c")[0]["c"]
    meta = run("MATCH (m:GraphMeta {name:'domain'}) RETURN m{.*} AS meta")
    count = lambda q: run(q)[0]["c"]  # noqa: E731
    lexical = {
        "source_docs": count("MATCH (d:SourceDoc) RETURN count(d) AS c"),
        "current_source_docs": count("MATCH (d:SourceDoc {is_current: true}) RETURN count(d) AS c"),
        "chunks": count("MATCH (c:Chunk) RETURN count(c) AS c"),
        "chunk_mentions": count("MATCH (:Chunk)-[m:MENTIONS]->(:Entity) RETURN count(m) AS c"),
        "chunks_with_mentions": count("MATCH (c:Chunk)-[:MENTIONS]->(:Entity) RETURN count(DISTINCT c) AS c"),
        "adr_rfc_links": count("MATCH (:SourceDoc)-[r:RECORDS]->(:Document) RETURN count(r) AS c"),
    }
    return {
        "nodes": sum(r["count"] for r in labels),
        "relationships": sum(r["count"] for r in rels),
        "aliases": aliases,
        "labels": labels,
        "relationship_types": rels,
        "meta": meta[0]["meta"] if meta else None,
        "documents": lexical,
    }
