"""Lexical layer: SourceDoc and Chunk nodes, linked to the domain graph.

(:SourceDoc)-[:HAS_CHUNK]->(:Chunk)-[:NEXT]->(:Chunk)
(:Chunk)-[:MENTIONS {count, via}]->(:Entity)        found by the gazetteer (tier 2)
(:SourceDoc)-[:RECORDS]->(:Document)                ADR/RFC/exception files and their record
Chunk text lives here too, so graph retrieval can answer without a second lookup.
"""
import re
from datetime import datetime, timezone

from . import db
from .gazetteer import get_gazetteer

REF_RE = re.compile(r"^(?P<ref>[^_]+)__")
CANDIDATE_RE = re.compile(r"([A-Za-z]{2,5})-(\d{2,4}(?:-\d{2,4})?)")


def doc_ref(file_name: str):
    m = REF_RE.match(file_name or "")
    return m.group("ref") if m else None


def record_candidates(file_name: str) -> list[str]:
    """'LS-ADR-2023-0001__adr-0040-...' -> ['ADR-0040', ...] (kept only if such a Document exists)."""
    return sorted({f"{a}-{b}".upper() for a, b in CANDIDATE_RE.findall(file_name or "")})


def chunk_id(doc_id, version, index):
    return f"{doc_id}:v{version}:{index}"


def sync_document(state: dict, run=None) -> dict:
    """Write one ingested document to the graph. Raises GraphNotConfigured when there is no Neo4j."""
    run = run or db.run
    gaz = get_gazetteer(run)  # first call: fails fast when the graph is not configured
    doc_id, version, file_name = state["doc_id"], state["version"], state["file_name"]
    sid = f"{doc_id}:v{version}"

    rows, mention_rows = [], []
    for i, text in enumerate(state["chunks"]):
        cid = chunk_id(doc_id, version, i)
        mentions, unresolved = gaz.find(text)
        rows.append({"cid": cid, "doc_id": doc_id, "version": version, "index": i, "text": text,
                     "file_name": file_name, "is_current": True, "unresolved": unresolved})
        mention_rows += [{**m, "cid": cid} for m in mentions]

    run("MERGE (d:SourceDoc {sid: $sid}) SET d += $props",
        sid=sid, props={"doc_id": doc_id, "version": version, "file_name": file_name,
                        "file_url": state.get("url") or "", "ref": doc_ref(file_name),
                        "doc_type": (state.get("extension") or "").lstrip("."), "is_current": True,
                        "chunk_count": len(rows),
                        "ingested_at": datetime.now(timezone.utc).isoformat()})
    for start in range(0, len(rows), 200):
        run("UNWIND $rows AS row MATCH (d:SourceDoc {sid: $sid}) "
            "MERGE (c:Chunk {cid: row.cid}) SET c += row MERGE (d)-[:HAS_CHUNK]->(c)",
            rows=rows[start : start + 200], sid=sid)
    pairs = [{"a": rows[i]["cid"], "b": rows[i + 1]["cid"]} for i in range(len(rows) - 1)]
    if pairs:
        run("UNWIND $pairs AS p MATCH (a:Chunk {cid: p.a}) MATCH (b:Chunk {cid: p.b}) MERGE (a)-[:NEXT]->(b)",
            pairs=pairs)
    for start in range(0, len(mention_rows), 500):
        run("UNWIND $rows AS row MATCH (c:Chunk {cid: row.cid}) MATCH (e:Entity {uid: row.uid}) "
            "MERGE (c)-[m:MENTIONS]->(e) SET m.count = row.count, m.via = row.via",
            rows=mention_rows[start : start + 500])

    linked = run("MATCH (d:SourceDoc {sid: $sid}) MATCH (x:Document) WHERE x.id IN $cands "
                 "MERGE (d)-[:RECORDS]->(x) RETURN x.id AS id",
                 sid=sid, cands=record_candidates(file_name)) or []

    if state.get("supersedes"):
        retire_older(doc_id, version, run)

    return {"chunks": len(rows), "mentions": len(mention_rows),
            "entities": len({m["uid"] for m in mention_rows}), "records": [r["id"] for r in linked]}


def retire_older(doc_id: str, keep_version: int, run=None):
    run = run or db.run
    run("MATCH (d:SourceDoc {doc_id: $doc_id}) WHERE d.version < $v "
        "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c) SET d.is_current = false, c.is_current = false",
        doc_id=doc_id, v=keep_version)


def delete_source_docs(doc_id: str, run=None):
    """Remove every version of a document (SourceDoc and its Chunks) from the graph."""
    run = run or db.run
    run("MATCH (d:SourceDoc {doc_id: $doc_id}) OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c) DETACH DELETE c, d",
        doc_id=doc_id)
