"""Tier-1 graph loader: structured exports -> Neo4j, no LLM involved.

Reads kg_node.csv, kg_edge.csv and alias.csv from the corpus and writes
  (:Entity:<Label> {uid, id, name, ...props})
  (a)-[:REL {eid, valid_from, valid_to, ...props}]->(b)
  (:Alias {aid, alias, norm, alias_type, ...})-[:ALIAS_OF]->(:Entity)
Every statement is a MERGE, so re-running is safe. Use --reset after the corpus changes.

    python -m knowledge_graph.loader --csv-dir C:\\company_corpus\\world\\csv --dry-run
    python -m knowledge_graph.loader --csv-dir C:\\company_corpus\\world\\csv --reset
"""
import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

BATCH = 1000
LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
REL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
RESERVED = {"uid", "id", "name", "eid", "aid"}


# ------------------------------- pure helpers -------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation: 'Dr. S. Pothineni' -> 'dr s pothineni'."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def uid(label: str, ident: str) -> str:
    return f"{label}:{ident}"


def _scalar(value):
    """Neo4j properties must be primitives; nested values are kept as JSON text."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def flatten(props: dict | None) -> dict:
    return {k: _scalar(v) for k, v in (props or {}).items() if v is not None and v != ""}


def _json(text: str) -> dict:
    return json.loads(text) if text and text.strip() else {}


def _digest(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prepare_nodes(rows: list[dict]) -> dict[str, list[dict]]:
    """Group nodes by label. Raises on a bad label so it can't reach a Cypher string."""
    by_label = defaultdict(list)
    for r in rows:
        label = r["label"]
        if not LABEL_RE.match(label):
            raise ValueError(f"Unsafe node label: {label!r}")
        props = {k: v for k, v in flatten(_json(r["props"])).items() if k not in RESERVED}
        props.update({"uid": uid(label, r["id"]), "id": r["id"], "name": r["name"]})
        by_label[label].append(props)
    return dict(by_label)


def prepare_edges(rows: list[dict], known_uids: set[str]):
    """Group edges by relationship type. Returns (grouped, skipped_dangling, duplicates)."""
    by_rel, seen, skipped, duplicates = defaultdict(list), set(), 0, 0
    for r in rows:
        rel = r["rel"]
        if not REL_RE.match(rel):
            raise ValueError(f"Unsafe relationship type: {rel!r}")
        src, dst = uid(r["src_label"], r["src_id"]), uid(r["dst_label"], r["dst_id"])
        if src not in known_uids or dst not in known_uids:
            skipped += 1
            continue
        props = {k: v for k, v in flatten(_json(r["props"])).items() if k not in RESERVED}
        props["valid_from"] = r["valid_from"] or None
        props["valid_to"] = r["valid_to"] or None
        props = {k: v for k, v in props.items() if v is not None}
        eid = _digest(src, rel, dst, props.get("valid_from", ""), props.get("valid_to", ""),
                      json.dumps(props, sort_keys=True))
        if eid in seen:
            duplicates += 1
            continue
        seen.add(eid)
        props["eid"] = eid
        by_rel[rel].append({"src": src, "dst": dst, "props": props})
    return dict(by_rel), skipped, duplicates


def prepare_aliases(rows: list[dict], known_uids: set[str]):
    out, skipped, seen = [], 0, set()
    for r in rows:
        owner = uid(r["entity_label"], r["entity_id"])
        if owner not in known_uids or not r["alias"].strip():
            skipped += 1
            continue
        aid = _digest(owner, r["alias"], r["alias_type"], r["valid_from"], r["valid_to"])
        if aid in seen:
            continue
        seen.add(aid)
        others = [o for o in (r["ambiguous_with"] or "").split(";") if o]
        out.append({
            "uid": owner, "aid": aid, "alias": r["alias"], "norm": normalize(r["alias"]),
            "alias_type": r["alias_type"], "valid_from": r["valid_from"] or None,
            "valid_to": r["valid_to"] or None, "ambiguous": bool(others), "ambiguous_with": others,
        })
    return out, skipped


# Labels the alias table does not cover: (use name, use id). Order matters; the first claim on a surface wins.
DERIVED = {
    "Technology": (True, False), "ExternalOrg": (True, False), "Service": (True, False),
    "Certification": (True, False), "Policy": (True, True), "PolicyVersion": (False, True),
    "Office": (True, True), "Award": (True, False), "Product": (True, False),
    "Document": (False, True), "Incident": (False, True), "Contract": (False, True),
    "Objective": (False, True), "Budget": (False, True),
}


def derive_aliases(nodes: dict[str, list[dict]], existing: list[dict]) -> list[dict]:
    """Aliases from node names/ids for entity types the alias table leaves out (technologies, orgs...)."""
    taken = {a["norm"] for a in existing}
    out = []
    for label, (use_name, use_id) in DERIVED.items():
        for n in nodes.get(label, []):
            for text, kind, wanted in ((n["name"], "derived_name", use_name), (n["id"], "derived_id", use_id)):
                norm = normalize(text)
                if not wanted or not norm or norm in taken:
                    continue
                taken.add(norm)
                out.append({"uid": n["uid"], "aid": _digest(n["uid"], text, kind), "alias": text, "norm": norm,
                            "alias_type": kind, "valid_from": None, "valid_to": None,
                            "ambiguous": False, "ambiguous_with": []})
    return out


def load_corpus(csv_dir: str) -> dict:
    nodes = prepare_nodes(read_csv(os.path.join(csv_dir, "kg_node.csv")))
    known = {n["uid"] for group in nodes.values() for n in group}
    edges, dangling, duplicates = prepare_edges(read_csv(os.path.join(csv_dir, "kg_edge.csv")), known)
    aliases, alias_skipped = prepare_aliases(read_csv(os.path.join(csv_dir, "alias.csv")), known)
    aliases += derive_aliases(nodes, aliases)
    meta = {r["key"]: r["value"] for r in read_csv(os.path.join(csv_dir, "meta.csv"))} \
        if os.path.exists(os.path.join(csv_dir, "meta.csv")) else {}
    return {"nodes": nodes, "edges": edges, "aliases": aliases, "meta": meta,
            "dangling_edges": dangling, "duplicate_edges": duplicates, "alias_skipped": alias_skipped}


def summary(data: dict) -> dict:
    return {
        "nodes": sum(len(v) for v in data["nodes"].values()),
        "nodes_by_label": {k: len(v) for k, v in sorted(data["nodes"].items())},
        "edges": sum(len(v) for v in data["edges"].values()),
        "edges_by_rel": dict(Counter({k: len(v) for k, v in data["edges"].items()}).most_common()),
        "aliases": len(data["aliases"]),
        "dangling_edges_skipped": data["dangling_edges"],
        "duplicate_edges_merged": data["duplicate_edges"],
        "aliases_skipped": data["alias_skipped"],
    }


# --------------------------------- Neo4j I/O ---------------------------------
def _batches(items):
    size = BATCH
    for i in range(0, len(items), size):
        yield items[i : i + size]


def reset_graph(run):
    """Remove the domain graph (Entity and Alias nodes). Other layers are left alone."""
    run("MATCH (n) WHERE n:Entity OR n:Alias DETACH DELETE n")


def write_graph(data: dict, run) -> None:
    for label, rows in data["nodes"].items():
        query = (f"UNWIND $rows AS row MERGE (n:Entity:{label} {{uid: row.uid}}) "
                 "SET n += row")
        for batch in _batches(rows):
            run(query, rows=batch)

    for rel, rows in data["edges"].items():
        query = ("UNWIND $rows AS row "
                 "MATCH (a:Entity {uid: row.src}) MATCH (b:Entity {uid: row.dst}) "
                 f"MERGE (a)-[r:{rel} {{eid: row.props.eid}}]->(b) "
                 "SET r += row.props")
        for batch in _batches(rows):
            run(query, rows=batch)

    alias_query = (
        "UNWIND $rows AS row MATCH (e:Entity {uid: row.uid}) "
        "MERGE (a:Alias {aid: row.aid}) "
        "SET a.alias = row.alias, a.norm = row.norm, a.alias_type = row.alias_type, "
        "a.valid_from = row.valid_from, a.valid_to = row.valid_to, "
        "a.ambiguous = row.ambiguous, a.ambiguous_with = row.ambiguous_with "
        "MERGE (a)-[:ALIAS_OF]->(e)"
    )
    for batch in _batches(data["aliases"]):
        run(alias_query, rows=batch)

    meta = {**data["meta"], "nodes": summary(data)["nodes"], "edges": summary(data)["edges"]}
    run("MERGE (m:GraphMeta {name: 'domain'}) SET m += $meta, m.loaded_at = toString(datetime())", meta=meta)


def verify(data: dict, run) -> dict:
    """Compare what is in Neo4j with what we meant to write."""
    s = summary(data)
    nodes = run("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    rels = run("MATCH (:Entity)-[r]->(:Entity) WHERE r.eid IS NOT NULL RETURN count(r) AS c")[0]["c"]
    aliases = run("MATCH (a:Alias) RETURN count(a) AS c")[0]["c"]
    result = {"nodes": (nodes, s["nodes"]), "edges": (rels, s["edges"]), "aliases": (aliases, s["aliases"])}
    result["ok"] = all(a == b for a, b in result.values() if isinstance(a, int))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load the structured company graph into Neo4j.")
    parser.add_argument("--csv-dir", default=os.getenv("CORPUS_CSV_DIR"), help="folder with kg_node.csv, kg_edge.csv, alias.csv")
    parser.add_argument("--reset", action="store_true", help="delete the existing domain graph first")
    parser.add_argument("--dry-run", action="store_true", help="parse and validate only; no Neo4j needed")
    args = parser.parse_args(argv)
    if not args.csv_dir:
        parser.error("--csv-dir is required (or set CORPUS_CSV_DIR)")

    data = load_corpus(args.csv_dir)
    print(json.dumps(summary(data), indent=2))
    if args.dry_run:
        print("Dry run only. Nothing was written.")
        return 0

    from knowledge_graph.db import close_driver, run
    from knowledge_graph.schema import ensure_schema

    try:
        if args.reset:
            print("Resetting domain graph...")
            reset_graph(run)
        ensure_schema(run)
        print("Writing nodes, edges and aliases...")
        write_graph(data, run)
        result = verify(data, run)
    finally:
        close_driver()
    for name in ("nodes", "edges", "aliases"):
        found, expected = result[name]
        print(f"{name:8} in Neo4j {found:>7}   expected {expected:>7}")
    print("OK" if result["ok"] else "MISMATCH - run again with --reset")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
