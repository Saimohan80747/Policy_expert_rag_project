"""Question -> read-only Cypher, with the schema read from the live graph and a strict validator."""
import re
import time

from pydantic import BaseModel

import llm
from app_config import CYPHER_MODEL, GRAPH_MAX_ROWS

from . import db


class CypherQuery(BaseModel):
    cypher: str


# ------------------------------- schema -------------------------------
SCHEMA_QUERIES = {
    "nodes": "MATCH (n:Entity) WITH [l IN labels(n) WHERE l <> 'Entity'][0] AS label, keys(n) AS k "
             "UNWIND k AS key WITH label, collect(DISTINCT key) AS keys RETURN label, keys ORDER BY label",
    "edges": "MATCH (a:Entity)-[r]->(b:Entity) "
             "WITH [l IN labels(a) WHERE l <> 'Entity'][0] AS src, type(r) AS rel, "
             "[l IN labels(b) WHERE l <> 'Entity'][0] AS dst, count(*) AS n "
             "RETURN src, rel, dst, n ORDER BY rel, n DESC",
    "edge_props": "MATCH (:Entity)-[r]->(:Entity) WITH type(r) AS rel, keys(r) AS k UNWIND k AS key "
                  "WITH rel, collect(DISTINCT key) AS keys RETURN rel, keys",
}
HIDDEN_RELS = {"EARNS"}                              # compensation is never exposed
HIDDEN_PROPS = {"eid", "uid_internal"}
_schema_cache = {"at": 0.0, "text": None}


def build_schema(run) -> str:
    nodes, edges, props = (run(SCHEMA_QUERIES[k]) for k in ("nodes", "edges", "edge_props"))
    lines = ["NODES (each also has the label Entity; every node has uid, id, name):"]
    for r in nodes:
        keys = [k for k in r["keys"] if k not in ("uid", "id", "name")]
        lines.append(f"  {r['label']}({', '.join(keys)})")
    by_rel = {}
    for r in edges:
        if r["rel"] not in HIDDEN_RELS:
            by_rel.setdefault(r["rel"], []).append(f"({r['src']})->({r['dst']})")
    edge_keys = {r["rel"]: [k for k in r["keys"] if k not in HIDDEN_PROPS] for r in props}
    lines.append("RELATIONSHIPS (direction matters; props listed after the pairs):")
    for rel, pairs in by_rel.items():
        extra = [k for k in edge_keys.get(rel, []) if k not in ("valid_from", "valid_to")]
        lines.append(f"  [:{rel}] {', '.join(pairs[:6])}" + (f"   props: {', '.join(extra)}" if extra else ""))
    lines.append("Every relationship also has valid_from and valid_to (ISO date strings; valid_to is missing/NULL while current).")
    return "\n".join(lines)


def get_schema(run=None, ttl=900) -> str:
    run = run or db.run
    if _schema_cache["text"] is None or time.time() - _schema_cache["at"] > ttl:
        _schema_cache["text"], _schema_cache["at"] = build_schema(run), time.time()
    return _schema_cache["text"]


def reset_schema_cache():
    _schema_cache["text"], _schema_cache["at"] = None, 0.0


# ------------------------------ validation ------------------------------
class RestrictedQuery(ValueError):
    """The question needs data the assistant must not expose (compensation)."""


FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL|APOC|INDEX|CONSTRAINT|GRANT|DENY|"
    r"REVOKE|SHOW|DBMS|TERMINATE|USE|UNION)\b", re.I)
RESTRICTED = re.compile(r"\b(EARNS|base_amount|variable_pct|confidentiality|salary|compensation)\b", re.I)
INTERNAL = re.compile(r"\b(Alias|GraphMeta)\b|\.eid\b", re.I)
START = re.compile(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b", re.I)
STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def validate_cypher(raw: str) -> str:
    """Return a safe, limited query or raise ValueError / RestrictedQuery."""
    text = re.sub(r"^```[a-z]*\s*|```\s*$", "", (raw or "").strip(), flags=re.I | re.M).strip()
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S).strip().rstrip(";").strip()
    if not text:
        raise ValueError("Empty query.")
    scan = STRING.sub("''", text)
    if ";" in scan:
        raise ValueError("Only one statement is allowed.")
    if not START.match(scan):
        raise ValueError("Query must start with MATCH, WITH, UNWIND or RETURN.")
    if RESTRICTED.search(scan):
        raise RestrictedQuery("That data is restricted.")
    if FORBIDDEN.search(scan) or INTERNAL.search(scan):
        raise ValueError("Query uses a keyword or label that is not allowed.")
    if not re.search(r"\bRETURN\b", scan, re.I):
        raise ValueError("Query must RETURN something.")
    if not re.search(r"\bLIMIT\b", scan, re.I):
        text += f"\nLIMIT {GRAPH_MAX_ROWS}"
    return text


# ------------------------------ generation ------------------------------
SYSTEM = """You translate questions about a company into ONE read-only Cypher query for Neo4j.

{schema}

RULES
- Use only the labels, relationship types and properties listed above. Respect relationship direction.
- Linked entities are given with their uid. Anchor on them: MATCH (x:Entity {{uid: 'Person:LS00001'}}).
  If a name was NOT linked, match by name: WHERE toLower(n.name) CONTAINS 'text'. Never invent ids.
- Today's date for "current" questions is {as_of}. A relationship is current when r.valid_to IS NULL.
  For "in <year>" or "as of <date>": r.valid_from <= 'DATE' AND (r.valid_to IS NULL OR r.valid_to >= 'DATE').
  People who left have status 'former'. Prefer returning dates when the question is about history.
- Return readable columns with aliases (names, ids, dates), not whole nodes. Use DISTINCT where paths can repeat.
- Counting/aggregation is fine (count, collect, ORDER BY). Never write data. One statement, no semicolon.
- Compensation (salary, pay, EARNS) is not available: if asked, return the query MATCH (n) RETURN n.salary.

EXAMPLES
Q: Who does Person:LS00121 report to now?
MATCH (p:Entity {{uid: 'Person:LS00121'}})-[r:REPORTS_TO]->(m:Person) WHERE r.valid_to IS NULL RETURN m.name AS manager, m.id AS id
Q: Which clients are served by the projects Person:LS00121 works on?
MATCH (p:Entity {{uid: 'Person:LS00121'}})-[w:WORKS_ON]->(pr:Project)-[:DELIVERED_FOR]->(c:Client) WHERE w.valid_to IS NULL RETURN DISTINCT c.name AS client, pr.name AS project
Q: How many current members does Department:ENG-PLAT have?
MATCH (p:Person)-[m:MEMBER_OF]->(d:Entity {{uid: 'Department:ENG-PLAT'}}) WHERE m.valid_to IS NULL RETURN count(DISTINCT p) AS headcount
Q: Who reported to Person:LS00001 at the end of 2018?
MATCH (p:Person)-[r:REPORTS_TO]->(b:Entity {{uid: 'Person:LS00001'}}) WHERE r.valid_from <= '2018-12-31' AND (r.valid_to IS NULL OR r.valid_to >= '2018-12-31') RETURN p.name AS person, r.valid_from AS from, r.valid_to AS to
Q: Which services are owned by the team that Person:LS00433 is in, and what technology do they use?
MATCH (p:Entity {{uid: 'Person:LS00433'}})-[m:MEMBER_OF]->(t:Team)-[:OWNS_SERVICE]->(s:Service)-[:USES_TECH]->(tech:Technology) WHERE m.valid_to IS NULL RETURN t.name AS team, s.name AS service, collect(DISTINCT tech.name) AS technologies"""


def generate_cypher(question, linked, as_of, schema, feedback=None, run=None):
    entities = "\n".join(f"- {e['name']} ({e['label']}) uid={e['uid']} [{e['how']}]" for e in linked) or "(none linked)"
    prompt = f"Question: {question}\n\nLinked entities:\n{entities}"
    if feedback:
        prompt += f"\n\nYour previous query did not work: {feedback}\nWrite a corrected query."
    out = llm.generate_json(prompt, CypherQuery, system=SYSTEM.format(schema=schema, as_of=as_of), model=CYPHER_MODEL)
    return out.cypher
