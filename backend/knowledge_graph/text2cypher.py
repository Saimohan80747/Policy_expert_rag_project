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


# ---------------------------- neighbourhoods ----------------------------
NEIGHBOURHOOD_QUERY = """
MATCH (e:Entity {uid: $uid})-[r]-(o:Entity)
WHERE type(r) <> 'EARNS'
WITH type(r) AS rel, startNode(r) = e AS outgoing, [l IN labels(o) WHERE l <> 'Entity'][0] AS label,
     count(*) AS n, sum(CASE WHEN r.valid_to IS NULL THEN 1 ELSE 0 END) AS current
RETURN rel, outgoing, label, n, current
ORDER BY n DESC
LIMIT 14
"""
MAX_NEIGHBOURHOODS = 4
_neighbour_cache = {}


def neighbourhood(linked, run=None) -> str:
    """What is attached to each linked entity (relationship, direction, neighbour label, counts).

    Shown to the model so it writes paths that exist instead of guessing directions and labels.
    """
    run = run or db.run
    blocks = []
    for e in [x for x in linked or [] if x.get("how") in ("alias", "exact")][:MAX_NEIGHBOURHOODS]:
        uid = e["uid"]
        if uid not in _neighbour_cache:
            try:
                _neighbour_cache[uid] = run(NEIGHBOURHOOD_QUERY, uid=uid)
            except Exception:  # noqa: BLE001 - a hint, never a blocker
                _neighbour_cache[uid] = []
        rows = _neighbour_cache[uid]
        if not rows:
            continue
        parts = []
        for r in rows:
            arrow = f"-[:{r['rel']}]->" if r["outgoing"] else f"<-[:{r['rel']}]-"
            parts.append(f"{arrow}({r['label']}) x{r['n']}" + (f" ({r['current']} current)" if r["current"] != r["n"] else ""))
        blocks.append(f"- {e['name']} ({e['label']}, uid={uid}) has: " + "; ".join(parts))
    return "\n".join(blocks)


def reset_neighbourhood_cache():
    _neighbour_cache.clear()


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
- "What the entities are connected to" is listed under each linked entity: use those relationship names, directions
  and labels. Do not invent a relationship that is not in the schema.
- Multi-hop questions: write the WHOLE chain as one path, one relationship per hop, in the order the question
  states it, and end on the entity being asked for. Do not stop one hop early. Resolve phrases like "that client's
  account manager" or "the department of its project manager" by following the glossary below.
- "At the time" / "when it started": take the date of the earlier relationship (r1.valid_from) and require every
  later hop to be valid then: r2.valid_from <= r1.valid_from AND (r2.valid_to IS NULL OR r2.valid_to >= r1.valid_from).
  "Current"/"now": r.valid_to IS NULL on every hop.
- Relationships without dates (for example DESCRIBES) are timeless: do not filter them by valid_from/valid_to, and do
  not add filters the question did not ask for (status, date, doc_type) except the ones it names.
- "How many/which documents or decision records describe X": MATCH (d:Document)-[:DESCRIBES]->(x:Entity {{uid: ...}}),
  with d.doc_type = 'ADR' (or 'RFC', 'EXC') when the question names that kind.

GLOSSARY (source -> target)
  REPORTS_TO: person -> their manager. HEADS: person -> department/team/office they lead.
  MEMBER_OF: person -> team/department. ACCOUNT_MANAGER_OF, CSM_OF: person -> client.
  MANAGES_PROJECT, WORKS_ON: person -> project. DELIVERED_FOR: project -> client.
  OWNS_SERVICE: team -> service. MENTORS: mentor -> mentee. HAS_SKILL: person -> skill.
  OWNED_BY: vendor/policy/budget/objective -> owning department or person. CONTRACTED_WITH: contract -> client/vendor/partner.
  DESCRIBES: document (ADR/RFC/EXC) -> service or policy version. USES_TECH: service -> technology.

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
MATCH (p:Entity {{uid: 'Person:LS00433'}})-[m:MEMBER_OF]->(t:Team)-[:OWNS_SERVICE]->(s:Service)-[:USES_TECH]->(tech:Technology) WHERE m.valid_to IS NULL RETURN t.name AS team, s.name AS service, collect(DISTINCT tech.name) AS technologies
Q: Who manages the customer success manager of Client:CL-001?
MATCH (c:Entity {{uid: 'Client:CL-001'}})<-[r1:CSM_OF]-(csm:Person)-[r2:REPORTS_TO]->(boss:Person) WHERE r1.valid_to IS NULL AND r2.valid_to IS NULL RETURN csm.name AS customer_success_manager, boss.name AS manager
Q: Who headed the department of the manager of Project:PRJ-2012-01 when the project started?
MATCH (pr:Entity {{uid: 'Project:PRJ-2012-01'}})<-[r1:MANAGES_PROJECT]-(pm:Person)-[r2:MEMBER_OF]->(d:Department)<-[r3:HEADS]-(h:Person) WHERE r2.valid_from <= r1.valid_from AND (r2.valid_to IS NULL OR r2.valid_to >= r1.valid_from) AND r3.valid_from <= r1.valid_from AND (r3.valid_to IS NULL OR r3.valid_to >= r1.valid_from) RETURN pm.name AS project_manager, d.name AS department, h.name AS head, r1.valid_from AS project_manager_since
Q: How many architecture decision records describe the service Service:fd-web?
MATCH (d:Document {{doc_type: 'ADR'}})-[:DESCRIBES]->(s:Entity {{uid: 'Service:fd-web'}}) RETURN count(DISTINCT d) AS adr_count, collect(DISTINCT d.id) AS adrs"""


def generate_cypher(question, linked, as_of, schema, feedback=None, run=None, nearby=""):
    entities = "\n".join(f"- {e['name']} ({e['label']}) uid={e['uid']} [{e['how']}]" for e in linked) or "(none linked)"
    prompt = f"Question: {question}\n\nLinked entities:\n{entities}"
    if nearby:
        prompt += f"\n\nWhat the linked entities are connected to (relationship, direction, neighbour label, count):\n{nearby}"
    if feedback:
        prompt += f"\n\nYour previous query did not work: {feedback}\nWrite a corrected query."
    out = llm.generate_json(prompt, CypherQuery, system=SYSTEM.format(schema=schema, as_of=as_of), model=CYPHER_MODEL)
    return out.cypher
