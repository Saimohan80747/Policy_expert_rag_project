"""Entity linking: question text -> graph entities, using the alias table. No LLM."""
import re

from . import db
from .gazetteer import get_gazetteer
from .loader import normalize

EXACT_QUERY = (
    "MATCH (a:Alias {norm: $norm})-[:ALIAS_OF]->(e:Entity) "
    "RETURN e.uid AS uid, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, "
    "a.ambiguous AS ambiguous LIMIT 5"
)
FUZZY_QUERY = (
    "CALL db.index.fulltext.queryNodes('alias_ft', $q) YIELD node, score "
    "MATCH (node)-[:ALIAS_OF]->(e:Entity) "
    "RETURN e.uid AS uid, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, score "
    "ORDER BY score DESC LIMIT 3"
)
NAMES_QUERY = (
    "MATCH (e:Entity) WHERE e.uid IN $uids "
    "RETURN e.uid AS uid, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label"
)
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def fuzzy_query(surface: str) -> str:
    tokens = [LUCENE_SPECIAL.sub(" ", t) for t in surface.split()]
    return " ".join(f"{t.strip()}~" for t in tokens if len(t.strip()) > 2)


def link_entities(question: str, surfaces: list[str] | None = None, run=None, limit: int = 8) -> list[dict]:
    """Return [{uid, name, label, how, surface}], best evidence first.

    how: 'alias' (found in the question text), 'exact' (a surface the router extracted),
         'ambiguous' (several entities share it), 'fuzzy' (spelling-tolerant guess).
    """
    run = run or db.run
    found, seen = [], set()

    def add(entity, how, surface):
        if entity["uid"] not in seen and len(found) < limit:
            seen.add(entity["uid"])
            found.append({"uid": entity["uid"], "name": entity["name"], "label": entity["label"],
                          "how": how, "surface": surface})

    mentions, _ = get_gazetteer(run).find(question)
    if mentions:
        names = {r["uid"]: r for r in run(NAMES_QUERY, uids=[m["uid"] for m in mentions])}
        for m in mentions:
            if m["uid"] in names:
                add(names[m["uid"]], "alias", m["uid"])

    for surface in surfaces or []:
        norm = normalize(surface)
        if not norm:
            continue
        rows = run(EXACT_QUERY, norm=norm)
        if rows:
            how = "ambiguous" if len(rows) > 1 or rows[0].get("ambiguous") else "exact"
            for row in rows:
                add(row, how, surface)
            continue
        q = fuzzy_query(surface)
        if q:
            for row in run(FUZZY_QUERY, q=q):
                add(row, "fuzzy", surface)
    return found
